import sys
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.config import AxmanConfig
from axman.core.heads.diffusion_head import (
    FlowMatchingLossOutput,
    LatentInvariantDiffusionHead,
    SinusoidalTimeEmbedding,
    get_separate_optimizer_param_groups,
)
from axman.core.heads.policy_head import MorphismPolicyHead
from axman.core.heads.value_head import DistanceToClosureValueHead
from axman.core.blocks.amb_backbone import AXMANetBackbone


def test_sinusoidal_time_embedding():
    time_embed = SinusoidalTimeEmbedding(dim=64)
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    emb = time_embed(t)
    assert emb.shape == (5, 64)
    assert not torch.isnan(emb).any()


def test_diffusion_head_vector_field_shape():
    latent_dim = 64
    context_dim = 64
    batch_size = 4

    head = LatentInvariantDiffusionHead(
        latent_dim=latent_dim,
        context_dim=context_dim,
        num_blocks=2,
    )

    x_t = torch.randn(batch_size, latent_dim)
    t = torch.rand(batch_size)
    context = torch.randn(batch_size, context_dim)

    v_pred = head.forward_vector_field(x_t, t, context)
    assert v_pred.shape == (batch_size, latent_dim)

    # Unbatched test
    v_single = head.forward_vector_field(x_t[0], t[0], context[0])
    assert v_single.shape == (latent_dim,)


def test_diffusion_head_loss_computation():
    latent_dim = 64
    context_dim = 64
    batch_size = 4

    head = LatentInvariantDiffusionHead(
        latent_dim=latent_dim,
        context_dim=context_dim,
        num_blocks=2,
    )

    targets = torch.randn(batch_size, latent_dim)
    context = torch.randn(batch_size, context_dim)

    loss_out = head.compute_loss(targets, context)
    assert isinstance(loss_out, FlowMatchingLossOutput)
    assert loss_out.loss.dim() == 0  # scalar
    assert loss_out.loss.item() >= 0.0
    assert loss_out.v_pred.shape == (batch_size, latent_dim)
    assert loss_out.v_target.shape == (batch_size, latent_dim)


def test_diffusion_head_gradient_flow():
    latent_dim = 32
    head = LatentInvariantDiffusionHead(latent_dim=latent_dim, context_dim=latent_dim, num_blocks=1)

    targets = torch.randn(2, latent_dim, requires_grad=True)
    context = torch.randn(2, latent_dim, requires_grad=True)

    loss_out = head.compute_loss(targets, context)
    loss_out.loss.backward()

    assert targets.grad is not None
    assert context.grad is not None
    for name, p in head.named_parameters():
        assert p.grad is not None, f"Parameter {name} did not receive gradients."


@pytest.mark.parametrize("solver", ["euler", "midpoint", "rk4"])
def test_diffusion_head_sampling_solvers(solver):
    latent_dim = 32
    context_dim = 32
    batch_size = 2

    head = LatentInvariantDiffusionHead(
        latent_dim=latent_dim,
        context_dim=context_dim,
        num_blocks=2,
    )

    context = torch.randn(batch_size, context_dim)
    sample = head.sample_latent_invariant(
        context=context,
        num_steps=5,
        solver=solver,
    )
    assert sample.shape == (batch_size, latent_dim)
    assert not torch.isnan(sample).any()


def test_diffusion_head_unbatched_sampling():
    latent_dim = 32
    head = LatentInvariantDiffusionHead(latent_dim=latent_dim, context_dim=latent_dim, num_blocks=1)
    context = torch.randn(latent_dim)
    sample = head.sample_latent_invariant(context, num_steps=3, solver="midpoint")
    assert sample.shape == (latent_dim,)


def test_separate_optimizer_param_groups():
    config = AxmanConfig(d_model=32, num_layers=1, num_heads=2, port_dim=8)
    backbone = AXMANetBackbone(config)
    policy_head = MorphismPolicyHead.from_config(config, num_actions=8)
    value_head = DistanceToClosureValueHead.from_config(config)
    diffusion_head = LatentInvariantDiffusionHead.from_config(config)

    param_groups = get_separate_optimizer_param_groups(
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        diffusion_head=diffusion_head,
        lr_backbone=1e-4,
        lr_policy=1e-4,
        lr_value=1e-4,
        lr_flow=2e-4,
    )

    assert len(param_groups) == 4
    group_names = [g["name"] for g in param_groups]
    assert group_names == ["backbone", "policy_head", "value_head", "flow_head"]
    assert param_groups[0]["lr"] == 1e-4
    assert param_groups[3]["lr"] == 2e-4

    # Verify optimizer initialization with these groups
    optimizer = torch.optim.AdamW(param_groups)
    assert len(optimizer.param_groups) == 4


def test_e2e_full_phase2_model_heads_with_backbone():
    config = AxmanConfig(d_model=64, num_layers=2, num_heads=4, port_dim=16)
    backbone = AXMANetBackbone(config)
    policy_head = MorphismPolicyHead.from_config(config, num_actions=16)
    value_head = DistanceToClosureValueHead.from_config(config)
    diffusion_head = LatentInvariantDiffusionHead.from_config(config)

    # 4 nodes, 2 edges
    edge_targets = torch.tensor([2, 3], dtype=torch.long)
    edge_offsets = torch.tensor([0, 2, 3], dtype=torch.long)
    edge_sources = torch.tensor([0, 1, 2], dtype=torch.long)
    edge_ports = torch.tensor([0, 1, 0], dtype=torch.long)
    kind_ids = torch.tensor([2, 2, 3, 3], dtype=torch.long)

    backbone_out = backbone(
        edge_targets=edge_targets,
        edge_offsets=edge_offsets,
        edge_sources=edge_sources,
        edge_ports=edge_ports,
        kind_ids=kind_ids,
    )

    # 1. Value Head forward
    value_out = value_head(backbone_out.node_embeddings)
    assert 0.0 <= value_out.value_prob.item() <= 1.0

    # 2. Policy Head forward
    goal_emb = backbone_out.node_embeddings[3:4]
    candidates = backbone_out.node_embeddings
    policy_out = policy_head(goal_emb, candidates)
    assert policy_out.action_logits.shape == (1, 16)
    assert policy_out.premise_logits.shape == (1, 4)

    # 3. Diffusion Head sampling for lemma abduction conditioned on graph latent
    sampled_lemma = diffusion_head.sample_latent_invariant(
        context=value_out.graph_latent,
        num_steps=5,
        solver="midpoint",
    )
    assert sampled_lemma.shape == (config.d_model,)
