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
from axman.core.heads.value_head import (
    DistanceToClosureValueHead,
    MultiHeadAttentionPooling,
    ValueOutput,
)
from axman.core.blocks.amb_backbone import AXMANetBackbone


def test_attention_pooling_shapes():
    d_model = 64
    num_heads = 4
    num_nodes = 12

    pooling = MultiHeadAttentionPooling(d_model=d_model, num_heads=num_heads)
    nodes = torch.randn(num_nodes, d_model)

    pooled, weights = pooling(nodes, return_weights=True)
    assert pooled.shape == (d_model,)
    assert weights.shape == (num_heads, 1, num_nodes)

    # Test batched
    batch_nodes = torch.randn(2, num_nodes, d_model)
    batch_pooled, batch_weights = pooling(batch_nodes, return_weights=True)
    assert batch_pooled.shape == (2, d_model)
    assert batch_weights.shape == (2, num_heads, 1, num_nodes)


def test_attention_pooling_masking():
    d_model = 32
    num_heads = 2
    num_nodes = 5

    pooling = MultiHeadAttentionPooling(d_model=d_model, num_heads=num_heads)
    nodes = torch.randn(num_nodes, d_model)

    # Mask out last 2 nodes
    mask = torch.tensor([True, True, True, False, False])
    pooled, weights = pooling(nodes, node_mask=mask, return_weights=True)

    # Attention weights on masked nodes must be zero
    assert torch.all(weights[:, :, 3:] < 1e-6)
    assert torch.all(weights[:, :, :3].sum(dim=-1) > 0.999)


def test_value_head_unbatched_output():
    d_model = 128
    num_heads = 8
    num_nodes = 15

    head = DistanceToClosureValueHead(d_model=d_model, num_heads=num_heads)
    nodes = torch.randn(num_nodes, d_model)

    out = head(nodes, return_pooling_weights=True)
    assert isinstance(out, ValueOutput)
    assert out.value_prob.shape == (1,)
    assert out.distance_to_qed.shape == (1,)
    assert out.graph_latent.shape == (d_model,)
    assert 0.0 <= out.value_prob.item() <= 1.0
    assert out.distance_to_qed.item() >= 0.0


def test_value_head_batched_output():
    d_model = 64
    num_heads = 4
    batch_size = 3
    num_nodes = 10

    head = DistanceToClosureValueHead(d_model=d_model, num_heads=num_heads)
    nodes = torch.randn(batch_size, num_nodes, d_model)

    out = head(nodes)
    assert out.value_prob.shape == (batch_size, 1)
    assert out.distance_to_qed.shape == (batch_size, 1)
    assert out.graph_latent.shape == (batch_size, d_model)
    assert torch.all(out.value_prob >= 0.0) and torch.all(out.value_prob <= 1.0)
    assert torch.all(out.distance_to_qed >= 0.0)


def test_value_head_masked_nodes():
    d_model = 64
    head = DistanceToClosureValueHead(d_model=d_model, num_heads=4)

    nodes = torch.randn(8, d_model)
    mask = torch.tensor([True, True, True, True, False, False, False, False])

    out = head(nodes, node_mask=mask, return_pooling_weights=True)
    assert out.pooling_weights is not None
    assert torch.all(out.pooling_weights[:, :, 4:] < 1e-6)


def test_value_head_from_config():
    config = AxmanConfig.base()
    head = DistanceToClosureValueHead.from_config(config)
    assert head.d_model == config.d_model
    assert head.pooling.num_heads == config.num_heads


def test_value_head_gradient_flow():
    d_model = 64
    head = DistanceToClosureValueHead(d_model=d_model, num_heads=4)

    nodes = torch.randn(6, d_model, requires_grad=True)
    target_value = torch.tensor([1.0])
    target_dist = torch.tensor([3.0])

    out = head(nodes)
    loss = F.binary_cross_entropy(out.value_prob, target_value) + F.mse_loss(out.distance_to_qed, target_dist)
    loss.backward()

    assert nodes.grad is not None
    for name, p in head.named_parameters():
        assert p.grad is not None, f"Parameter {name} has no gradient."


def test_value_head_e2e_with_backbone():
    config = AxmanConfig(
        d_model=64,
        num_layers=2,
        num_heads=4,
        port_dim=16,
    )
    backbone = AXMANetBackbone(config)
    value_head = DistanceToClosureValueHead.from_config(config)

    # 5 nodes, 2 edges
    edge_targets = torch.tensor([2, 4], dtype=torch.long)
    edge_offsets = torch.tensor([0, 2, 4], dtype=torch.long)
    edge_sources = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    edge_ports = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    kind_ids = torch.tensor([2, 2, 3, 2, 3], dtype=torch.long)

    backbone_out = backbone(
        edge_targets=edge_targets,
        edge_offsets=edge_offsets,
        edge_sources=edge_sources,
        edge_ports=edge_ports,
        kind_ids=kind_ids,
    )

    # Goal mask: e.g. nodes 2 and 4 are open goals (kind_ids == 3)
    goal_mask = (kind_ids == 3)

    out = value_head(backbone_out.node_embeddings, node_mask=goal_mask)
    assert out.value_prob.shape == (1,)
    assert out.distance_to_qed.shape == (1,)
    assert 0.0 <= out.value_prob.item() <= 1.0
    assert out.distance_to_qed.item() >= 0.0
