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
from axman.core.heads.policy_head import (
    MorphismPolicyHead,
    PolicyOutput,
    TacticConstructorKind,
)
from axman.core.blocks.amb_backbone import AXMANetBackbone


def test_tactic_constructor_kind_enum():
    assert TacticConstructorKind.INTRO == 0
    assert TacticConstructorKind.EXACT == 1
    assert TacticConstructorKind.APPLY == 2
    assert TacticConstructorKind.CUSTOM == 16
    assert len(TacticConstructorKind) >= 17


def test_policy_head_shapes():
    d_model = 128
    num_actions = 24
    num_goals = 3
    num_candidates = 10

    head = MorphismPolicyHead(
        d_model=d_model,
        num_actions=num_actions,
        scoring_mode="bilinear",
    )

    goal_embeddings = torch.randn(num_goals, d_model)
    candidate_embeddings = torch.randn(num_candidates, d_model)

    out = head(goal_embeddings, candidate_embeddings)
    assert isinstance(out, PolicyOutput)
    assert out.action_logits.shape == (num_goals, num_actions)
    assert out.premise_logits.shape == (num_goals, num_candidates)


def test_policy_head_dot_product_mode():
    d_model = 64
    num_actions = 16
    num_goals = 2
    num_candidates = 8

    head = MorphismPolicyHead(
        d_model=d_model,
        num_actions=num_actions,
        scoring_mode="dot_product",
    )

    goal_embeddings = torch.randn(num_goals, d_model)
    candidate_embeddings = torch.randn(num_candidates, d_model)

    out = head(goal_embeddings, candidate_embeddings)
    assert out.action_logits.shape == (num_goals, num_actions)
    assert out.premise_logits.shape == (num_goals, num_candidates)


def test_policy_head_action_masking():
    d_model = 64
    num_actions = 8
    num_goals = 2
    num_candidates = 5

    head = MorphismPolicyHead(
        d_model=d_model,
        num_actions=num_actions,
    )

    goal_embeddings = torch.randn(num_goals, d_model)
    candidate_embeddings = torch.randn(num_candidates, d_model)

    # Allow only actions 0, 1, 3
    mask = torch.zeros(num_goals, num_actions, dtype=torch.bool)
    mask[:, [0, 1, 3]] = True

    out = head(goal_embeddings, candidate_embeddings, action_mask=mask)
    probs = F.softmax(out.action_logits, dim=-1)

    # Masked positions should have virtually 0 probability
    for a in [2, 4, 5, 6, 7]:
        assert torch.all(probs[:, a] < 1e-6)
    assert torch.all(probs[:, [0, 1, 3]].sum(dim=-1) > 0.99)


def test_policy_head_candidate_masking():
    d_model = 64
    num_actions = 4
    num_goals = 1
    num_candidates = 6

    head = MorphismPolicyHead(
        d_model=d_model,
        num_actions=num_actions,
    )

    goal_embeddings = torch.randn(num_goals, d_model)
    candidate_embeddings = torch.randn(num_candidates, d_model)

    # Mask out candidate 4 and 5
    cand_mask = torch.tensor([[True, True, True, True, False, False]])

    out = head(goal_embeddings, candidate_embeddings, candidate_mask=cand_mask)
    probs = F.softmax(out.premise_logits, dim=-1)

    assert probs[0, 4] < 1e-6
    assert probs[0, 5] < 1e-6
    assert probs[0, :4].sum() > 0.99


def test_policy_head_from_config():
    config = AxmanConfig.small()
    head = MorphismPolicyHead.from_config(config, num_actions=32)
    assert head.d_model == config.d_model
    assert head.num_actions == 32


def test_policy_head_backpropagation():
    d_model = 64
    num_actions = 10
    head = MorphismPolicyHead(d_model=d_model, num_actions=num_actions)

    goal = torch.randn(2, d_model, requires_grad=True)
    candidates = torch.randn(5, d_model, requires_grad=True)

    out = head(goal, candidates)
    loss = out.action_logits.sum() + out.premise_logits.sum()
    loss.backward()

    assert goal.grad is not None
    assert candidates.grad is not None
    for p in head.parameters():
        assert p.grad is not None


def test_policy_head_e2e_with_backbone():
    config = AxmanConfig(
        d_model=64,
        num_layers=2,
        num_heads=4,
        port_dim=16,
    )
    backbone = AXMANetBackbone(config)
    head = MorphismPolicyHead.from_config(config, num_actions=16)

    # Construct hypergraph: 4 nodes, 2 edges
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

    # Goal is node 3 (e.g. open goal), candidate premises are all nodes 0..3
    goal_emb = backbone_out.node_embeddings[3:4]
    candidate_emb = backbone_out.node_embeddings

    policy_out = head(goal_emb, candidate_emb)
    assert policy_out.action_logits.shape == (1, 16)
    assert policy_out.premise_logits.shape == (1, 4)
