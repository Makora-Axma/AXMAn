"""
Unit and integration tests for AXMANet Phase 6.2: Multi-task Training Procedure,
GradNorm, PCGrad, and AXMANetTrainer.
Tests axman.core.training and axman/train.py.
"""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.config import AxmanConfig
from axman.core.data import (
    HypergraphSample,
    HypergraphDataset,
    create_hypergraph_dataloader,
)
from axman.core.heads.policy_head import TacticConstructorKind
from axman.core.training import (
    GradNormLossBalancer,
    PCGrad,
    AXMANetModel,
    ModelOutput,
    TaskLosses,
    TrainingConfig,
    AXMANetTrainer,
)


def _create_sample_hypergraph(
    num_nodes: int = 6,
    num_edges: int = 3,
    feat_dim: int = 64,
    seed: int = 42,
) -> HypergraphSample:
    """Helper to generate a valid synthetic HypergraphSample."""
    rng = np.random.RandomState(seed)

    node_features = rng.randn(num_nodes, feat_dim).astype(np.float32)

    sources = []
    ports = []
    offsets = [0]
    targets = []

    for _ in range(num_edges):
        arity = rng.randint(1, min(num_nodes, 4) + 1)
        src_nodes = rng.choice(num_nodes, size=arity, replace=False)
        tgt_node = rng.choice(num_nodes)

        sources.extend(src_nodes)
        ports.extend(list(range(arity)))
        targets.append(tgt_node)
        offsets.append(offsets[-1] + arity)

    target_action = rng.randint(0, len(TacticConstructorKind))
    target_premises = rng.choice(num_nodes, size=min(2, num_nodes), replace=False).tolist()
    target_value = float(rng.rand())
    target_distance = float(rng.randint(1, 10))
    target_latent = rng.randn(feat_dim).astype(np.float32)
    open_goal_mask = (rng.rand(num_nodes) > 0.5).astype(bool)

    return HypergraphSample(
        node_features=node_features,
        edge_sources=np.array(sources, dtype=np.int64),
        edge_ports=np.array(ports, dtype=np.int64),
        edge_targets=np.array(targets, dtype=np.int64),
        edge_offsets=np.array(offsets, dtype=np.int64),
        target_action=target_action,
        target_premises=target_premises,
        target_value=target_value,
        target_distance=target_distance,
        target_latent=target_latent,
        open_goal_mask=open_goal_mask,
    )


class TestLossBalancers(unittest.TestCase):
    """Test GradNorm and PCGrad multi-task balancing mechanisms."""

    def test_gradnorm_initialization_and_weights(self):
        balancer = GradNormLossBalancer(
            task_names=["policy", "premise", "value", "flow"],
            alpha=0.16,
            lr_weights=0.01,
        )
        weights = balancer.get_weights_dict()
        self.assertEqual(len(weights), 4)
        self.assertAlmostEqual(sum(weights.values()), 4.0, places=4)

        # Mock losses
        losses = {
            "policy": torch.tensor(2.5, requires_grad=True),
            "premise": torch.tensor(3.0, requires_grad=True),
            "value": torch.tensor(0.8, requires_grad=True),
            "flow": torch.tensor(1.2, requires_grad=True),
        }

        weighted_loss = balancer.compute_weighted_loss(losses)
        self.assertGreater(weighted_loss.item(), 0.0)

        # Step GradNorm with mock shared parameter
        shared_param = nn.Parameter(torch.randn(32, 32))
        # Compute task loss connected to shared_param
        connected_losses = {
            "policy": (losses["policy"] * shared_param.sum()),
            "premise": (losses["premise"] * shared_param.sum()),
            "value": (losses["value"] * shared_param.sum()),
            "flow": (losses["flow"] * shared_param.sum()),
        }

        updated_weights = balancer.step(connected_losses, shared_param)
        self.assertAlmostEqual(sum(updated_weights.values()), 4.0, places=3)

    def test_pcgrad_projection(self):
        # Two conflicting tasks
        param = nn.Parameter(torch.tensor([1.0, 1.0]))
        optimizer = torch.optim.SGD([param], lr=0.1)
        pcgrad = PCGrad(optimizer)

        # Conflicting loss functions: loss1 wants param > 0, loss2 wants param < 0
        loss1 = 2.0 * param[0] + 0.0 * param[1]
        loss2 = -1.0 * param[0] + 1.0 * param[1]

        pcgrad.zero_grad()
        pcgrad.pcgrad_backward([loss1, loss2])

        # Dot product of task grads would normally be: [2, 0] . [-1, 1] = -2 (conflict)
        # PCGrad must project so that dot product is >= 0
        grad = param.grad
        self.assertIsNotNone(grad)
        self.assertFalse(torch.isnan(grad).any())


class TestAXMANetModelAndLosses(unittest.TestCase):
    """Test composite AXMANetModel forward and multi-task loss calculation."""

    def setUp(self):
        self.config = AxmanConfig.small(
            d_model=64,
            in_features=64,
            num_layers=2,
            num_heads=4,
            port_dim=16,
        )
        self.model = AXMANetModel(self.config)
        self.samples = [_create_sample_hypergraph(feat_dim=64, seed=i) for i in range(4)]
        self.loader = create_hypergraph_dataloader(HypergraphDataset(self.samples), batch_size=4)

    def test_model_forward_and_output(self):
        b_graph, b_targets = next(iter(self.loader))
        out: ModelOutput = self.model(
            batched_graph=b_graph,
            open_goal_mask=b_targets.open_goal_mask,
            sample_lemmas=True,
            diffusion_steps=3,
        )

        self.assertEqual(out.node_embeddings.shape, (b_graph.total_nodes, 64))
        self.assertEqual(out.action_logits.shape, (4, len(TacticConstructorKind)))
        self.assertEqual(out.closure_probabilities.shape, (4, 1))
        self.assertEqual(out.topological_distances.shape, (4, 1))
        self.assertEqual(out.pooled_context.shape, (4, 64))
        self.assertEqual(len(out.subgraph_premise_logits), 4)
        self.assertIsNotNone(out.sampled_lemma_invariants)
        self.assertEqual(out.sampled_lemma_invariants.shape, (4, 64))

    def test_model_compute_losses(self):
        b_graph, b_targets = next(iter(self.loader))
        losses: TaskLosses = self.model.compute_losses(b_graph, b_targets)

        self.assertGreater(losses.total_loss.item(), 0.0)
        self.assertGreater(losses.policy_loss.item(), 0.0)
        self.assertGreater(losses.premise_loss.item(), 0.0)
        self.assertGreater(losses.value_loss.item(), 0.0)
        self.assertGreater(losses.flow_loss.item(), 0.0)

        # Verify metrics dict
        self.assertIn("policy_acc", losses.metrics)
        self.assertIn("premise_top1", losses.metrics)
        self.assertIn("value_closure_mse", losses.metrics)


class TestAXMANetTrainer(unittest.TestCase):
    """Test AXMANetTrainer training loop, balancers, and checkpointing."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = AxmanConfig.small(
            d_model=32,
            in_features=32,
            num_layers=2,
            num_heads=2,
            port_dim=8,
        )
        self.model = AXMANetModel(self.config)
        self.samples = [_create_sample_hypergraph(feat_dim=32, seed=10 + i) for i in range(8)]
        self.train_loader = create_hypergraph_dataloader(HypergraphDataset(self.samples), batch_size=4, shuffle=True)
        self.val_loader = create_hypergraph_dataloader(HypergraphDataset(self.samples[:4]), batch_size=4, shuffle=False)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_training_loop_gradnorm(self):
        train_cfg = TrainingConfig(
            epochs=2,
            batch_size=4,
            lr_backbone=1e-3,
            balancer_type="gradnorm",
            output_dir=self.temp_dir,
            save_every=1,
            eval_every=1,
        )
        trainer = AXMANetTrainer(
            model=self.model,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            train_config=train_cfg,
        )

        history = trainer.train()
        self.assertEqual(len(history), 2)
        self.assertIsNotNone(history[0].train_total_loss)
        self.assertIsNotNone(history[0].val_total_loss)

        # Check saved files
        ckpt_pt = Path(self.temp_dir) / "axman_model_latest.pt"
        ckpt_st = Path(self.temp_dir) / "axman_model_latest.safetensors"
        self.assertTrue(ckpt_pt.exists())
        self.assertTrue(ckpt_st.exists())
        self.assertGreater(ckpt_pt.stat().st_size, 1000)
        self.assertGreater(ckpt_st.stat().st_size, 1000)

    def test_training_loop_pcgrad(self):
        train_cfg = TrainingConfig(
            epochs=1,
            batch_size=4,
            balancer_type="pcgrad",
            output_dir=self.temp_dir,
        )
        trainer = AXMANetTrainer(
            model=self.model,
            train_loader=self.train_loader,
            train_config=train_cfg,
        )
        history = trainer.train()
        self.assertEqual(len(history), 1)
        self.assertGreater(history[0].train_total_loss, 0.0)


class TestTrainCLI(unittest.TestCase):
    """Test top-level train.py command-line execution."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_train_cli_synthetic_run(self):
        cmd = [
            sys.executable,
            str(Path(axman_root) / "axman" / "train.py"),
            "--config", "small",
            "--synthetic-samples", "8",
            "--batch-size", "4",
            "--epochs", "1",
            "--balancer", "static",
            "--output-dir", str(self.temp_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"train.py failed with:\n{res.stderr}\n{res.stdout}")
        self.assertTrue((Path(self.temp_dir) / "axman_model_latest.pt").exists())
        self.assertTrue((Path(self.temp_dir) / "axman_model_latest.safetensors").exists())

    def test_train_cli_gradnorm_run(self):
        cmd = [
            sys.executable,
            str(Path(axman_root) / "axman" / "train.py"),
            "--config", "small",
            "--synthetic-samples", "4",
            "--batch-size", "2",
            "--epochs", "1",
            "--balancer", "gradnorm",
            "--output-dir", str(self.temp_dir),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"train.py failed with:\n{res.stderr}\n{res.stdout}")


class TestEdgeCasesAndRobustness(unittest.TestCase):
    """Test edge cases: zero/unlabeled losses in PCGrad, GradNorm detachment, goal mask variations."""

    def test_pcgrad_unlabeled_losses(self):
        param = nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.SGD([param], lr=0.1)
        pcgrad = PCGrad(optimizer)

        # loss1 has grad, loss2 has no grad (e.g. constant 0.0 without grad)
        loss1 = 3.0 * param[0]
        loss2 = torch.tensor(0.0, requires_grad=False)

        pcgrad.zero_grad()
        pcgrad.pcgrad_backward([loss1, loss2])
        self.assertIsNotNone(param.grad)
        self.assertEqual(param.grad[0].item(), 3.0)
        self.assertEqual(param.grad[1].item(), 0.0)

    def test_gradnorm_weights_not_in_backward_graph(self):
        balancer = GradNormLossBalancer(task_names=["t1", "t2"])
        param = nn.Parameter(torch.tensor([2.0, 3.0]))

        loss1 = param[0] ** 2
        loss2 = param[1] ** 2

        losses = {"t1": loss1, "t2": loss2}
        weighted_loss = balancer.compute_weighted_loss(losses)

        # Ensure balancer.weights does NOT have grad after weighted_loss.backward()
        weighted_loss.backward()
        self.assertIsNone(balancer.weights.grad)
        self.assertIsNotNone(param.grad)

    def test_forward_with_no_open_goals(self):
        config = AxmanConfig.small(d_model=32, in_features=32, num_layers=2, num_heads=2, port_dim=8)
        model = AXMANetModel(config)
        samples = [_create_sample_hypergraph(feat_dim=32, seed=100 + i) for i in range(2)]
        for s in samples:
            s.open_goal_mask = np.zeros(len(s.node_features), dtype=bool)
        loader = create_hypergraph_dataloader(HypergraphDataset(samples), batch_size=2)
        b_graph, b_targets = next(iter(loader))

        out = model(b_graph, open_goal_mask=b_targets.open_goal_mask, sample_lemmas=False)
        self.assertEqual(out.pooled_context.shape, (2, 32))

    def test_lr_scheduler_clamping(self):
        config = AxmanConfig.small(d_model=32, in_features=32, num_layers=2, num_heads=2, port_dim=8)
        model = AXMANetModel(config)
        samples = [_create_sample_hypergraph(feat_dim=32, seed=200 + i) for i in range(4)]
        loader = create_hypergraph_dataloader(HypergraphDataset(samples), batch_size=2)
        train_cfg = TrainingConfig(epochs=2, batch_size=2, warmup_steps=2, lr_backbone=1e-3)
        trainer = AXMANetTrainer(model, loader, train_config=train_cfg)

        # Total steps = 2 epochs * 2 batches = 4 steps
        self.assertEqual(trainer.total_steps, 4)

        # Test warmup at step 1
        trainer._adjust_learning_rate(1)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 0.5 * 1e-3)

        # Test step beyond total_steps (clamping)
        trainer._adjust_learning_rate(10)
        self.assertAlmostEqual(trainer.optimizer.param_groups[0]["lr"], 1e-5)  # 1e-3 * max(alpha, 1e-2) where alpha=0.0



if __name__ == "__main__":
    unittest.main()
