"""
Unit tests for AXMANet Phase 6.1: Autonomous Hypergraph Dataset, Memory-Mapped I/O,
and Topological Augmentations.
Tests axman.core.data module.
"""

from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

# Ensure axman is in python path
axman_root = Path(__file__).resolve().parent.parent.parent
if str(axman_root) not in sys.path:
    sys.path.insert(0, str(axman_root))

from axman.core.config import AxmanConfig
from axman.core.blocks.amb_backbone import AXMANetBackbone
from axman.core.heads.policy_head import MorphismPolicyHead, TacticConstructorKind
from axman.core.heads.value_head import DistanceToClosureValueHead
from axman.core.heads.diffusion_head import LatentInvariantDiffusionHead
from axman.core.data import (
    HypergraphSample,
    BatchedTargets,
    TopologicalAugmentor,
    BinaryDumpReader,
    ArrowDatasetReader,
    save_hypergraphs_to_binary_dump,
    save_hypergraphs_to_arrow,
    HypergraphDataset,
    collate_hypergraph_samples,
    create_hypergraph_dataloader,
)


def _create_sample_hypergraph(
    num_nodes: int = 5,
    num_edges: int = 3,
    feat_dim: int = 64,
    seed: int = 42,
) -> HypergraphSample:
    """Helper to generate a valid synthetic HypergraphSample."""
    rng = np.random.RandomState(seed)

    # Node features
    node_features = rng.randn(num_nodes, feat_dim).astype(np.float32)

    # Build CSR hyperedges
    sources = []
    ports = []
    offsets = [0]
    targets = []

    for e in range(num_edges):
        arity = rng.randint(1, min(num_nodes, 4) + 1)
        src_nodes = rng.choice(num_nodes, size=arity, replace=False)
        target_node = rng.choice(num_nodes)

        sources.extend(src_nodes)
        ports.extend(list(range(arity)))
        targets.append(target_node)
        offsets.append(offsets[-1] + arity)

    edge_sources = np.array(sources, dtype=np.int64)
    edge_ports = np.array(ports, dtype=np.int64)
    edge_targets = np.array(targets, dtype=np.int64)
    edge_offsets = np.array(offsets, dtype=np.int64)

    target_action = rng.randint(0, len(TacticConstructorKind))
    target_premises = rng.choice(num_nodes, size=min(2, num_nodes), replace=False).tolist()
    target_value = float(rng.rand())
    target_distance = float(rng.randint(1, 15))
    target_latent = rng.randn(feat_dim).astype(np.float32)
    open_goal_mask = (rng.rand(num_nodes) > 0.6).astype(bool)

    metadata = {"proof_id": f"thm_{seed}", "step": seed}

    return HypergraphSample(
        node_features=node_features,
        edge_sources=edge_sources,
        edge_ports=edge_ports,
        edge_targets=edge_targets,
        edge_offsets=edge_offsets,
        target_action=target_action,
        target_premises=target_premises,
        target_value=target_value,
        target_distance=target_distance,
        target_latent=target_latent,
        open_goal_mask=open_goal_mask,
        metadata=metadata,
    )


class TestHypergraphSample(unittest.TestCase):
    """Test HypergraphSample basic invariants and dictionary serialization."""

    def test_sample_properties_and_dict(self):
        sample = _create_sample_hypergraph(num_nodes=6, num_edges=4, feat_dim=32, seed=10)
        self.assertEqual(sample.num_nodes, 6)
        self.assertEqual(sample.num_edges, 4)
        self.assertGreater(sample.num_sources, 0)

        d = sample.to_dict()
        self.assertIn("node_features", d)
        self.assertIn("edge_offsets", d)
        self.assertEqual(d["metadata"]["proof_id"], "thm_10")

        # Roundtrip from_dict
        restored = HypergraphSample.from_dict(d)
        self.assertEqual(restored.num_nodes, sample.num_nodes)
        self.assertEqual(restored.num_edges, sample.num_edges)
        np.testing.assert_allclose(restored.node_features, sample.node_features)
        np.testing.assert_array_equal(restored.edge_sources, sample.edge_sources)
        np.testing.assert_array_equal(restored.edge_offsets, sample.edge_offsets)


class TestTopologicalAugmentor(unittest.TestCase):
    """Test topological data augmentations preserving categorical semantics."""

    def test_isomorphic_node_permutation(self):
        sample = _create_sample_hypergraph(num_nodes=7, num_edges=4, feat_dim=16, seed=20)
        augmentor = TopologicalAugmentor(
            p_permute=1.0,
            p_jitter=0.0,
            p_edge_dropout=0.0,
            rng=np.random.RandomState(123),
        )

        augmented = augmentor(sample)

        # Invariants
        self.assertEqual(augmented.num_nodes, sample.num_nodes)
        self.assertEqual(augmented.num_edges, sample.num_edges)
        self.assertEqual(augmented.num_sources, sample.num_sources)
        self.assertEqual(len(augmented.edge_offsets), len(sample.edge_offsets))

        # Check that nodes are permuted (features are row-permuted)
        diff = np.abs(augmented.node_features - sample.node_features).sum()
        self.assertGreater(diff, 1e-4, "Features should be permuted to different rows.")

        # Check multiset equality of node features (same set of feature vectors)
        orig_norms = np.sort(np.linalg.norm(sample.node_features, axis=-1))
        aug_norms = np.sort(np.linalg.norm(augmented.node_features, axis=-1))
        np.testing.assert_allclose(orig_norms, aug_norms, rtol=1e-5)

        # Port indices and CSR offset lengths must remain invariant
        np.testing.assert_array_equal(augmented.edge_ports, sample.edge_ports)
        np.testing.assert_array_equal(augmented.edge_offsets, sample.edge_offsets)

    def test_feature_jitter(self):
        sample = _create_sample_hypergraph(num_nodes=5, num_edges=3, feat_dim=16, seed=30)
        augmentor = TopologicalAugmentor(
            p_permute=0.0,
            p_jitter=1.0,
            jitter_std=0.05,
            p_edge_dropout=0.0,
            rng=np.random.RandomState(456),
        )

        augmented = augmentor(sample)
        # Graph structure remains identical
        np.testing.assert_array_equal(augmented.edge_sources, sample.edge_sources)
        np.testing.assert_array_equal(augmented.edge_targets, sample.edge_targets)
        # Features have small jitter
        diff = np.abs(augmented.node_features - sample.node_features)
        self.assertGreater(np.mean(diff), 0.01)
        self.assertLess(np.max(diff), 0.5)

    def test_edge_dropout(self):
        sample = _create_sample_hypergraph(num_nodes=8, num_edges=6, feat_dim=16, seed=40)
        augmentor = TopologicalAugmentor(
            p_permute=0.0,
            p_jitter=0.0,
            p_edge_dropout=1.0,
            edge_dropout_rate=0.4,
            rng=np.random.RandomState(789),
        )

        augmented = augmentor(sample)
        self.assertLessEqual(augmented.num_edges, sample.num_edges)
        self.assertGreaterEqual(augmented.num_edges, 1)
        # CSR invariant: offsets start with 0 and monotonically increase
        self.assertEqual(augmented.edge_offsets[0], 0)
        self.assertTrue(np.all(np.diff(augmented.edge_offsets) >= 0))
        self.assertEqual(augmented.edge_offsets[-1], augmented.num_sources)

    def test_torch_tensor_roundtrip(self):
        sample = _create_sample_hypergraph(num_nodes=6, num_edges=3, feat_dim=16, seed=45)
        # Convert all to torch tensors
        sample_torch = HypergraphSample(
            node_features=torch.from_numpy(sample.node_features),
            edge_sources=torch.from_numpy(sample.edge_sources),
            edge_ports=torch.from_numpy(sample.edge_ports),
            edge_targets=torch.from_numpy(sample.edge_targets),
            edge_offsets=torch.from_numpy(sample.edge_offsets),
            target_action=sample.target_action,
            target_premises=torch.tensor(sample.target_premises, dtype=torch.long),
            target_value=sample.target_value,
            target_distance=sample.target_distance,
            target_latent=torch.from_numpy(sample.target_latent),
            open_goal_mask=torch.from_numpy(sample.open_goal_mask),
            metadata=sample.metadata,
        )

        augmentor = TopologicalAugmentor(
            p_permute=1.0,
            p_jitter=1.0,
            jitter_std=0.01,
            p_edge_dropout=0.0,
            rng=np.random.RandomState(999),
        )
        augmented = augmentor(sample_torch)

        self.assertIsInstance(augmented.node_features, torch.Tensor)
        self.assertIsInstance(augmented.edge_sources, torch.Tensor)
        self.assertIsInstance(augmented.edge_targets, torch.Tensor)
        self.assertIsInstance(augmented.open_goal_mask, torch.Tensor)
        self.assertEqual(augmented.node_features.shape, sample_torch.node_features.shape)

    def test_discrete_integer_features(self):
        # Node features are discrete primitive type IDs [N]
        num_nodes = 5
        node_ids = np.array([101, 202, 303, 404, 505], dtype=np.int64)
        sample = HypergraphSample(
            node_features=node_ids,
            edge_sources=np.array([0, 1], dtype=np.int64),
            edge_ports=np.array([0, 1], dtype=np.int64),
            edge_targets=np.array([2], dtype=np.int64),
            edge_offsets=np.array([0, 2], dtype=np.int64),
        )

        augmentor = TopologicalAugmentor(
            p_permute=1.0,
            p_jitter=1.0,  # Should NOT corrupt integer IDs with noise
            jitter_std=0.5,
            rng=np.random.RandomState(321),
        )
        augmented = augmentor(sample)
        # Verify node features are a permutation of original integer IDs without decimal noise
        self.assertTrue(np.issubdtype(augmented.node_features.dtype, np.integer))
        self.assertEqual(set(augmented.node_features.tolist()), set(node_ids.tolist()))


class TestBinaryDumpStorage(unittest.TestCase):
    """Test native zero-copy memory-mapped binary dump (.axmb) format."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_binary_dump_roundtrip_mmap(self):
        dump_path = Path(self.temp_dir) / "test_dump.axmb"
        samples = [
            _create_sample_hypergraph(num_nodes=4 + i, num_edges=2 + i, feat_dim=32, seed=100 + i)
            for i in range(5)
        ]

        count = save_hypergraphs_to_binary_dump(samples, dump_path)
        self.assertEqual(count, 5)
        self.assertTrue(dump_path.exists())

        reader = BinaryDumpReader(dump_path)
        self.assertEqual(len(reader), 5)

        for i, expected in enumerate(samples):
            loaded = reader[i]
            self.assertEqual(loaded.num_nodes, expected.num_nodes)
            self.assertEqual(loaded.num_edges, expected.num_edges)
            self.assertEqual(loaded.num_sources, expected.num_sources)
            np.testing.assert_allclose(loaded.node_features, expected.node_features, rtol=1e-5)
            np.testing.assert_array_equal(loaded.edge_sources, expected.edge_sources)
            np.testing.assert_array_equal(loaded.edge_ports, expected.edge_ports)
            np.testing.assert_array_equal(loaded.edge_targets, expected.edge_targets)
            np.testing.assert_array_equal(loaded.edge_offsets, expected.edge_offsets)
            self.assertEqual(loaded.target_action, expected.target_action)
            self.assertAlmostEqual(loaded.target_value, expected.target_value, places=5)
            self.assertAlmostEqual(loaded.target_distance, expected.target_distance, places=5)
            np.testing.assert_allclose(loaded.target_latent, expected.target_latent, rtol=1e-5)
            np.testing.assert_array_equal(loaded.open_goal_mask, expected.open_goal_mask)
            self.assertEqual(loaded.metadata, expected.metadata)

        reader.close()

    def test_binary_dump_edge_cases_and_null_values(self):
        """Test binary dump roundtrip when optional fields are None or all-false."""
        dump_path = Path(self.temp_dir) / "test_null_dump.axmb"

        s1 = HypergraphSample(
            node_features=np.ones((4, 16), dtype=np.float32),
            edge_sources=np.array([0, 1], dtype=np.int64),
            edge_ports=np.array([0, 1], dtype=np.int64),
            edge_targets=np.array([2], dtype=np.int64),
            edge_offsets=np.array([0, 2], dtype=np.int64),
            target_action=None,
            target_premises=None,
            target_value=None,
            target_distance=None,
            target_latent=None,
            open_goal_mask=None,
            metadata=None,
        )

        s2 = HypergraphSample(
            node_features=np.zeros((3, 16), dtype=np.float32),
            edge_sources=np.empty(0, dtype=np.int64),
            edge_ports=np.empty(0, dtype=np.int64),
            edge_targets=np.empty(0, dtype=np.int64),
            edge_offsets=np.array([0], dtype=np.int64),
            open_goal_mask=np.array([False, False, False]),
            metadata={"source": "test"},
        )

        save_hypergraphs_to_binary_dump([s1, s2], dump_path)
        reader = BinaryDumpReader(dump_path)
        self.assertEqual(len(reader), 2)

        loaded_s1 = reader[0]
        self.assertIsNone(loaded_s1.target_action)
        self.assertIsNone(loaded_s1.target_premises)
        self.assertIsNone(loaded_s1.target_value)
        self.assertIsNone(loaded_s1.target_latent)
        self.assertIsNone(loaded_s1.open_goal_mask)
        self.assertIsNone(loaded_s1.metadata)

        loaded_s2 = reader[1]
        self.assertIsNotNone(loaded_s2.open_goal_mask)
        np.testing.assert_array_equal(loaded_s2.open_goal_mask, np.array([False, False, False]))
        self.assertEqual(loaded_s2.metadata, {"source": "test"})
        reader.close()


class TestArrowStorage(unittest.TestCase):
    """Test Apache Arrow IPC / Feather memory-mapped dataset format."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_arrow_roundtrip_mmap(self):
        arrow_path = Path(self.temp_dir) / "test_data.arrow"
        samples = [
            _create_sample_hypergraph(num_nodes=5 + i, num_edges=3 + i, feat_dim=32, seed=200 + i)
            for i in range(4)
        ]

        count = save_hypergraphs_to_arrow(samples, arrow_path)
        self.assertEqual(count, 4)
        self.assertTrue(arrow_path.exists())

        reader = ArrowDatasetReader(arrow_path)
        self.assertEqual(len(reader), 4)

        for i, expected in enumerate(samples):
            loaded = reader[i]
            self.assertEqual(loaded.num_nodes, expected.num_nodes)
            self.assertEqual(loaded.num_edges, expected.num_edges)
            np.testing.assert_allclose(loaded.node_features, expected.node_features, rtol=1e-5)
            np.testing.assert_array_equal(loaded.edge_sources, expected.edge_sources)
            np.testing.assert_array_equal(loaded.edge_ports, expected.edge_ports)
            np.testing.assert_array_equal(loaded.edge_targets, expected.edge_targets)
            np.testing.assert_array_equal(loaded.edge_offsets, expected.edge_offsets)
            self.assertEqual(loaded.target_action, expected.target_action)
            self.assertAlmostEqual(loaded.target_value, expected.target_value, places=5)
            self.assertAlmostEqual(loaded.target_distance, expected.target_distance, places=5)
            np.testing.assert_allclose(loaded.target_latent, expected.target_latent, rtol=1e-5)
            np.testing.assert_array_equal(loaded.open_goal_mask, expected.open_goal_mask)
            self.assertEqual(loaded.metadata, expected.metadata)

        reader.close()

    def test_arrow_edge_cases_and_null_values(self):
        """Test Arrow roundtrip when optional fields are None or all-false."""
        arrow_path = Path(self.temp_dir) / "test_null_data.arrow"

        s1 = HypergraphSample(
            node_features=np.ones((4, 16), dtype=np.float32),
            edge_sources=np.array([0, 1], dtype=np.int64),
            edge_ports=np.array([0, 1], dtype=np.int64),
            edge_targets=np.array([2], dtype=np.int64),
            edge_offsets=np.array([0, 2], dtype=np.int64),
            target_action=None,
            target_premises=None,
            target_value=None,
            target_distance=None,
            target_latent=None,
            open_goal_mask=None,
            metadata=None,
        )

        s2 = HypergraphSample(
            node_features=np.zeros((3, 16), dtype=np.float32),
            edge_sources=np.empty(0, dtype=np.int64),
            edge_ports=np.empty(0, dtype=np.int64),
            edge_targets=np.empty(0, dtype=np.int64),
            edge_offsets=np.array([0], dtype=np.int64),
            open_goal_mask=np.array([False, False, False]),
            metadata={"source": "test_arrow"},
        )

        save_hypergraphs_to_arrow([s1, s2], arrow_path)
        reader = ArrowDatasetReader(arrow_path)
        self.assertEqual(len(reader), 2)

        loaded_s1 = reader[0]
        self.assertIsNone(loaded_s1.target_action)
        self.assertIsNone(loaded_s1.target_premises)
        self.assertIsNone(loaded_s1.target_value)
        self.assertIsNone(loaded_s1.target_latent)
        self.assertIsNone(loaded_s1.open_goal_mask)
        self.assertIsNone(loaded_s1.metadata)

        loaded_s2 = reader[1]
        self.assertIsNotNone(loaded_s2.open_goal_mask)
        np.testing.assert_array_equal(loaded_s2.open_goal_mask, np.array([False, False, False]))
        self.assertEqual(loaded_s2.metadata, {"source": "test_arrow"})
        reader.close()


class TestHypergraphDatasetAndDataLoader(unittest.TestCase):
    """Test PyTorch Dataset and DataLoader integration with continuous batching."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.samples = [
            _create_sample_hypergraph(num_nodes=4 + (i % 3), num_edges=2 + (i % 2), feat_dim=32, seed=300 + i)
            for i in range(12)
        ]

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_in_memory(self):
        ds = HypergraphDataset(self.samples)
        self.assertEqual(len(ds), 12)
        s0 = ds[0]
        self.assertEqual(s0.num_nodes, self.samples[0].num_nodes)

    def test_collate_hypergraph_samples(self):
        batch_slice = self.samples[:4]
        batched_graph, batched_targets = collate_hypergraph_samples(batch_slice)

        # Check BatchedHypergraph properties
        self.assertEqual(batched_graph.num_graphs, 4)
        expected_total_nodes = sum(s.num_nodes for s in batch_slice)
        expected_total_edges = sum(s.num_edges for s in batch_slice)
        expected_total_sources = sum(s.num_sources for s in batch_slice)

        self.assertEqual(batched_graph.total_nodes, expected_total_nodes)
        self.assertEqual(batched_graph.total_edges, expected_total_edges)
        self.assertEqual(batched_graph.total_sources, expected_total_sources)

        # Check target collation
        self.assertEqual(batched_targets.target_actions.shape, (4,))
        self.assertEqual(batched_targets.target_values.shape, (4, 1))
        self.assertEqual(batched_targets.target_distances.shape, (4, 1))
        self.assertEqual(batched_targets.target_latents.shape, (4, 32))
        self.assertEqual(batched_targets.open_goal_mask.shape, (expected_total_nodes,))

        # Verify remapped premise indices match graph_node_ptr offsets
        node_ptr = batched_graph.graph_node_ptr.tolist()
        for i, s in enumerate(batch_slice):
            prem = batched_targets.target_premises[i]
            if prem is not None and len(prem) > 0:
                local_prem = np.array(s.target_premises)
                expected_global = local_prem + node_ptr[i]
                np.testing.assert_array_equal(prem.numpy(), expected_global)

    def test_dataloader_iteration(self):
        ds = HypergraphDataset(self.samples)
        loader = create_hypergraph_dataloader(ds, batch_size=4, shuffle=False)

        batches = list(loader)
        self.assertEqual(len(batches), 3)

        for b_graph, b_targets in batches:
            self.assertEqual(b_graph.num_graphs, 4)
            self.assertIsInstance(b_graph.node_features, torch.Tensor)
            self.assertIsInstance(b_targets.target_actions, torch.Tensor)
            self.assertIsInstance(b_targets.target_values, torch.Tensor)

    def test_multi_worker_dataloader(self):
        dump_path = Path(self.temp_dir) / "workers_test.axmb"
        save_hypergraphs_to_binary_dump(self.samples, dump_path)

        ds = HypergraphDataset(dump_path)
        loader = create_hypergraph_dataloader(ds, batch_size=4, shuffle=True, num_workers=2)

        count = 0
        for b_graph, b_targets in loader:
            count += b_graph.num_graphs
            self.assertGreater(b_graph.total_nodes, 0)
        self.assertEqual(count, 12)

    def test_end_to_end_neural_pipeline_forward(self):
        """Verify DataLoader batches pass through Backbone and all Model Heads."""
        ds = HypergraphDataset(self.samples)
        loader = create_hypergraph_dataloader(ds, batch_size=4, shuffle=False)

        b_graph, b_targets = next(iter(loader))

        config = AxmanConfig.small(
            d_model=64,
            in_features=32,
            num_layers=2,
            num_heads=4,
            port_dim=16,
        )

        backbone = AXMANetBackbone(config)
        policy_head = MorphismPolicyHead(d_model=config.d_model)
        value_head = DistanceToClosureValueHead(d_model=config.d_model, num_heads=config.num_heads)
        diff_head = LatentInvariantDiffusionHead(
            latent_dim=config.d_model,
            context_dim=config.d_model,
        )

        # 1. Backbone forward
        bb_out = backbone(
            node_features=b_graph.node_features,
            edge_sources=b_graph.edge_sources,
            edge_ports=b_graph.edge_ports,
            edge_targets=b_graph.edge_targets,
            edge_offsets=b_graph.edge_offsets,
        )

        node_emb = bb_out.node_embeddings
        edge_emb = bb_out.edge_embeddings
        self.assertEqual(node_emb.shape, (b_graph.total_nodes, 64))

        # Unbatch node representations for each graph in the batch
        slices = b_graph.unbatch_node_features(node_emb)
        self.assertEqual(len(slices), 4)

        # 2. Policy Head forward for each sub-graph
        for i, s_nodes in enumerate(slices):
            goal_emb = s_nodes[0:1]  # first node as goal
            cand_emb = s_nodes
            policy_out = policy_head(
                goal_embeddings=goal_emb,
                candidate_embeddings=cand_emb,
            )
            self.assertEqual(policy_out.action_logits.shape, (1, 32))
            self.assertEqual(policy_out.premise_logits.shape, (1, s_nodes.shape[0]))

        # 3. Value Head forward for each sub-graph
        val_latents = []
        for i, s_nodes in enumerate(slices):
            val_out = value_head(node_embeddings=s_nodes)
            self.assertEqual(val_out.value_prob.shape, (1,))
            self.assertEqual(val_out.distance_to_qed.shape, (1,))
            val_latents.append(val_out.graph_latent)

        pooled_context = torch.stack(val_latents, dim=0)
        self.assertEqual(pooled_context.shape, (4, 64))

        # 4. Diffusion Head flow matching loss
        target_invariants = torch.randn(4, 64)
        fm_out = diff_head.compute_loss(
            target_invariants=target_invariants,
            context=pooled_context,
        )
        self.assertGreater(fm_out.loss.item(), 0.0)

    def test_collate_hypergraph_with_empty_edge_graphs(self):
        """Verify continuous batching when the first or multiple graphs have 0 edges."""
        # g0 has 0 edges
        g0 = HypergraphSample(
            node_features=np.ones((2, 16), dtype=np.float32),
            edge_sources=np.empty(0, dtype=np.int64),
            edge_ports=np.empty(0, dtype=np.int64),
            edge_targets=np.empty(0, dtype=np.int64),
            edge_offsets=np.array([0], dtype=np.int64),
        )
        # g1 has 1 edge
        g1 = HypergraphSample(
            node_features=np.ones((3, 16), dtype=np.float32),
            edge_sources=np.array([0, 1], dtype=np.int64),
            edge_ports=np.array([0, 1], dtype=np.int64),
            edge_targets=np.array([2], dtype=np.int64),
            edge_offsets=np.array([0, 2], dtype=np.int64),
        )

        bg, bt = collate_hypergraph_samples([g0, g1])
        self.assertEqual(bg.total_edges, 1)
        self.assertEqual(bg.total_sources, 2)
        self.assertEqual(bg.edge_targets.shape[0], 1)
        self.assertEqual(bg.edge_sources.shape[0], 2)
        # Verify node target index was shifted by 2 (nodes in g0)
        self.assertEqual(bg.edge_targets.tolist(), [4])  # 2 in g1 + 2 in g0 = 4
        self.assertEqual(bg.edge_sources.tolist(), [2, 3])  # 0, 1 in g1 + 2 in g0 = 2, 3

    def test_batched_targets_device_transfer_and_numpy_safety(self):
        """Verify BatchedTargets.to(...) works on both torch and numpy targets."""
        # Collate as torch
        _, bt_torch = collate_hypergraph_samples(self.samples[:2], as_torch=True)
        moved = bt_torch.to("cpu")
        self.assertIsInstance(moved.target_actions, torch.Tensor)

        # Collate as numpy
        _, bt_np = collate_hypergraph_samples(self.samples[:2], as_torch=False)
        moved_np = bt_np.to("cpu")
        self.assertIsInstance(moved_np.target_actions, np.ndarray)

    def test_dataset_context_manager_and_close(self):
        dump_path = Path(self.temp_dir) / "cm_test.axmb"
        save_hypergraphs_to_binary_dump(self.samples[:3], dump_path)
        with HypergraphDataset(dump_path) as ds:
            self.assertEqual(len(ds), 3)
            _ = ds[0]


if __name__ == "__main__":
    unittest.main()
