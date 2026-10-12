"""
Unit tests for Python bindings (axman-py).

Tests:
1. Version check
2. Hypergraph creation from NumPy and PyTorch tensors
3. Validation and structural hash calculation
4. Model lifecycle (create, forward, destroy, context manager)
5. Zero-copy views with PyTorch and NumPy
6. Multithreaded execution across Python threads
7. Error propagation from C-ABI
"""

import threading
import pytest
import numpy as np
import torch
import sys
import os
import tempfile
from pathlib import Path

# Ensure axman and bindings/python are in sys.path
_axman_root = Path(__file__).resolve().parent.parent.parent.parent
_bindings_py = Path(__file__).resolve().parent.parent
for p in [str(_axman_root), str(_bindings_py)]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from bindings.python import axman
    from bindings.python.axman import _c_api as ffi
except ModuleNotFoundError:
    import axman
    from axman import _c_api as ffi


def test_version():
    ver = axman.version()
    assert ver == "1.0.0"
    assert axman.__version__ == "1.0.0"


def test_hypergraph_numpy_and_hash():
    num_nodes = 3
    feat_dim = 32
    node_features = np.full((num_nodes, feat_dim), 0.5, dtype=np.float32)
    edge_sources = np.array([0, 0, 1], dtype=np.int64)
    edge_ports = np.array([0, 0, 1], dtype=np.int32)
    edge_targets = np.array([1, 2], dtype=np.int64)
    edge_offsets = np.array([0, 1, 3], dtype=np.uint64)

    graph = axman.Hypergraph(
        node_features=node_features,
        edge_sources=edge_sources,
        edge_ports=edge_ports,
        edge_targets=edge_targets,
        edge_offsets=edge_offsets,
    )

    graph.validate()
    h = graph.structural_hash()
    assert h != 0


def test_hypergraph_torch_integration():
    num_nodes = 2
    feat_dim = 16
    node_features = torch.randn(num_nodes, feat_dim, dtype=torch.float32)
    edge_sources = torch.tensor([0], dtype=torch.int64)
    edge_ports = torch.tensor([0], dtype=torch.int32)
    edge_targets = torch.tensor([1], dtype=torch.int64)
    edge_offsets = torch.tensor([0, 1], dtype=torch.int64)

    graph = axman.Hypergraph(
        node_features=node_features,
        edge_sources=edge_sources,
        edge_ports=edge_ports,
        edge_targets=edge_targets,
        edge_offsets=edge_offsets,
    )
    graph.validate()
    h = graph.structural_hash()
    assert h != 0


def test_model_forward_numpy_and_torch():
    config = {
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_actions": 16,
        "in_features": 32,
    }

    with axman.AxmanNet.load(config=config) as model:
        # Build graph
        feats = np.full((3, 32), 0.25, dtype=np.float32)
        sources = np.array([0, 0, 1], dtype=np.int64)
        ports = np.array([0, 0, 1], dtype=np.int32)
        targets = np.array([1, 2], dtype=np.int64)
        offsets = np.array([0, 1, 3], dtype=np.uint64)

        graph = axman.Hypergraph(
            node_features=feats,
            edge_sources=sources,
            edge_ports=ports,
            edge_targets=targets,
            edge_offsets=offsets,
        )

        # 1. Forward returning PyTorch tensors
        out_torch = model(graph, return_torch=True)
        assert isinstance(out_torch.policy_logits, torch.Tensor)
        assert out_torch.policy_logits.shape == (3, 16)
        assert isinstance(out_torch.latent_embedding, torch.Tensor)
        assert out_torch.latent_embedding.shape == (1, 64)
        assert 0.0 <= out_torch.value_estimate <= 1.0

        # 2. Forward returning NumPy arrays
        out_numpy = model(graph, return_torch=False)
        assert isinstance(out_numpy.policy_logits, np.ndarray)
        assert out_numpy.policy_logits.shape == (3, 16)
        assert isinstance(out_numpy.latent_embedding, np.ndarray)
        assert out_numpy.latent_embedding.shape == (1, 64)
        assert 0.0 <= out_numpy.value_estimate <= 1.0

        # Verify numerical match between torch and numpy
        np.testing.assert_allclose(
            out_torch.policy_logits.numpy(),
            out_numpy.policy_logits,
            rtol=1e-5,
            atol=1e-5,
        )


def test_multithreaded_inference():
    config = {
        "d_model": 64,
        "num_layers": 1,
        "num_heads": 2,
        "num_actions": 8,
        "in_features": 16,
    }

    model = axman.AxmanNet.load(config=config)
    feats = np.full((2, 16), 0.1, dtype=np.float32)
    sources = np.array([0], dtype=np.int64)
    ports = np.array([0], dtype=np.int32)
    targets = np.array([1], dtype=np.int64)
    offsets = np.array([0, 1], dtype=np.uint64)

    graph = axman.Hypergraph(feats, sources, ports, targets, offsets)

    results = [None] * 20

    def worker(idx):
        out = model(graph)
        results[idx] = out.value_estimate

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    model.close()

    for val in results:
        assert val is not None
        assert 0.0 <= val <= 1.0


def test_invalid_topology_error_handling():
    feats = np.zeros((2, 16), dtype=np.float32)
    sources = np.array([0], dtype=np.int64)
    ports = np.array([0], dtype=np.int32)
    targets = np.array([999], dtype=np.int64) # Invalid!
    offsets = np.array([0, 1], dtype=np.uint64)

    graph = axman.Hypergraph(feats, sources, ports, targets, offsets)

    with pytest.raises(axman.AxmanError) as exc_info:
        graph.validate()
    assert exc_info.value.code == ffi.AXMAN_ERR_INDEX_OUT_OF_BOUNDS

    with axman.AxmanNet.load() as model:
        with pytest.raises(axman.AxmanError) as exc_info:
            model.forward(graph)
        assert exc_info.value.code == ffi.AXMAN_ERR_INDEX_OUT_OF_BOUNDS


def test_hypergraph_builder():
    builder = axman.HypergraphBuilder(node_feat_dim=32)
    n0 = builder.add_node_fill(0.1)
    n1 = builder.add_node(np.full((32,), 0.2, dtype=np.float32))
    n2 = builder.add_node(torch.full((32,), 0.3, dtype=torch.float32))

    assert n0 == 0
    assert n1 == 1
    assert n2 == 2

    builder.add_edge(n0, n1)
    builder.add_hyperedge([n0, n1], [0, 1], n2)

    graph = builder.build()
    assert graph.num_nodes == 3
    assert graph.num_hyperedges == 2
    graph.validate()
    h = graph.structural_hash()
    assert h != 0


def test_model_config_presets():
    small = axman.ModelConfig.small()
    assert small.d_model == 256
    assert small.num_layers == 6

    base = axman.ModelConfig.base()
    assert base.d_model == 512

    large = axman.ModelConfig.large()
    assert large.d_model == 1024

    with axman.AxmanNet.from_config(small) as model:
        assert model._handle is not None


def test_safetensors_file_load():
    import struct

    temp_path = os.path.join(tempfile.gettempdir(), "axman_python_test.safetensors")

    w_data = [1.2, -1.8, 2.5, 0.5]
    b_data = [0.05, -0.1]

    json_header = (
        '{"__metadata__":{"d_model":"2","num_layers":"1","num_heads":"1","in_features":"2"},'
        '"node_encoder.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},'
        '"node_encoder.bias":{"dtype":"F32","shape":[2],"data_offsets":[16,24]}}'
    )
    header_bytes = json_header.encode("utf-8")
    header_len = len(header_bytes)

    with open(temp_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_bytes)
        for w in w_data:
            f.write(struct.pack("<f", w))
        for b in b_data:
            f.write(struct.pack("<f", b))

    try:
        with axman.AxmanNet.from_file(temp_path) as model:
            builder = axman.HypergraphBuilder(node_feat_dim=2)
            n0 = builder.add_node_fill(1.0)
            n1 = builder.add_node_fill(0.5)
            builder.add_edge(n0, n1)
            graph = builder.build()

            out = model.forward(graph)
            assert 0.0 <= out.value_estimate <= 1.0
            assert out.policy_logits.shape[0] == 2
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
