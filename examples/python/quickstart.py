#!/usr/bin/env python3
"""
AXMANet Python Quickstart Example.
Demonstrates constructing a Port-Hypergraph and executing zero-copy forward inference.
"""

import numpy as np
import torch
from pathlib import Path
import sys

# Ensure axman packages are resolvable
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root / "axman" / "bindings" / "python"))
sys.path.insert(0, str(repo_root / "bindings" / "python"))
sys.path.insert(0, str(repo_root))

try:
    from axman.bindings.python.axman import Hypergraph, AxmanNet, HypergraphBuilder
except (ImportError, AttributeError):
    try:
        from bindings.python.axman import Hypergraph, AxmanNet, HypergraphBuilder
    except (ImportError, AttributeError):
        from axman import Hypergraph, AxmanNet, HypergraphBuilder


def main():
    print("=" * 60)
    print("       AXMANet Python Quickstart: Formal Reasoning Inference      ")
    print("=" * 60)

    # 1. Construct a Port-Hypergraph using the fluent builder
    feat_dim = 256
    builder = HypergraphBuilder(feat_dim)

    # Add hypothesis and premise nodes
    h_idx = builder.add_node(np.random.randn(feat_dim).astype(np.float32))  # Node 0 (Hypothesis)
    p1_idx = builder.add_node(np.random.randn(feat_dim).astype(np.float32)) # Node 1 (Premise 1)
    p2_idx = builder.add_node(np.random.randn(feat_dim).astype(np.float32)) # Node 2 (Premise 2)
    goal_idx = builder.add_node(np.random.randn(feat_dim).astype(np.float32)) # Node 3 (Goal)

    # Add a binary hyperedge (premise 1 + premise 2 -> goal) with de Bruijn port ordering
    builder.add_hyperedge(sources=[p1_idx, p2_idx], ports=[0, 1], target=goal_idx)
    # Add a unary edge (hypothesis -> premise 1)
    builder.add_edge(source=h_idx, target=p1_idx, port=0)

    graph = builder.build()
    print(f"[1] Hypergraph constructed: {graph.num_nodes} nodes, {graph.num_hyperedges} hyperedges.")

    # 2. Validate topology and compute deterministic 64-bit structural hash
    graph.validate()
    struct_hash = graph.structural_hash()
    print(f"[2] Topological validation passed. Structural Hash: 0x{struct_hash:016x}")

    # 3. Model Configuration (AXMAN-Small default)
    config = {
        "d_model": feat_dim,
        "num_layers": 4,
        "num_heads": 8,
        "port_dim": 64,
        "num_actions": 17,
        "device": "cpu"
    }

    # 4. Perform Inference via native runtime
    print("[3] Running zero-copy forward inference...")
    with AxmanNet.load(config=config) as model:
        output = model.forward(graph, return_torch=True)

        print(f"  -> Value Estimate (Distance to Closure): {output.value_estimate:.4f}")
        print(f"  -> Policy Logits shape: {output.policy_logits.shape}")
        print(f"  -> Latent Invariant Embedding shape: {output.latent_embedding.shape}")

    print("\n[SUCCESS] AXMANet Python Quickstart completed successfully!")


if __name__ == "__main__":
    main()
