import json
import struct
from pathlib import Path
import numpy as np

def generate_golden_dataset():
    data_dir = Path(__file__).resolve().parent
    
    # 1. Config
    config = {
        "d_model": 64,
        "num_layers": 2,
        "num_heads": 4,
        "num_actions": 16,
        "in_features": 32,
    }
    config_path = data_dir / "golden_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    # 2. Graph topology
    # 4 nodes with 32 features each
    num_nodes = 4
    feat_dim = 32
    
    # Deterministic features: f_i,j = sin(i * 10 + j)
    feats = np.zeros((num_nodes, feat_dim), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(feat_dim):
            feats[i, j] = np.sin(float(i * 10 + j) * 0.1)

    # Hyperedges:
    # Edge 0: (0) -> 1, port 0
    # Edge 1: (0, 1) -> 2, ports (0, 1)
    # Edge 2: (1, 2) -> 3, ports (0, 2)
    num_edges = 3
    edge_sources = np.array([0, 0, 1, 1, 2], dtype=np.int64)
    edge_ports = np.array([0, 0, 1, 0, 2], dtype=np.int32)
    edge_targets = np.array([1, 2, 3], dtype=np.int64)
    edge_offsets = np.array([0, 1, 3, 5], dtype=np.uint64)

    graph_dict = {
        "num_nodes": num_nodes,
        "node_feat_dim": feat_dim,
        "num_hyperedges": num_edges,
        "node_features": feats.tolist(),
        "edge_sources": edge_sources.tolist(),
        "edge_ports": edge_ports.tolist(),
        "edge_targets": edge_targets.tolist(),
        "edge_offsets": edge_offsets.tolist(),
    }
    
    graph_path = data_dir / "golden_graph.json"
    with open(graph_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2)

    # Also binary representation for high speed:
    # magic 'AXMG' (4 bytes), num_nodes (u64), feat_dim (u64), num_edges (u64), num_sources (u64)
    # then raw buffers
    bin_path = data_dir / "golden_sample.bin"
    with open(bin_path, "wb") as f:
        f.write(b"AXMG")
        f.write(struct.pack("<QQQQ", num_nodes, feat_dim, num_edges, len(edge_sources)))
        f.write(feats.tobytes())
        f.write(edge_sources.tobytes())
        f.write(edge_ports.tobytes())
        f.write(edge_targets.tobytes())
        f.write(edge_offsets.tobytes())

    print(f"Generated golden dataset in {data_dir}")

if __name__ == "__main__":
    generate_golden_dataset()
