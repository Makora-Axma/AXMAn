import sys
import json
import struct
from pathlib import Path
import numpy as np

# Add bindings/python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bindings" / "python"))
import axman

def run_python_benchmark(base_dir_str="."):
    base_dir = Path(base_dir_str)
    config_path = base_dir / "golden_config.json"
    bin_path = base_dir / "golden_sample.bin"
    out_path = base_dir / "golden_output_python.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(bin_path, "rb") as f:
        magic = f.read(4)
        assert magic == b"AXMG"
        num_nodes, feat_dim, num_edges, num_sources = struct.unpack("<QQQQ", f.read(32))
        features = np.frombuffer(f.read(num_nodes * feat_dim * 4), dtype=np.float32).reshape((num_nodes, feat_dim))
        sources = np.frombuffer(f.read(num_sources * 8), dtype=np.int64)
        ports = np.frombuffer(f.read(num_sources * 4), dtype=np.int32)
        targets = np.frombuffer(f.read(num_edges * 8), dtype=np.int64)
        offsets = np.frombuffer(f.read((num_edges + 1) * 8), dtype=np.uint64)

    graph = axman.Hypergraph(
        node_features=features,
        edge_sources=sources,
        edge_ports=ports,
        edge_targets=targets,
        edge_offsets=offsets,
    )
    graph.validate()
    structural_hash = graph.structural_hash()

    with axman.AxmanNet.load(config=config) as model:
        output = model.forward(graph, return_torch=False)
        val = float(output.value_estimate)
        policy = output.policy_logits.flatten().tolist()
        latent = output.latent_embedding.flatten().tolist()

    out_data = {
        "language": "python",
        "structural_hash": structural_hash,
        "value_estimate": val,
        "policy_logits": policy,
        "latent_embedding": latent,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    print(f"[Python] Executed successfully. Structural hash: {structural_hash}, value: {val}")

if __name__ == "__main__":
    bdir = sys.argv[1] if len(sys.argv) > 1 else "."
    run_python_benchmark(bdir)
