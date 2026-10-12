"""
Autonomous Hypergraph Dataset, Memory-Mapped I/O, and Topological Augmentation.
Part of AXMAN (AXMANet) Phase 6.1 Implementation.

This module provides:
1. HypergraphSample & BatchedTargets: Strictly typed representations of mathematical
   proof hypergraphs and multi-task learning targets.
2. TopologicalAugmentor: Isomorphic node permutations, feature jittering, and edge dropout
   preserving category-theoretic invariants and de Bruijn port semantics.
3. Memory-Mapped Datasets:
   - ArrowDataset: Ultra-high-speed memory-mapped Arrow IPC / Feather / Parquet loader.
   - BinaryDumpDataset: Native zero-dependency mmap binary file format (.axmb).
   - In-memory / list dataset wrappers.
4. PyTorch Dataset & Continuous Collation:
   - HypergraphDataset: Unified PyTorch Dataset abstraction supporting streaming,
     memory-mapping, and on-the-fly topological augmentations.
   - collate_hypergraph_samples: Fuses variable-topology samples into a single
     BatchedHypergraph via ContinuousGraphBatcher without padding (100% compute density).
   - create_hypergraph_dataloader: Multi-worker PyTorch DataLoader factory.
"""

from dataclasses import dataclass, field
import io
import json
import mmap
import os
from pathlib import Path
import struct
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    torch = None
    Dataset = object
    DataLoader = None
    HAS_TORCH = False

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pq
    HAS_ARROW = True
except ImportError:
    pa = None
    pa_ipc = None
    pq = None
    HAS_ARROW = False

from axman.core.memory.graph_pool import BatchedHypergraph, ContinuousGraphBatcher


# Magic bytes for AXMAN native binary dump format
AXMB_MAGIC = b"AXMBDUMP"
AXMB_VERSION = 1


@dataclass
class HypergraphSample:
    """
    Representation of a single mathematical proof hypergraph state with optional targets.

    Attributes:
        node_features: [N, D] float32 array or [N] int64 array of type primitive kinds.
        edge_sources: [S] int64 flat array of source premise node indices.
        edge_ports: [S] int64 array of de Bruijn positional port indices (0..k-1).
        edge_targets: [E] int64 array of target node indices.
        edge_offsets: [E + 1] int64 CSR offset boundaries into edge_sources.
        target_action: Optional int indicating ground-truth tactic constructor action.
        target_premises: Optional list/array of premise node indices chosen by the tactic.
        target_value: Optional float (in [0, 1]) probability of goal closure.
        target_distance: Optional float topological distance in steps to Q.E.D.
        target_latent: Optional [d_model] float32 vector for Flow Matching lemma abduction.
        open_goal_mask: Optional [N] bool/uint8 mask marking open sub-goals for Attention Pooling.
        metadata: Optional dictionary with proof state provenance or Lean 4 metadata.
    """
    node_features: Union[np.ndarray, "torch.Tensor"]
    edge_sources: Union[np.ndarray, "torch.Tensor"]
    edge_ports: Union[np.ndarray, "torch.Tensor"]
    edge_targets: Union[np.ndarray, "torch.Tensor"]
    edge_offsets: Union[np.ndarray, "torch.Tensor"]
    target_action: Optional[int] = None
    target_premises: Optional[Union[List[int], np.ndarray, "torch.Tensor"]] = None
    target_value: Optional[float] = None
    target_distance: Optional[float] = None
    target_latent: Optional[Union[np.ndarray, "torch.Tensor"]] = None
    open_goal_mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_targets.shape[0])

    @property
    def num_sources(self) -> int:
        return int(self.edge_sources.shape[0])

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample arrays to standard numpy-friendly dictionaries."""
        def _to_np(x):
            if x is None:
                return None
            if HAS_TORCH and isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        return {
            "node_features": _to_np(self.node_features),
            "edge_sources": _to_np(self.edge_sources),
            "edge_ports": _to_np(self.edge_ports),
            "edge_targets": _to_np(self.edge_targets),
            "edge_offsets": _to_np(self.edge_offsets),
            "target_action": self.target_action,
            "target_premises": _to_np(self.target_premises),
            "target_value": self.target_value,
            "target_distance": self.target_distance,
            "target_latent": _to_np(self.target_latent),
            "open_goal_mask": _to_np(self.open_goal_mask),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HypergraphSample":
        return cls(
            node_features=np.asarray(d["node_features"], dtype=np.float32),
            edge_sources=np.asarray(d["edge_sources"], dtype=np.int64),
            edge_ports=np.asarray(d["edge_ports"], dtype=np.int64),
            edge_targets=np.asarray(d["edge_targets"], dtype=np.int64),
            edge_offsets=np.asarray(d["edge_offsets"], dtype=np.int64),
            target_action=d.get("target_action"),
            target_premises=d.get("target_premises"),
            target_value=d.get("target_value"),
            target_distance=d.get("target_distance"),
            target_latent=np.asarray(d["target_latent"], dtype=np.float32) if d.get("target_latent") is not None else None,
            open_goal_mask=np.asarray(d["open_goal_mask"], dtype=bool) if d.get("open_goal_mask") is not None else None,
            metadata=d.get("metadata"),
        )


@dataclass
class BatchedTargets:
    """
    Collation container for multi-task targets corresponding to a BatchedHypergraph.

    Attributes:
        target_actions: [B] long tensor of tactic actions (or None).
        target_values: [B, 1] float tensor of closure probabilities (or None).
        target_distances: [B, 1] float tensor of steps to Q.E.D. (or None).
        target_latents: [B, d_model] float tensor of lemma abduction vectors (or None).
        target_premises: List of tensors, each containing remapped global node indices
                         for each graph in the batch.
        open_goal_mask: [total_nodes] bool tensor marking open subgoals across the super-graph.
    """
    target_actions: Optional[Any] = None
    target_values: Optional[Any] = None
    target_distances: Optional[Any] = None
    target_latents: Optional[Any] = None
    target_premises: Optional[List[Any]] = None
    open_goal_mask: Optional[Any] = None

    def to(self, device: Union[str, Any], non_blocking: bool = True) -> "BatchedTargets":
        """Move all target tensors to the specified device."""
        if not HAS_TORCH:
            return self

        dev = torch.device(device) if isinstance(device, str) else device
        target_actions = (
            self.target_actions.to(dev, non_blocking=non_blocking)
            if self.target_actions is not None and isinstance(self.target_actions, torch.Tensor)
            else self.target_actions
        )
        target_values = (
            self.target_values.to(dev, non_blocking=non_blocking)
            if self.target_values is not None and isinstance(self.target_values, torch.Tensor)
            else self.target_values
        )
        target_distances = (
            self.target_distances.to(dev, non_blocking=non_blocking)
            if self.target_distances is not None and isinstance(self.target_distances, torch.Tensor)
            else self.target_distances
        )
        target_latents = (
            self.target_latents.to(dev, non_blocking=non_blocking)
            if self.target_latents is not None and isinstance(self.target_latents, torch.Tensor)
            else self.target_latents
        )
        open_goal_mask = (
            self.open_goal_mask.to(dev, non_blocking=non_blocking)
            if self.open_goal_mask is not None and isinstance(self.open_goal_mask, torch.Tensor)
            else self.open_goal_mask
        )

        target_premises = None
        if self.target_premises is not None:
            target_premises = [
                p.to(dev, non_blocking=non_blocking) if isinstance(p, torch.Tensor) else p
                for p in self.target_premises
            ]

        return BatchedTargets(
            target_actions=target_actions,
            target_values=target_values,
            target_distances=target_distances,
            target_latents=target_latents,
            target_premises=target_premises,
            open_goal_mask=open_goal_mask,
        )


# ==============================================================================
# Topological Augmentation Pipeline
# ==============================================================================

class TopologicalAugmentor:
    """
    Applies structure-preserving category-theoretic and topological augmentations:
    1. Isomorphic Node Permutation: Random bijective relabeling pi: V -> V preserving
       all morphism port connections and topological connectivity.
    2. Continuous Feature Jitter: Additive Gaussian noise to node embeddings to prevent
       overfitting on static embeddings while maintaining numerical stability.
    3. Hyperedge Dropout: Randomly drops non-essential hyperedges to simulate incomplete
       proof state observation.
    """

    def __init__(
        self,
        p_permute: float = 0.5,
        p_jitter: float = 0.5,
        jitter_std: float = 0.01,
        p_edge_dropout: float = 0.0,
        edge_dropout_rate: float = 0.1,
        rng: Optional[np.random.RandomState] = None,
    ):
        self.p_permute = p_permute
        self.p_jitter = p_jitter
        self.jitter_std = jitter_std
        self.p_edge_dropout = p_edge_dropout
        self.edge_dropout_rate = edge_dropout_rate
        self.rng = rng if rng is not None else np.random.RandomState()

    def __call__(self, sample: HypergraphSample) -> HypergraphSample:
        return self.augment(sample)

    def augment(self, sample: HypergraphSample) -> HypergraphSample:
        """Apply active augmentations to a HypergraphSample."""
        def _to_numpy(x):
            if x is None:
                return None
            if HAS_TORCH and isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        is_torch = HAS_TORCH and isinstance(sample.node_features, torch.Tensor)

        node_feats = _to_numpy(sample.node_features).copy()
        edge_sources = _to_numpy(sample.edge_sources).copy()
        edge_ports = _to_numpy(sample.edge_ports).copy()
        edge_targets = _to_numpy(sample.edge_targets).copy()
        edge_offsets = _to_numpy(sample.edge_offsets).copy()
        target_premises = _to_numpy(sample.target_premises).copy() if sample.target_premises is not None else None
        open_goal_mask = _to_numpy(sample.open_goal_mask).copy() if sample.open_goal_mask is not None else None

        num_nodes = node_feats.shape[0]

        # 1. Isomorphic Node Permutation
        if self.p_permute > 0 and self.rng.rand() < self.p_permute and num_nodes > 1:
            perm = self.rng.permutation(num_nodes)
            # Inverse permutation: new_idx = perm[old_idx]
            # Permute node features
            new_node_feats = np.empty_like(node_feats)
            new_node_feats[perm] = node_feats
            node_feats = new_node_feats

            # Update edge sources & targets with perm
            if edge_sources.size > 0:
                edge_sources = perm[edge_sources]
            if edge_targets.size > 0:
                edge_targets = perm[edge_targets]

            # Update target premises if specified
            if target_premises is not None and target_premises.size > 0:
                target_premises = perm[target_premises]

            # Update open goal mask if specified
            if open_goal_mask is not None and open_goal_mask.size > 0:
                new_mask = np.empty_like(open_goal_mask)
                new_mask[perm] = open_goal_mask
                open_goal_mask = new_mask

        # 2. Continuous Feature Jitter (only applied to continuous floating-point embeddings)
        if (
            self.p_jitter > 0
            and self.rng.rand() < self.p_jitter
            and self.jitter_std > 0
            and np.issubdtype(node_feats.dtype, np.floating)
        ):
            noise = self.rng.normal(0.0, self.jitter_std, size=node_feats.shape).astype(node_feats.dtype)
            node_feats = node_feats + noise

        # 3. Hyperedge Dropout (Drop hyperedges while maintaining CSR invariants)
        num_edges = edge_targets.shape[0]
        if (
            self.p_edge_dropout > 0
            and self.rng.rand() < self.p_edge_dropout
            and num_edges > 2
            and self.edge_dropout_rate > 0
        ):
            keep_mask = self.rng.rand(num_edges) >= self.edge_dropout_rate
            # Ensure at least 1 edge is retained
            if not np.any(keep_mask):
                keep_mask[0] = True

            new_edge_targets = edge_targets[keep_mask]
            new_sources_list = []
            new_ports_list = []
            new_offsets = [0]

            for e_idx in range(num_edges):
                start = edge_offsets[e_idx]
                end = edge_offsets[e_idx + 1]
                if keep_mask[e_idx]:
                    new_sources_list.append(edge_sources[start:end])
                    new_ports_list.append(edge_ports[start:end])
                    new_offsets.append(new_offsets[-1] + (end - start))

            edge_targets = new_edge_targets
            edge_sources = np.concatenate(new_sources_list) if new_sources_list else np.empty(0, dtype=np.int64)
            edge_ports = np.concatenate(new_ports_list) if new_ports_list else np.empty(0, dtype=np.int64)
            edge_offsets = np.array(new_offsets, dtype=np.int64)

        if is_torch:
            return HypergraphSample(
                node_features=torch.from_numpy(node_feats),
                edge_sources=torch.from_numpy(edge_sources),
                edge_ports=torch.from_numpy(edge_ports),
                edge_targets=torch.from_numpy(edge_targets),
                edge_offsets=torch.from_numpy(edge_offsets),
                target_action=sample.target_action,
                target_premises=torch.from_numpy(target_premises) if target_premises is not None else None,
                target_value=sample.target_value,
                target_distance=sample.target_distance,
                target_latent=sample.target_latent if isinstance(sample.target_latent, torch.Tensor) or sample.target_latent is None else torch.from_numpy(sample.target_latent),
                open_goal_mask=torch.from_numpy(open_goal_mask) if open_goal_mask is not None else None,
                metadata=sample.metadata,
            )

        return HypergraphSample(
            node_features=node_feats,
            edge_sources=edge_sources,
            edge_ports=edge_ports,
            edge_targets=edge_targets,
            edge_offsets=edge_offsets,
            target_action=sample.target_action,
            target_premises=target_premises,
            target_value=sample.target_value,
            target_distance=sample.target_distance,
            target_latent=sample.target_latent,
            open_goal_mask=open_goal_mask,
            metadata=sample.metadata,
        )


# ==============================================================================
# Native Binary Dump Storage (.axmb) with Memory Mapping
# ==============================================================================

def save_hypergraphs_to_binary_dump(
    samples: Sequence[HypergraphSample],
    filepath: Union[str, Path],
) -> int:
    """
    Serializes a sequence of HypergraphSample objects into an optimized .axmb binary dump.

    Format:
    [8B Magic: AXMBDUMP][4B Version][4B NumSamples][8B IndexTableOffset]
    ... [4B PayloadLength][Serialized Sample Payload] ...
    [Index Table: NumSamples * 8B Offsets]
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    num_samples = len(samples)
    offsets: List[int] = []

    with open(filepath, "wb") as f:
        # Reserve header space: 8 + 4 + 4 + 8 = 24 bytes
        header_placeholder = struct.pack("<8sIIQ", AXMB_MAGIC, AXMB_VERSION, num_samples, 0)
        f.write(header_placeholder)

        for sample in samples:
            offset = f.tell()
            offsets.append(offset)

            # Serialize sample data
            s_dict = sample.to_dict()
            nf = s_dict["node_features"].astype(np.float32)
            src = s_dict["edge_sources"].astype(np.int64)
            prt = s_dict["edge_ports"].astype(np.int64)
            tgt = s_dict["edge_targets"].astype(np.int64)
            off = s_dict["edge_offsets"].astype(np.int64)

            action = -1 if s_dict["target_action"] is None else int(s_dict["target_action"])
            val = float(s_dict["target_value"]) if s_dict["target_value"] is not None else float("nan")
            dist = float(s_dict["target_distance"]) if s_dict["target_distance"] is not None else float("nan")

            premises = s_dict["target_premises"]
            prem_arr = premises.astype(np.int64) if premises is not None else np.empty(0, dtype=np.int64)

            latent = s_dict["target_latent"]
            latent_arr = latent.astype(np.float32) if latent is not None else np.empty(0, dtype=np.float32)

            goal_mask = s_dict["open_goal_mask"]
            goal_arr = goal_mask.astype(np.uint8) if goal_mask is not None else np.empty(0, dtype=np.uint8)

            meta_json = json.dumps(s_dict["metadata"]).encode("utf-8") if s_dict.get("metadata") is not None else b""

            # Pack payload
            bio = io.BytesIO()
            # Header per sample:
            # num_nodes(I), feat_dim(I), num_edges(I), num_sources(I), num_prem(I), latent_dim(I), goal_len(I), meta_len(I), action(i), val(f), dist(f)
            bio.write(struct.pack(
                "<IIIIIIIIiff",
                nf.shape[0], nf.shape[1] if nf.ndim > 1 else 1,
                tgt.shape[0], src.shape[0], prem_arr.shape[0], latent_arr.shape[0],
                goal_arr.shape[0], len(meta_json), action, val, dist
            ))
            bio.write(nf.tobytes())
            bio.write(src.tobytes())
            bio.write(prt.tobytes())
            bio.write(tgt.tobytes())
            bio.write(off.tobytes())
            bio.write(prem_arr.tobytes())
            bio.write(latent_arr.tobytes())
            bio.write(goal_arr.tobytes())
            bio.write(meta_json)

            payload = bio.getvalue()
            # Write 4B length + payload
            f.write(struct.pack("<I", len(payload)))
            f.write(payload)

        # Write index table
        index_table_offset = f.tell()
        for off_pos in offsets:
            f.write(struct.pack("<Q", off_pos))

        # Backpatch header
        f.seek(0)
        f.write(struct.pack("<8sIIQ", AXMB_MAGIC, AXMB_VERSION, num_samples, index_table_offset))

    return num_samples


class BinaryDumpReader:
    """
    Zero-copy memory-mapped random-access reader for .axmb files.
    Supports multi-process PyTorch DataLoader workers with lazy reopening.
    """

    def __init__(self, filepath: Union[str, Path]):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Binary dump file not found: {self.filepath}")

        self._file: Optional[Any] = None
        self._mmap: Optional[mmap.mmap] = None
        self._offsets: Optional[List[int]] = None
        self.num_samples: int = 0
        self.index_offset: int = 0

        self._open_and_read_index()

    def _open_and_read_index(self):
        self._file = open(self.filepath, "rb")
        self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Parse header
        magic, version, num_samples, index_offset = struct.unpack_from("<8sIIQ", self._mmap, 0)
        if magic != AXMB_MAGIC:
            raise ValueError(f"Invalid magic bytes in {self.filepath}: {magic}")
        if version != AXMB_VERSION:
            raise ValueError(f"Unsupported binary dump version {version} in {self.filepath}")

        self.num_samples = num_samples
        self.index_offset = index_offset

        # Read offsets from index table
        self._offsets = [
            struct.unpack_from("<Q", self._mmap, self.index_offset + i * 8)[0]
            for i in range(self.num_samples)
        ]

    def _ensure_open(self):
        if self._mmap is None or self._file is None:
            self._file = open(self.filepath, "rb")
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        state["_mmap"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._file = None
        self._mmap = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> HypergraphSample:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        self._ensure_open()

        rec_offset = self._offsets[idx]
        payload_len = struct.unpack_from("<I", self._mmap, rec_offset)[0]
        pos = rec_offset + 4

        # Read sample header
        (
            n_nodes, feat_dim, n_edges, n_src, n_prem,
            latent_dim, goal_len, meta_len, action, val, dist
        ) = struct.unpack_from("<IIIIIIIIiff", self._mmap, pos)
        pos += struct.calcsize("<IIIIIIIIiff")

        # Read arrays and copy to release buffer locks immediately
        nf_bytes = n_nodes * feat_dim * 4
        node_features = np.frombuffer(self._mmap, dtype=np.float32, count=n_nodes * feat_dim, offset=pos).reshape(n_nodes, feat_dim).copy()
        pos += nf_bytes

        src_bytes = n_src * 8
        edge_sources = np.frombuffer(self._mmap, dtype=np.int64, count=n_src, offset=pos).copy()
        pos += src_bytes

        prt_bytes = n_src * 8
        edge_ports = np.frombuffer(self._mmap, dtype=np.int64, count=n_src, offset=pos).copy()
        pos += prt_bytes

        tgt_bytes = n_edges * 8
        edge_targets = np.frombuffer(self._mmap, dtype=np.int64, count=n_edges, offset=pos).copy()
        pos += tgt_bytes

        off_bytes = (n_edges + 1) * 8
        edge_offsets = np.frombuffer(self._mmap, dtype=np.int64, count=n_edges + 1, offset=pos).copy()
        pos += off_bytes

        target_premises = None
        if n_prem > 0:
            target_premises = np.frombuffer(self._mmap, dtype=np.int64, count=n_prem, offset=pos).copy()
            pos += n_prem * 8

        target_latent = None
        if latent_dim > 0:
            target_latent = np.frombuffer(self._mmap, dtype=np.float32, count=latent_dim, offset=pos).copy()
            pos += latent_dim * 4

        open_goal_mask = None
        if goal_len > 0:
            goal_raw = np.frombuffer(self._mmap, dtype=np.uint8, count=goal_len, offset=pos).copy()
            pos += goal_len
            open_goal_mask = goal_raw.astype(bool)

        metadata = None
        if meta_len > 0:
            meta_raw = bytes(self._mmap[pos:pos + meta_len])
            metadata = json.loads(meta_raw.decode("utf-8"))

        return HypergraphSample(
            node_features=node_features,
            edge_sources=edge_sources,
            edge_ports=edge_ports,
            edge_targets=edge_targets,
            edge_offsets=edge_offsets,
            target_action=None if action == -1 else action,
            target_premises=target_premises,
            target_value=None if np.isnan(val) else val,
            target_distance=None if np.isnan(dist) else dist,
            target_latent=target_latent,
            open_goal_mask=open_goal_mask,
            metadata=metadata,
        )

    def close(self):
        if hasattr(self, "_mmap") and self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:
                pass
            self._mmap = None
        if hasattr(self, "_file") and self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self.close()


# ==============================================================================
# Apache Arrow IPC / Parquet Dataset Storage with Memory Mapping
# ==============================================================================

def save_hypergraphs_to_arrow(
    samples: Sequence[HypergraphSample],
    filepath: Union[str, Path],
) -> int:
    """
    Serializes hypergraph samples to an Apache Arrow IPC RecordBatch / Feather file.
    """
    if not HAS_ARROW:
        raise ImportError("pyarrow is required to save hypergraphs to Arrow format.")

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    rows: Dict[str, List[Any]] = {
        "node_features": [],
        "node_features_shape": [],
        "edge_sources": [],
        "edge_ports": [],
        "edge_targets": [],
        "edge_offsets": [],
        "target_action": [],
        "target_premises": [],
        "target_value": [],
        "target_distance": [],
        "target_latent": [],
        "open_goal_mask": [],
        "metadata": [],
    }

    for s in samples:
        d = s.to_dict()
        nf = d["node_features"].astype(np.float32)
        rows["node_features"].append(nf.tobytes())
        rows["node_features_shape"].append(list(nf.shape))
        rows["edge_sources"].append(list(d["edge_sources"]))
        rows["edge_ports"].append(list(d["edge_ports"]))
        rows["edge_targets"].append(list(d["edge_targets"]))
        rows["edge_offsets"].append(list(d["edge_offsets"]))
        rows["target_action"].append(d["target_action"] if d["target_action"] is not None else -1)
        premises = d["target_premises"]
        rows["target_premises"].append(list(premises) if premises is not None else None)
        rows["target_value"].append(float(d["target_value"]) if d["target_value"] is not None else float("nan"))
        rows["target_distance"].append(float(d["target_distance"]) if d["target_distance"] is not None else float("nan"))
        latent = d["target_latent"]
        rows["target_latent"].append(latent.tobytes() if latent is not None else None)
        goal_mask = d["open_goal_mask"]
        rows["open_goal_mask"].append(list(goal_mask) if goal_mask is not None else None)
        rows["metadata"].append(json.dumps(d["metadata"]) if d.get("metadata") is not None else None)

    table = pa.Table.from_arrays(
        [
            pa.array(rows["node_features"], type=pa.binary()),
            pa.array(rows["node_features_shape"], type=pa.list_(pa.int64())),
            pa.array(rows["edge_sources"], type=pa.list_(pa.int64())),
            pa.array(rows["edge_ports"], type=pa.list_(pa.int64())),
            pa.array(rows["edge_targets"], type=pa.list_(pa.int64())),
            pa.array(rows["edge_offsets"], type=pa.list_(pa.int64())),
            pa.array(rows["target_action"], type=pa.int64()),
            pa.array(rows["target_premises"], type=pa.list_(pa.int64())),
            pa.array(rows["target_value"], type=pa.float32()),
            pa.array(rows["target_distance"], type=pa.float32()),
            pa.array(rows["target_latent"], type=pa.binary()),
            pa.array(rows["open_goal_mask"], type=pa.list_(pa.bool_())),
            pa.array(rows["metadata"], type=pa.string()),
        ],
        names=[
            "node_features", "node_features_shape",
            "edge_sources", "edge_ports", "edge_targets", "edge_offsets",
            "target_action", "target_premises", "target_value", "target_distance",
            "target_latent", "open_goal_mask", "metadata"
        ]
    )

    with pa.OSFile(str(filepath), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)

    return len(samples)


class ArrowDatasetReader:
    """
    Memory-mapped reader for Apache Arrow IPC / Feather files.
    Supports multi-process PyTorch DataLoader workers with lazy reopening.
    """

    def __init__(self, filepath: Union[str, Path]):
        if not HAS_ARROW:
            raise ImportError("pyarrow is required to read Arrow files.")

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Arrow file not found: {self.filepath}")

        self._mmap_source: Optional[Any] = None
        self._reader: Optional[Any] = None
        self._table: Optional[Any] = None
        self.num_samples: int = 0

        self._open_table()

    def _open_table(self):
        self._mmap_source = pa.memory_map(str(self.filepath))
        self._reader = pa.ipc.open_file(self._mmap_source)
        self._table = self._reader.read_all()
        self.num_samples = self._table.num_rows

    def _ensure_open(self):
        if self._table is None or self._mmap_source is None:
            self._open_table()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mmap_source"] = None
        state["_reader"] = None
        state["_table"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._mmap_source = None
        self._reader = None
        self._table = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> HypergraphSample:
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.num_samples})")

        self._ensure_open()

        row = self._table.slice(idx, 1)

        nf_bytes = row.column("node_features")[0].as_py()
        nf_shape = row.column("node_features_shape")[0].as_py()
        node_features = np.frombuffer(nf_bytes, dtype=np.float32).reshape(nf_shape).copy()

        edge_sources = np.array(row.column("edge_sources")[0].as_py(), dtype=np.int64)
        edge_ports = np.array(row.column("edge_ports")[0].as_py(), dtype=np.int64)
        edge_targets = np.array(row.column("edge_targets")[0].as_py(), dtype=np.int64)
        edge_offsets = np.array(row.column("edge_offsets")[0].as_py(), dtype=np.int64)

        action = row.column("target_action")[0].as_py()
        premises = row.column("target_premises")[0].as_py()
        val = row.column("target_value")[0].as_py()
        dist = row.column("target_distance")[0].as_py()

        latent_bytes = row.column("target_latent")[0].as_py()
        latent = np.frombuffer(latent_bytes, dtype=np.float32).copy() if latent_bytes is not None else None

        goals = row.column("open_goal_mask")[0].as_py()
        goal_mask = np.array(goals, dtype=bool) if goals is not None else None

        meta_str = row.column("metadata")[0].as_py()
        meta = json.loads(meta_str) if meta_str is not None else None

        return HypergraphSample(
            node_features=node_features,
            edge_sources=edge_sources,
            edge_ports=edge_ports,
            edge_targets=edge_targets,
            edge_offsets=edge_offsets,
            target_action=None if action == -1 else int(action),
            target_premises=np.array(premises, dtype=np.int64) if premises is not None else None,
            target_value=None if np.isnan(val) else float(val),
            target_distance=None if np.isnan(dist) else float(dist),
            target_latent=latent,
            open_goal_mask=goal_mask,
            metadata=meta,
        )

    def close(self):
        if hasattr(self, "_mmap_source") and self._mmap_source is not None:
            try:
                self._mmap_source.close()
            except Exception:
                pass
            self._mmap_source = None
        self._reader = None
        self._table = None

    def __del__(self):
        self.close()


# ==============================================================================
# Unified PyTorch Dataset Abstraction & Continuous Collation
# ==============================================================================

class HypergraphDataset(Dataset):
    """
    Unified PyTorch Dataset supporting:
    - In-memory sequences of HypergraphSample
    - Native zero-copy memory-mapped binary dumps (.axmb)
    - Apache Arrow IPC (.arrow / .feather) memory-mapped files
    - Dynamic on-the-fly topological augmentations (Isomorphic permutations, jitter, dropout)
    """

    def __init__(
        self,
        source: Union[Sequence[HypergraphSample], str, Path],
        transform: Optional[Callable[[HypergraphSample], HypergraphSample]] = None,
    ):
        self.transform = transform
        self._reader: Optional[Any] = None
        self._in_memory_samples: Optional[Sequence[HypergraphSample]] = None

        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.suffix.lower() == ".axmb" or (path.exists() and not path.suffix):
                # Binary dump
                self._reader = BinaryDumpReader(path)
            elif path.suffix.lower() in (".arrow", ".feather", ".ipc"):
                self._reader = ArrowDatasetReader(path)
            else:
                # Default to binary dump reader if file matches magic
                try:
                    self._reader = BinaryDumpReader(path)
                except Exception:
                    self._reader = ArrowDatasetReader(path)
        else:
            self._in_memory_samples = source

    def __len__(self) -> int:
        if self._reader is not None:
            return len(self._reader)
        return len(self._in_memory_samples or [])

    def __getitem__(self, idx: int) -> HypergraphSample:
        if self._reader is not None:
            sample = self._reader[idx]
        else:
            sample = self._in_memory_samples[idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def close(self):
        """Close underlying memory mapped reader descriptors."""
        if self._reader is not None and hasattr(self._reader, "close"):
            self._reader.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()


def collate_hypergraph_samples(
    samples: List[HypergraphSample],
    as_torch: bool = True,
) -> Tuple[BatchedHypergraph, BatchedTargets]:
    """
    Continuous batching collator that packs variable-topology HypergraphSample objects
    into a single contiguous BatchedHypergraph (via ContinuousGraphBatcher) and collates
    multi-task targets with remapped premise indices.

    Args:
        samples: List of HypergraphSample objects.
        as_torch: Return PyTorch tensors (default True).

    Returns:
        Tuple of (BatchedHypergraph, BatchedTargets).
    """
    if not samples:
        raise ValueError("Cannot collate empty samples list.")

    # 1. Batch hypergraphs using ContinuousGraphBatcher
    batched_graph = ContinuousGraphBatcher.batch(samples, as_torch=as_torch)

    # 2. Extract and remap targets
    actions: List[int] = []
    has_actions = True

    values: List[float] = []
    has_values = True

    distances: List[float] = []
    has_distances = True

    latents: List[Any] = []
    has_latents = True

    remapped_premises: List[Any] = []
    open_goal_masks: List[Any] = []

    # Node pointers from batched graph
    if HAS_TORCH and isinstance(batched_graph.graph_node_ptr, torch.Tensor):
        node_ptr = batched_graph.graph_node_ptr.cpu().tolist()
    else:
        node_ptr = list(batched_graph.graph_node_ptr)

    for i, s in enumerate(samples):
        node_offset = node_ptr[i]

        # Action
        if s.target_action is not None:
            actions.append(s.target_action)
        else:
            has_actions = False

        # Value
        if s.target_value is not None:
            values.append(s.target_value)
        else:
            has_values = False

        # Distance
        if s.target_distance is not None:
            distances.append(s.target_distance)
        else:
            has_distances = False

        # Latent
        if s.target_latent is not None:
            lat = s.target_latent
            if HAS_TORCH and isinstance(lat, torch.Tensor):
                latents.append(lat.detach().cpu().numpy())
            else:
                latents.append(np.asarray(lat))
        else:
            has_latents = False

        # Remap premise indices by node_offset
        if s.target_premises is not None:
            prems = np.asarray(s.target_premises, dtype=np.int64)
            if prems.size > 0:
                remapped = prems + node_offset
                remapped_premises.append(
                    torch.tensor(remapped, dtype=torch.long) if as_torch and HAS_TORCH else remapped
                )
            else:
                remapped_premises.append(
                    torch.empty(0, dtype=torch.long) if as_torch and HAS_TORCH else np.empty(0, dtype=np.int64)
                )
        else:
            remapped_premises.append(None)

        # Open goal mask
        num_n = s.num_nodes
        if s.open_goal_mask is not None:
            gm = np.asarray(s.open_goal_mask, dtype=bool)
        else:
            gm = np.zeros(num_n, dtype=bool)
        open_goal_masks.append(gm)

    # 3. Assemble target tensors
    target_actions = None
    if has_actions and actions:
        target_actions = torch.tensor(actions, dtype=torch.long) if as_torch and HAS_TORCH else np.array(actions, dtype=np.int64)

    target_values = None
    if has_values and values:
        target_values = (
            torch.tensor(values, dtype=torch.float32).unsqueeze(-1)
            if as_torch and HAS_TORCH
            else np.array(values, dtype=np.float32)[:, None]
        )

    target_distances = None
    if has_distances and distances:
        target_distances = (
            torch.tensor(distances, dtype=torch.float32).unsqueeze(-1)
            if as_torch and HAS_TORCH
            else np.array(distances, dtype=np.float32)[:, None]
        )

    target_latents = None
    if has_latents and latents:
        stacked_lat = np.stack(latents, axis=0).astype(np.float32)
        target_latents = torch.tensor(stacked_lat, dtype=torch.float32) if as_torch and HAS_TORCH else stacked_lat

    cat_goal_mask = np.concatenate(open_goal_masks, axis=0)
    target_goal_mask = torch.tensor(cat_goal_mask, dtype=torch.bool) if as_torch and HAS_TORCH else cat_goal_mask

    batched_targets = BatchedTargets(
        target_actions=target_actions,
        target_values=target_values,
        target_distances=target_distances,
        target_latents=target_latents,
        target_premises=remapped_premises,
        open_goal_mask=target_goal_mask,
    )

    return batched_graph, batched_targets


def create_hypergraph_dataloader(
    dataset: HypergraphDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
) -> Any:
    """
    Creates an optimized PyTorch DataLoader for hypergraphs with continuous batching.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to create a Hypergraph DataLoader.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_hypergraph_samples,
    )
