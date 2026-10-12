"""
Pinned Graph Memory Pools and Continuous Batching for AXMANet.
Part of AXMAN (AXMANet) Phase 5.2 Implementation.

This module provides:
1. PinnedHostMemoryPool: Pre-allocated, page-locked (pinned) host memory buffers
   allowing fast, zero-copy, asynchronous DMA transfers to GPU via CUDA Streams without jitter.
2. DeviceRingBuffer: Cyclic ring buffers on GPU to eliminate runtime cudaMalloc/cudaFree overhead.
3. BatchedHypergraph: Continuous batching aggregator that fuses multiple variable-topology
   hypergraphs into a single contiguous super-graph with re-indexed nodes, offsets, and sub-graph masks.
4. GraphMemoryPool: Complete memory management coordinator for training and high-throughput inference.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


@dataclass
class BatchedHypergraph:
    """
    Representation of a fused super-graph produced by continuous batching
    of multiple independent variable-topology hypergraphs.

    Attributes:
        node_features: [total_nodes, feat_dim] continuous tensor.
        edge_sources: [total_sources] flat indices of sources remapped into global node IDs.
        edge_ports: [total_sources] de Bruijn positional port indices.
        edge_targets: [total_edges] target node indices remapped into global node IDs.
        edge_offsets: [total_edges + 1] CSR offsets into edge_sources.
        graph_node_ptr: [num_graphs + 1] boundary indices of nodes for each sub-graph.
        graph_edge_ptr: [num_graphs + 1] boundary indices of edges for each sub-graph.
        graph_source_ptr: [num_graphs + 1] boundary indices of sources for each sub-graph.
        num_graphs: Number of graphs batched in this super-graph.
        total_nodes: Total count of nodes across all graphs.
        total_edges: Total count of hyperedges across all graphs.
        total_sources: Total count of source premises across all graphs.
    """
    node_features: Any
    edge_sources: Any
    edge_ports: Any
    edge_targets: Any
    edge_offsets: Any
    graph_node_ptr: Any
    graph_edge_ptr: Any
    graph_source_ptr: Any
    num_graphs: int
    total_nodes: int
    total_edges: int
    total_sources: int

    def to(self, device: Union[str, Any], non_blocking: bool = True) -> "BatchedHypergraph":
        """Move all batched tensors to the specified device."""
        if not HAS_TORCH:
            return self

        dev = torch.device(device) if isinstance(device, str) else device
        return BatchedHypergraph(
            node_features=self.node_features.to(dev, non_blocking=non_blocking),
            edge_sources=self.edge_sources.to(dev, non_blocking=non_blocking),
            edge_ports=self.edge_ports.to(dev, non_blocking=non_blocking),
            edge_targets=self.edge_targets.to(dev, non_blocking=non_blocking),
            edge_offsets=self.edge_offsets.to(dev, non_blocking=non_blocking),
            graph_node_ptr=self.graph_node_ptr.to(dev, non_blocking=non_blocking),
            graph_edge_ptr=self.graph_edge_ptr.to(dev, non_blocking=non_blocking),
            graph_source_ptr=self.graph_source_ptr.to(dev, non_blocking=non_blocking),
            num_graphs=self.num_graphs,
            total_nodes=self.total_nodes,
            total_edges=self.total_edges,
            total_sources=self.total_sources,
        )

    def unbatch_node_features(self, out_node_embeddings: Any) -> List[Any]:
        """
        Split a batched node embeddings tensor back into a list of individual
        node embeddings per original graph.
        """
        slices = []
        node_ptr = self.graph_node_ptr
        if HAS_TORCH and isinstance(node_ptr, torch.Tensor):
            ptr = node_ptr.cpu().tolist()
        else:
            ptr = list(node_ptr)

        for i in range(self.num_graphs):
            start, end = ptr[i], ptr[i + 1]
            slices.append(out_node_embeddings[start:end])
        return slices

    def unbatch_edge_embeddings(self, out_edge_embeddings: Any) -> List[Any]:
        """
        Split a batched edge embeddings tensor back into a list of individual
        edge embeddings per original graph.
        """
        slices = []
        edge_ptr = self.graph_edge_ptr
        if HAS_TORCH and isinstance(edge_ptr, torch.Tensor):
            ptr = edge_ptr.cpu().tolist()
        else:
            ptr = list(edge_ptr)

        for i in range(self.num_graphs):
            start, end = ptr[i], ptr[i + 1]
            slices.append(out_edge_embeddings[start:end])
        return slices


class ContinuousGraphBatcher:
    """
    Assembles multiple independent variable-topology hypergraphs into a single
    disjoint super-graph.

    Maintains CSR invariants and offsets:
      - Shifts node indices in `edge_sources` and `edge_targets` by the cumulative node count.
      - Shifts CSR `edge_offsets` by cumulative source premise count.
      - Eliminates padding, achieving 100% compute density.
    """

    @staticmethod
    def batch(graphs: List[Any], as_torch: bool = True) -> BatchedHypergraph:
        """
        Batch a list of hypergraphs into a unified BatchedHypergraph.

        Args:
            graphs: List of hypergraph objects or dictionaries with keys:
                    ['node_features', 'edge_sources', 'edge_ports', 'edge_targets', 'edge_offsets']
            as_torch: Whether to return torch tensors or numpy arrays.
        """
        if not graphs:
            raise ValueError("Cannot batch an empty list of hypergraphs.")

        num_graphs = len(graphs)
        node_feats_list = []
        edge_sources_list = []
        edge_ports_list = []
        edge_targets_list = []
        edge_offsets_list = []

        graph_node_ptr = [0]
        graph_edge_ptr = [0]
        graph_source_ptr = [0]

        curr_node_offset = 0
        curr_source_offset = 0

        for g in graphs:
            # Extract attributes from dict or object
            if isinstance(g, dict):
                nf = g["node_features"]
                src = g["edge_sources"]
                prt = g["edge_ports"]
                tgt = g["edge_targets"]
                off = g["edge_offsets"]
            else:
                nf = g.node_features
                src = g.edge_sources
                prt = g.edge_ports
                tgt = g.edge_targets
                off = g.edge_offsets

            # Convert to numpy for fast unified slicing & offset calculation
            if HAS_TORCH and isinstance(nf, torch.Tensor):
                nf = nf.detach().cpu().numpy()
            if HAS_TORCH and isinstance(src, torch.Tensor):
                src = src.detach().cpu().numpy()
            if HAS_TORCH and isinstance(prt, torch.Tensor):
                prt = prt.detach().cpu().numpy()
            if HAS_TORCH and isinstance(tgt, torch.Tensor):
                tgt = tgt.detach().cpu().numpy()
            if HAS_TORCH and isinstance(off, torch.Tensor):
                off = off.detach().cpu().numpy()

            num_nodes = nf.shape[0]
            num_edges = len(tgt)
            num_sources = len(src)

            # Remap sources and targets by node offset
            shifted_sources = src.astype(np.int64) + curr_node_offset
            shifted_targets = tgt.astype(np.int64) + curr_node_offset

            # Remap edge_offsets (excluding the leading offset when concatenating)
            shifted_offsets = off.astype(np.int64) + curr_source_offset

            node_feats_list.append(nf)
            edge_sources_list.append(shifted_sources)
            edge_ports_list.append(prt.astype(np.int32))
            edge_targets_list.append(shifted_targets)

            # For CSR offsets: if first graph, keep [0..len], else skip leading element
            if not edge_offsets_list:
                edge_offsets_list.append(shifted_offsets)
            else:
                edge_offsets_list.append(shifted_offsets[1:])

            curr_node_offset += num_nodes
            curr_source_offset += num_sources

            graph_node_ptr.append(curr_node_offset)
            graph_edge_ptr.append(graph_edge_ptr[-1] + num_edges)
            graph_source_ptr.append(curr_source_offset)

        # Concatenate arrays
        batch_nf = np.concatenate(node_feats_list, axis=0)
        batch_src = np.concatenate(edge_sources_list, axis=0) if curr_source_offset > 0 else np.empty(0, dtype=np.int64)
        batch_prt = np.concatenate(edge_ports_list, axis=0) if curr_source_offset > 0 else np.empty(0, dtype=np.int32)
        batch_tgt = np.concatenate(edge_targets_list, axis=0) if graph_edge_ptr[-1] > 0 else np.empty(0, dtype=np.int64)
        batch_off = np.concatenate(edge_offsets_list, axis=0) if edge_offsets_list else np.array([0], dtype=np.int64)

        node_ptr_arr = np.array(graph_node_ptr, dtype=np.int64)
        edge_ptr_arr = np.array(graph_edge_ptr, dtype=np.int64)
        src_ptr_arr = np.array(graph_source_ptr, dtype=np.int64)

        if as_torch and HAS_TORCH:
            return BatchedHypergraph(
                node_features=torch.from_numpy(batch_nf).float(),
                edge_sources=torch.from_numpy(batch_src).long(),
                edge_ports=torch.from_numpy(batch_prt).int(),
                edge_targets=torch.from_numpy(batch_tgt).long(),
                edge_offsets=torch.from_numpy(batch_off).long(),
                graph_node_ptr=torch.from_numpy(node_ptr_arr).long(),
                graph_edge_ptr=torch.from_numpy(edge_ptr_arr).long(),
                graph_source_ptr=torch.from_numpy(src_ptr_arr).long(),
                num_graphs=num_graphs,
                total_nodes=curr_node_offset,
                total_edges=graph_edge_ptr[-1],
                total_sources=curr_source_offset,
            )

        return BatchedHypergraph(
            node_features=batch_nf,
            edge_sources=batch_src,
            edge_ports=batch_prt,
            edge_targets=batch_tgt,
            edge_offsets=batch_off,
            graph_node_ptr=node_ptr_arr,
            graph_edge_ptr=edge_ptr_arr,
            graph_source_ptr=src_ptr_arr,
            num_graphs=num_graphs,
            total_nodes=curr_node_offset,
            total_edges=graph_edge_ptr[-1],
            total_sources=curr_source_offset,
        )


class PinnedHostMemoryPool:
    """
    Fixed-capacity page-locked (pinned) host memory pool.
    Allocates contiguous host tensors marked as pinned_memory() for fast DMA transfers
    to GPU without operating system page-faults or page migration overhead.
    """

    def __init__(
        self,
        max_nodes: int = 65536,
        max_edges: int = 65536,
        max_sources: int = 131072,
        feat_dim: int = 256,
        device: str = "cpu",
    ):
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_sources = max_sources
        self.feat_dim = feat_dim

        self.has_pinned = HAS_TORCH and torch.cuda.is_available()

        if HAS_TORCH:
            # Pre-allocate pinned host tensors
            self.node_features = torch.empty((max_nodes, feat_dim), dtype=torch.float32, pin_memory=self.has_pinned)
            self.edge_sources = torch.empty((max_sources,), dtype=torch.int64, pin_memory=self.has_pinned)
            self.edge_ports = torch.empty((max_sources,), dtype=torch.int32, pin_memory=self.has_pinned)
            self.edge_targets = torch.empty((max_edges,), dtype=torch.int64, pin_memory=self.has_pinned)
            self.edge_offsets = torch.empty((max_edges + 1,), dtype=torch.int64, pin_memory=self.has_pinned)
        else:
            self.node_features = np.empty((max_nodes, feat_dim), dtype=np.float32)
            self.edge_sources = np.empty((max_sources,), dtype=np.int64)
            self.edge_ports = np.empty((max_sources,), dtype=np.int32)
            self.edge_targets = np.empty((max_edges,), dtype=np.int64)
            self.edge_offsets = np.empty((max_edges + 1,), dtype=np.int64)

    def copy_batched_graph(self, batched: BatchedHypergraph) -> BatchedHypergraph:
        """
        Copies batched hypergraph arrays into pre-allocated pinned memory slots
        and returns non-allocating sub-views.
        """
        if batched.total_nodes > self.max_nodes:
            raise ValueError(f"Total nodes ({batched.total_nodes}) exceeds pool capacity ({self.max_nodes}).")
        if batched.total_edges > self.max_edges:
            raise ValueError(f"Total edges ({batched.total_edges}) exceeds pool capacity ({self.max_edges}).")
        if batched.total_sources > self.max_sources:
            raise ValueError(f"Total sources ({batched.total_sources}) exceeds pool capacity ({self.max_sources}).")

        if HAS_TORCH and isinstance(batched.node_features, torch.Tensor):
            self.node_features[:batched.total_nodes].copy_(batched.node_features)
            self.edge_sources[:batched.total_sources].copy_(batched.edge_sources)
            self.edge_ports[:batched.total_sources].copy_(batched.edge_ports)
            self.edge_targets[:batched.total_edges].copy_(batched.edge_targets)
            self.edge_offsets[:batched.total_edges + 1].copy_(batched.edge_offsets)

            return BatchedHypergraph(
                node_features=self.node_features[:batched.total_nodes],
                edge_sources=self.edge_sources[:batched.total_sources],
                edge_ports=self.edge_ports[:batched.total_sources],
                edge_targets=self.edge_targets[:batched.total_edges],
                edge_offsets=self.edge_offsets[:batched.total_edges + 1],
                graph_node_ptr=batched.graph_node_ptr,
                graph_edge_ptr=batched.graph_edge_ptr,
                graph_source_ptr=batched.graph_source_ptr,
                num_graphs=batched.num_graphs,
                total_nodes=batched.total_nodes,
                total_edges=batched.total_edges,
                total_sources=batched.total_sources,
            )
        else:
            self.node_features[:batched.total_nodes] = batched.node_features
            self.edge_sources[:batched.total_sources] = batched.edge_sources
            self.edge_ports[:batched.total_sources] = batched.edge_ports
            self.edge_targets[:batched.total_edges] = batched.edge_targets
            self.edge_offsets[:batched.total_edges + 1] = batched.edge_offsets

            return BatchedHypergraph(
                node_features=self.node_features[:batched.total_nodes],
                edge_sources=self.edge_sources[:batched.total_sources],
                edge_ports=self.edge_ports[:batched.total_sources],
                edge_targets=self.edge_targets[:batched.total_edges],
                edge_offsets=self.edge_offsets[:batched.total_edges + 1],
                graph_node_ptr=batched.graph_node_ptr,
                graph_edge_ptr=batched.graph_edge_ptr,
                graph_source_ptr=batched.graph_source_ptr,
                num_graphs=batched.num_graphs,
                total_nodes=batched.total_nodes,
                total_edges=batched.total_edges,
                total_sources=batched.total_sources,
            )


class DeviceRingBuffer:
    """
    Cyclic ring buffer on device (GPU/CPU) eliminating per-forward allocation jitter.
    Maintains multiple fixed-capacity buffer slots rotated across inference iterations.
    """

    def __init__(
        self,
        num_slots: int = 3,
        max_nodes: int = 65536,
        max_edges: int = 65536,
        max_sources: int = 131072,
        feat_dim: int = 256,
        device: Union[str, Any] = "cpu",
    ):
        self.num_slots = num_slots
        self.current_slot = 0
        self.device = torch.device(device) if (HAS_TORCH and isinstance(device, str)) else device

        self.slots = []
        if HAS_TORCH:
            for _ in range(num_slots):
                slot = {
                    "node_features": torch.empty((max_nodes, feat_dim), dtype=torch.float32, device=self.device),
                    "edge_sources": torch.empty((max_sources,), dtype=torch.int64, device=self.device),
                    "edge_ports": torch.empty((max_sources,), dtype=torch.int32, device=self.device),
                    "edge_targets": torch.empty((max_edges,), dtype=torch.int64, device=self.device),
                    "edge_offsets": torch.empty((max_edges + 1,), dtype=torch.int64, device=self.device),
                }
                self.slots.append(slot)

    def acquire_slot(
        self,
        batched: BatchedHypergraph,
        non_blocking: bool = True,
        stream: Optional[Any] = None,
    ) -> BatchedHypergraph:
        """
        Acquires next cyclic buffer slot on device and transfers the batched graph
        via non-blocking DMA.
        """
        if not HAS_TORCH or not self.slots:
            return batched.to(self.device) if hasattr(batched, "to") else batched

        slot = self.slots[self.current_slot]
        self.current_slot = (self.current_slot + 1) % self.num_slots

        # Contextual stream if provided
        ctx = torch.cuda.stream(stream) if (stream is not None and torch.cuda.is_available()) else nullcontext()

        with ctx:
            slot["node_features"][:batched.total_nodes].copy_(batched.node_features, non_blocking=non_blocking)
            slot["edge_sources"][:batched.total_sources].copy_(batched.edge_sources, non_blocking=non_blocking)
            slot["edge_ports"][:batched.total_sources].copy_(batched.edge_ports, non_blocking=non_blocking)
            slot["edge_targets"][:batched.total_edges].copy_(batched.edge_targets, non_blocking=non_blocking)
            slot["edge_offsets"][:batched.total_edges + 1].copy_(batched.edge_offsets, non_blocking=non_blocking)

            return BatchedHypergraph(
                node_features=slot["node_features"][:batched.total_nodes],
                edge_sources=slot["edge_sources"][:batched.total_sources],
                edge_ports=slot["edge_ports"][:batched.total_sources],
                edge_targets=slot["edge_targets"][:batched.total_edges],
                edge_offsets=slot["edge_offsets"][:batched.total_edges + 1],
                graph_node_ptr=batched.graph_node_ptr.to(self.device, non_blocking=non_blocking),
                graph_edge_ptr=batched.graph_edge_ptr.to(self.device, non_blocking=non_blocking),
                graph_source_ptr=batched.graph_source_ptr.to(self.device, non_blocking=non_blocking),
                num_graphs=batched.num_graphs,
                total_nodes=batched.total_nodes,
                total_edges=batched.total_edges,
                total_sources=batched.total_sources,
            )


class GraphMemoryPool:
    """
    Unified Graph Memory Pool coordinating Pinned Host Memory, Continuous Batching,
    and Device Cyclic Ring Buffers for zero-jitter inference and training pipelines.
    """

    def __init__(
        self,
        max_nodes: int = 65536,
        max_edges: int = 65536,
        max_sources: int = 131072,
        feat_dim: int = 256,
        device: Union[str, Any] = "cpu",
        num_device_slots: int = 2,
    ):
        self.device = torch.device(device) if (HAS_TORCH and isinstance(device, str)) else device
        self.batcher = ContinuousGraphBatcher()
        self.pinned_pool = PinnedHostMemoryPool(
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_sources=max_sources,
            feat_dim=feat_dim,
        )
        self.device_ring = DeviceRingBuffer(
            num_slots=num_device_slots,
            max_nodes=max_nodes,
            max_edges=max_edges,
            max_sources=max_sources,
            feat_dim=feat_dim,
            device=self.device,
        )

    def prepare_batch(
        self,
        graphs: List[Any],
        non_blocking: bool = True,
        stream: Optional[Any] = None,
    ) -> BatchedHypergraph:
        """
        Full pipeline:
        1. Fuses multiple variable-topology graphs into a continuous super-graph.
        2. Ingests into pinned host memory buffer.
        3. Asynchronously transfers into device cyclic ring buffer slot.
        """
        batched = self.batcher.batch(graphs, as_torch=True)
        pinned_batched = self.pinned_pool.copy_batched_graph(batched)
        device_batched = self.device_ring.acquire_slot(pinned_batched, non_blocking=non_blocking, stream=stream)
        return device_batched


# Lightweight nullcontext helper when not using torch cuda streams
class nullcontext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
