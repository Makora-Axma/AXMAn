"""
High-level zero-copy Python wrapper for AXMANet.

Supports:
- Direct inference from PyTorch tensors (`torch.Tensor`) or NumPy arrays (`np.ndarray`)
- Zero-copy view conversion to and from C-ABI structures
- GIL release during forward pass execution for true multi-threaded scaling
- Idempotent RAII memory management via ctypes and object lifecycle
"""

import ctypes
import json
from dataclasses import dataclass
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from . import _c_api as ffi


class AxmanError(Exception):
    """Exception raised on errors from AXMANet native core."""
    def __init__(self, code: int, message: str):
        self.code = code
        self.message = message
        super().__init__(f"AXMAN Error [{code}]: {message}")


def _check_status(status: int):
    if status != ffi.AXMAN_SUCCESS:
        raw_msg = ffi._lib.axman_get_last_error_message()
        if raw_msg:
            msg = raw_msg.decode("utf-8")
        else:
            fallback = ffi._lib.axman_status_to_string(status)
            msg = fallback.decode("utf-8") if fallback else f"Error code {status}"
        raise AxmanError(status, msg)


def version() -> str:
    """Return the library version string."""
    v = ffi._lib.axman_get_version()
    return v.decode("utf-8") if v else "1.0.0"


@dataclass
class Hypergraph:
    """
    In-memory representation of a Port-Hypergraph conforming to AXMAN C-ABI.
    Can be created from PyTorch tensors or NumPy arrays.
    """
    node_features: Union[np.ndarray, "torch.Tensor"]
    edge_sources: Union[np.ndarray, "torch.Tensor"]
    edge_ports: Union[np.ndarray, "torch.Tensor"]
    edge_targets: Union[np.ndarray, "torch.Tensor"]
    edge_offsets: Union[np.ndarray, "torch.Tensor"]

    def __post_init__(self):
        # Convert torch tensors to contiguous numpy arrays if needed
        self._feat_np = self._ensure_numpy(self.node_features, np.float32)
        self._sources_np = self._ensure_numpy(self.edge_sources, np.int64)
        self._ports_np = self._ensure_numpy(self.edge_ports, np.int32)
        self._targets_np = self._ensure_numpy(self.edge_targets, np.int64)
        self._offsets_np = self._ensure_numpy(self.edge_offsets, np.uint64)

        if self._feat_np.ndim != 2:
            raise ValueError(f"node_features must be 2D, got shape {self._feat_np.shape}")

        self.num_nodes = self._feat_np.shape[0]
        self.node_feat_dim = self._feat_np.shape[1]
        self.num_hyperedges = len(self._targets_np)

        if len(self._offsets_np) != self.num_hyperedges + 1:
            raise ValueError(
                f"edge_offsets length ({len(self._offsets_np)}) must be num_hyperedges + 1 ({self.num_hyperedges + 1})"
            )

        if len(self._sources_np) != len(self._ports_np):
            raise ValueError(
                f"edge_sources length ({len(self._sources_np)}) must equal edge_ports length ({len(self._ports_np)})"
            )

        if self.num_hyperedges > 0:
            if self._offsets_np[0] != 0:
                raise ValueError(f"edge_offsets[0] must be 0, got {self._offsets_np[0]}")
            if self._offsets_np[self.num_hyperedges] != len(self._sources_np):
                raise ValueError(
                    f"edge_offsets[{self.num_hyperedges}] ({self._offsets_np[self.num_hyperedges]}) must match len(edge_sources) ({len(self._sources_np)})"
                )

    @staticmethod
    def _ensure_numpy(arr: Any, dtype: Any) -> np.ndarray:
        if _HAS_TORCH and isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()
        arr = np.ascontiguousarray(arr, dtype=dtype)
        return arr

    def validate(self) -> None:
        """Validate topological and arity invariants of the hypergraph."""
        raw_graph = self._to_c_struct()
        status = ffi._lib.axman_hypergraph_validate(ctypes.byref(raw_graph))
        _check_status(status)

    def structural_hash(self) -> int:
        """Compute the deterministic 64-bit de Bruijn structural hash of the hypergraph."""
        raw_graph = self._to_c_struct()
        out_hash = ctypes.c_uint64(0)
        status = ffi._lib.axman_hypergraph_hash(ctypes.byref(raw_graph), ctypes.byref(out_hash))
        _check_status(status)
        return out_hash.value

    def _to_c_struct(self) -> ffi.AxmanHypergraph:
        if self.num_nodes > 0 and self._feat_np.size > 0:
            shape_arr = (ctypes.c_int64 * 2)(self.num_nodes, self.node_feat_dim)
            feat_tensor = ffi.AxmanTensor(
                data=ctypes.c_void_p(self._feat_np.ctypes.data),
                shape=ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64)),
                ndim=2,
                dtype=ffi.AXMAN_DTYPE_FLOAT32,
                device=ffi.AXMAN_DEVICE_CPU,
            )
        else:
            shape_arr = None
            feat_tensor = ffi.AxmanTensor(
                data=None,
                shape=None,
                ndim=0,
                dtype=ffi.AXMAN_DTYPE_FLOAT32,
                device=ffi.AXMAN_DEVICE_CPU,
            )

        sources_ptr = (
            ctypes.cast(self._sources_np.ctypes.data, ctypes.POINTER(ctypes.c_int64))
            if len(self._sources_np) > 0
            else None
        )
        ports_ptr = (
            ctypes.cast(self._ports_np.ctypes.data, ctypes.POINTER(ctypes.c_int32))
            if len(self._ports_np) > 0
            else None
        )
        targets_ptr = (
            ctypes.cast(self._targets_np.ctypes.data, ctypes.POINTER(ctypes.c_int64))
            if len(self._targets_np) > 0
            else None
        )
        offsets_ptr = (
            ctypes.cast(self._offsets_np.ctypes.data, ctypes.POINTER(ctypes.c_size_t))
            if len(self._offsets_np) > 0
            else None
        )

        # Keep reference to prevent GC of shape_arr
        raw = ffi.AxmanHypergraph(
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            node_features=feat_tensor,
            edge_sources=sources_ptr,
            edge_ports=ports_ptr,
            edge_targets=targets_ptr,
            edge_offsets=offsets_ptr,
        )
        raw._shape_keepalive = shape_arr
        return raw


@dataclass
class ModelConfig:
    """Strongly typed model configuration matching AXMANet architectural parameters."""
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 16
    port_dim: int = 128
    num_kinds: int = 16
    in_features: int = 64
    num_actions: int = 256
    max_ports: int = 16

    @classmethod
    def small(cls) -> "ModelConfig":
        return cls(
            d_model=256,
            num_layers=6,
            num_heads=8,
            port_dim=64,
            num_kinds=16,
            in_features=64,
            num_actions=128,
            max_ports=16,
        )

    @classmethod
    def base(cls) -> "ModelConfig":
        return cls(
            d_model=512,
            num_layers=12,
            num_heads=16,
            port_dim=128,
            num_kinds=16,
            in_features=64,
            num_actions=256,
            max_ports=16,
        )

    @classmethod
    def large(cls) -> "ModelConfig":
        return cls(
            d_model=1024,
            num_layers=24,
            num_heads=16,
            port_dim=256,
            num_kinds=16,
            in_features=128,
            num_actions=512,
            max_ports=16,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "port_dim": self.port_dim,
            "num_kinds": self.num_kinds,
            "in_features": self.in_features,
            "num_actions": self.num_actions,
            "max_ports": self.max_ports,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class HypergraphBuilder:
    """
    Fluent builder for constructing Port-Hypergraphs with validation.
    """
    def __init__(self, node_feat_dim: int):
        self.node_feat_dim = node_feat_dim
        self._nodes = []
        self._sources = []
        self._ports = []
        self._targets = []
        self._offsets = [0]

    def add_node(self, features: Union[np.ndarray, "torch.Tensor", list]) -> int:
        if _HAS_TORCH and isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        arr = np.asarray(features, dtype=np.float32).flatten()
        if len(arr) != self.node_feat_dim:
            raise ValueError(f"features length ({len(arr)}) != node_feat_dim ({self.node_feat_dim})")
        idx = len(self._nodes)
        self._nodes.append(arr)
        return idx

    def add_node_fill(self, value: float = 0.0) -> int:
        arr = np.full((self.node_feat_dim,), value, dtype=np.float32)
        return self.add_node(arr)

    def add_hyperedge(
        self,
        sources: Union[list, np.ndarray, "torch.Tensor"],
        ports: Union[list, np.ndarray, "torch.Tensor"],
        target: int,
    ) -> "HypergraphBuilder":
        src_list = list(sources)
        port_list = list(ports)
        if len(src_list) != len(port_list):
            raise ValueError("sources and ports must have the same length")
        if len(src_list) > ffi.AXMAN_MAX_HYPEREDGE_ARITY:
            raise ValueError(f"Hyperedge arity {len(src_list)} exceeds k_max = {ffi.AXMAN_MAX_HYPEREDGE_ARITY}")

        self._sources.extend(src_list)
        self._ports.extend(port_list)
        self._targets.append(int(target))
        self._offsets.append(len(self._sources))
        return self

    def add_edge(self, source: int, target: int, port: int = 0) -> "HypergraphBuilder":
        return self.add_hyperedge([source], [port], target)

    def build(self) -> Hypergraph:
        if self._nodes:
            feat_mat = np.stack(self._nodes, axis=0)
        else:
            feat_mat = np.empty((0, self.node_feat_dim), dtype=np.float32)
        return Hypergraph(
            node_features=feat_mat,
            edge_sources=np.array(self._sources, dtype=np.int64),
            edge_ports=np.array(self._ports, dtype=np.int32),
            edge_targets=np.array(self._targets, dtype=np.int64),
            edge_offsets=np.array(self._offsets, dtype=np.uint64),
        )


class InferenceOutput:
    """
    RAII container for model outputs. Converts C-ABI tensors to NumPy / PyTorch views
    and ensures `axman_output_free` is called on garbage collection.
    """
    def __init__(self, raw_output: ffi.AxmanOutput, return_torch: bool = True):
        self._raw = raw_output
        self._return_torch = return_torch and _HAS_TORCH
        self._closed = False

        # Extract values
        self.value_estimate: float = float(self._raw.value_estimate)
        self.policy_logits = self._tensor_to_view(self._raw.policy_logits)
        self.latent_embedding = self._tensor_to_view(self._raw.latent_embedding)

    def _tensor_to_view(self, t: ffi.AxmanTensor) -> Union[np.ndarray, "torch.Tensor"]:
        if not t.data or t.ndim == 0:
            empty = np.empty((0,), dtype=np.float32)
            return torch.from_numpy(empty) if self._return_torch else empty

        shape = [t.shape[i] for i in range(t.ndim)]
        total_elements = 1
        for s in shape:
            if s <= 0:
                total_elements = 0
                break
            total_elements *= s

        if total_elements == 0:
            arr = np.empty(tuple(shape), dtype=np.float32)
            return torch.from_numpy(arr) if self._return_torch else arr

        # Zero-copy buffer pointer
        buffer_from_memory = ctypes.pythonapi.PyMemoryView_FromMemory
        buffer_from_memory.restype = ctypes.py_object
        buffer_from_memory.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t, ctypes.c_int]

        # Read-only memoryview (flags = 0x100)
        view = buffer_from_memory(t.data, total_elements * 4, 0x100)
        # Copy into independent numpy array so memory remains valid after output_free
        arr = np.frombuffer(view, dtype=np.float32).reshape(shape).copy()

        if self._return_torch:
            return torch.from_numpy(arr)
        return arr

    def close(self):
        if not self._closed:
            ffi._lib.axman_output_free(ctypes.byref(self._raw))
            self._closed = True

    def __del__(self):
        self.close()


class AxmanNet:
    """
    High-level Python model class for AXMANet inference.
    """
    def __init__(self, handle: ctypes.c_void_p):
        self._handle = handle
        self._closed = False

    @classmethod
    def load(
        cls,
        model_path: Optional[str] = None,
        config: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> "AxmanNet":
        """
        Load an AXMANet model instance.
        
        Args:
            model_path: Path to .safetensors weights file (optional).
            config: Model configuration dictionary or JSON string (optional).
        """
        path_bytes = model_path.encode("utf-8") if model_path else None
        
        if config is None:
            json_bytes = None
        elif isinstance(config, ModelConfig):
            json_bytes = config.to_json().encode("utf-8")
        elif isinstance(config, dict):
            json_bytes = json.dumps(config).encode("utf-8")
        elif isinstance(config, str):
            json_bytes = config.encode("utf-8")
        else:
            raise TypeError("config must be a dict, json str, or ModelConfig")

        handle = ctypes.c_void_p()
        status = ffi._lib.axman_model_create(path_bytes, json_bytes, ctypes.byref(handle))
        _check_status(status)

        return cls(handle)

    @classmethod
    def from_file(
        cls,
        model_path: str,
        config: Optional[Union[str, Dict[str, Any], ModelConfig]] = None,
    ) -> "AxmanNet":
        """Load model directly from weights file."""
        return cls.load(model_path=model_path, config=config)

    @classmethod
    def from_config(
        cls,
        config: Union[str, Dict[str, Any], ModelConfig],
    ) -> "AxmanNet":
        """Load model from configuration with initialized weights."""
        return cls.load(model_path=None, config=config)

    def forward(
        self,
        graph: Hypergraph,
        return_torch: bool = True,
    ) -> InferenceOutput:
        """
        Run forward inference on the provided hypergraph.
        """
        if self._closed or not self._handle:
            raise RuntimeError("Model is closed or not initialized")

        raw_graph = graph._to_c_struct()
        raw_out = ffi.AxmanOutput()

        status = ffi._lib.axman_model_forward(
            self._handle,
            ctypes.byref(raw_graph),
            ctypes.byref(raw_out),
        )
        _check_status(status)

        return InferenceOutput(raw_out, return_torch=return_torch)

    def __call__(self, graph: Hypergraph, return_torch: bool = True) -> InferenceOutput:
        return self.forward(graph, return_torch=return_torch)

    def close(self):
        if not self._closed and self._handle:
            ffi._lib.axman_model_destroy(self._handle)
            self._handle = None
            self._closed = True

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
