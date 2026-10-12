"""
C-ABI FFI ctypes bindings for libaxman.

Defines exact structure mappings and function prototypes matching `include/axman_c_api.h`.
"""

import ctypes
import os
import sys
from pathlib import Path
from typing import Optional

# Search for libaxman.dll / libaxman.so
def _find_libaxman() -> str:
    possible_paths = [
        Path(__file__).resolve().parent.parent.parent.parent / "libaxman.dll",
        Path(__file__).resolve().parent.parent.parent.parent / "build" / "libaxman.dll",
        Path(__file__).resolve().parent.parent.parent.parent / "libaxman.so",
        Path(__file__).resolve().parent.parent.parent.parent / "build" / "libaxman.so",
    ]
    for p in possible_paths:
        if p.exists():
            return str(p)
    # Fallback to system load
    if sys.platform == "win32":
        return "libaxman.dll"
    return "libaxman.so"

_LIB_PATH = _find_libaxman()
try:
    _lib = ctypes.CDLL(_LIB_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load AXMAN native library from '{_LIB_PATH}': {e}") from e

# Enums
AXMAN_SUCCESS = 0
AXMAN_ERR_INVALID_ARGUMENT = 1
AXMAN_ERR_OUT_OF_MEMORY = 2
AXMAN_ERR_DEVICE_UNAVAILABLE = 3
AXMAN_ERR_INFERENCE_FAILURE = 4
AXMAN_ERR_NULL_POINTER = 5
AXMAN_ERR_ARITY_OVERFLOW = 6
AXMAN_ERR_INDEX_OUT_OF_BOUNDS = 7
AXMAN_ERR_INVALID_TOPOLOGY = 8
AXMAN_ERR_INTERNAL = 99

AXMAN_DTYPE_FLOAT32 = 0
AXMAN_DTYPE_BFLOAT16 = 1
AXMAN_DTYPE_INT64 = 2
AXMAN_DTYPE_INT32 = 3
AXMAN_DTYPE_FLOAT16 = 4
AXMAN_DTYPE_INT8 = 5
AXMAN_DTYPE_FP8_E4M3 = 6

AXMAN_PRECISION_FP32 = 0
AXMAN_PRECISION_BF16 = 1
AXMAN_PRECISION_FP8_E4M3 = 2
AXMAN_PRECISION_INT8 = 3

AXMAN_DEVICE_CPU = 0
AXMAN_DEVICE_CUDA = 1

AXMAN_MAX_HYPEREDGE_ARITY = 16

class AxmanTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("ndim", ctypes.c_size_t),
        ("dtype", ctypes.c_int32),
        ("device", ctypes.c_int32),
    ]

class AxmanHypergraph(ctypes.Structure):
    _fields_ = [
        ("num_nodes", ctypes.c_size_t),
        ("num_hyperedges", ctypes.c_size_t),
        ("node_features", AxmanTensor),
        ("edge_sources", ctypes.POINTER(ctypes.c_int64)),
        ("edge_ports", ctypes.POINTER(ctypes.c_int32)),
        ("edge_targets", ctypes.POINTER(ctypes.c_int64)),
        ("edge_offsets", ctypes.POINTER(ctypes.c_size_t)),
    ]

class AxmanOutput(ctypes.Structure):
    _fields_ = [
        ("policy_logits", AxmanTensor),
        ("value_estimate", ctypes.c_float),
        ("latent_embedding", AxmanTensor),
    ]

# Function signatures
_lib.axman_model_create.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.axman_model_create.restype = ctypes.c_int

_lib.axman_model_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(AxmanHypergraph), ctypes.POINTER(AxmanOutput)]
_lib.axman_model_forward.restype = ctypes.c_int

_lib.axman_model_destroy.argtypes = [ctypes.c_void_p]
_lib.axman_model_destroy.restype = ctypes.c_int

_lib.axman_output_free.argtypes = [ctypes.POINTER(AxmanOutput)]
_lib.axman_output_free.restype = None

_lib.axman_hypergraph_validate.argtypes = [ctypes.POINTER(AxmanHypergraph)]
_lib.axman_hypergraph_validate.restype = ctypes.c_int

_lib.axman_hypergraph_hash.argtypes = [ctypes.POINTER(AxmanHypergraph), ctypes.POINTER(ctypes.c_uint64)]
_lib.axman_hypergraph_hash.restype = ctypes.c_int

_lib.axman_status_to_string.argtypes = [ctypes.c_int]
_lib.axman_status_to_string.restype = ctypes.c_char_p

_lib.axman_get_last_error_message.argtypes = []
_lib.axman_get_last_error_message.restype = ctypes.c_char_p

_lib.axman_get_version.argtypes = []
_lib.axman_get_version.restype = ctypes.c_char_p
