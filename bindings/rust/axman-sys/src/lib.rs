#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::os::raw::{c_char, c_float, c_void};


pub const AXMAN_VERSION_MAJOR: u32 = 0;
pub const AXMAN_VERSION_MINOR: u32 = 1;
pub const AXMAN_VERSION_PATCH: u32 = 0;
pub const AXMAN_VERSION_STRING: &str = "0.1.0";
pub const AXMAN_MAX_HYPEREDGE_ARITY: usize = 16;

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum axman_status_t {
    AXMAN_SUCCESS = 0,
    AXMAN_ERR_INVALID_ARGUMENT = 1,
    AXMAN_ERR_OUT_OF_MEMORY = 2,
    AXMAN_ERR_DEVICE_UNAVAILABLE = 3,
    AXMAN_ERR_INFERENCE_FAILURE = 4,
    AXMAN_ERR_NULL_POINTER = 5,
    AXMAN_ERR_ARITY_OVERFLOW = 6,
    AXMAN_ERR_INDEX_OUT_OF_BOUNDS = 7,
    AXMAN_ERR_INVALID_TOPOLOGY = 8,
    AXMAN_ERR_INTERNAL = 99,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum axman_dtype_t {
    AXMAN_DTYPE_FLOAT32 = 0,
    AXMAN_DTYPE_BFLOAT16 = 1,
    AXMAN_DTYPE_INT64 = 2,
    AXMAN_DTYPE_INT32 = 3,
    AXMAN_DTYPE_FLOAT16 = 4,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum axman_device_t {
    AXMAN_DEVICE_CPU = 0,
    AXMAN_DEVICE_CUDA = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct axman_tensor_t {
    pub data: *mut c_void,
    pub shape: *mut i64,
    pub ndim: usize,
    pub dtype: i32,
    pub device: i32,
}

impl Default for axman_tensor_t {
    fn default() -> Self {
        Self {
            data: std::ptr::null_mut(),
            shape: std::ptr::null_mut(),
            ndim: 0,
            dtype: axman_dtype_t::AXMAN_DTYPE_FLOAT32 as i32,
            device: axman_device_t::AXMAN_DEVICE_CPU as i32,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct axman_hypergraph_t {
    pub num_nodes: usize,
    pub num_hyperedges: usize,
    pub node_features: axman_tensor_t,
    pub edge_sources: *const i64,
    pub edge_ports: *const i32,
    pub edge_targets: *const i64,
    pub edge_offsets: *const usize,
}

impl Default for axman_hypergraph_t {
    fn default() -> Self {
        Self {
            num_nodes: 0,
            num_hyperedges: 0,
            node_features: axman_tensor_t::default(),
            edge_sources: std::ptr::null(),
            edge_ports: std::ptr::null(),
            edge_targets: std::ptr::null(),
            edge_offsets: std::ptr::null(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct axman_output_t {
    pub policy_logits: axman_tensor_t,
    pub value_estimate: c_float,
    pub latent_embedding: axman_tensor_t,
}

impl Default for axman_output_t {
    fn default() -> Self {
        Self {
            policy_logits: axman_tensor_t::default(),
            value_estimate: 0.0,
            latent_embedding: axman_tensor_t::default(),
        }
    }
}

pub enum axman_model_opaque {}
pub type axman_model_handle_t = *mut axman_model_opaque;

extern "C" {
    pub fn axman_model_create(
        model_path: *const c_char,
        config_json: *const c_char,
        out_handle: *mut axman_model_handle_t,
    ) -> axman_status_t;

    pub fn axman_model_forward(
        handle: axman_model_handle_t,
        input_graph: *const axman_hypergraph_t,
        out_result: *mut axman_output_t,
    ) -> axman_status_t;

    pub fn axman_model_destroy(handle: axman_model_handle_t) -> axman_status_t;

    pub fn axman_output_free(output: *mut axman_output_t);

    pub fn axman_dtype_element_size(dtype: axman_dtype_t) -> usize;

    pub fn axman_tensor_alloc(
        shape: *const i64,
        ndim: usize,
        dtype: axman_dtype_t,
        out_tensor: *mut axman_tensor_t,
    ) -> axman_status_t;

    pub fn axman_tensor_free(tensor: *mut axman_tensor_t);

    pub fn axman_hypergraph_validate(graph: *const axman_hypergraph_t) -> axman_status_t;

    pub fn axman_hypergraph_hash(
        graph: *const axman_hypergraph_t,
        out_hash: *mut u64,
    ) -> axman_status_t;

    pub fn axman_status_to_string(status: axman_status_t) -> *const c_char;

    pub fn axman_get_last_error_message() -> *const c_char;

    pub fn axman_get_version() -> *const c_char;
}
