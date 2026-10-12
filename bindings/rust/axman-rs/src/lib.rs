//! # axman-rs: High-Level Safe Rust Bindings for AXMANet
//!
//! Provides idiomatic RAII wrappers, safe type conversions, Send + Sync model handles,
//! and zero-copy view abstractions over the stable C-ABI (`libaxman`).
//!
//! Copyright (c) 2026 Project Makora / AXMA Authors.

use std::ffi::{CStr, CString};
use std::fmt;
use std::path::Path;
use thiserror::Error;

pub use axman_sys::{
    axman_device_t, axman_dtype_t, axman_status_t, AXMAN_MAX_HYPEREDGE_ARITY, AXMAN_VERSION_STRING,
};

/// Errors returned by AXMANet operations.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum AxmanError {
    #[error("Invalid argument passed to AXMANet: {message}")]
    InvalidArgument { message: String },

    #[error("Out of memory: {message}")]
    OutOfMemory { message: String },

    #[error("Compute device unavailable: {message}")]
    DeviceUnavailable { message: String },

    #[error("Inference execution failure: {message}")]
    InferenceFailure { message: String },

    #[error("Null pointer passed across C-ABI: {message}")]
    NullPointer { message: String },

    #[error("Hyperedge arity exceeded k_max = 16: {message}")]
    ArityOverflow { message: String },

    #[error("Index out of bounds in topology: {message}")]
    IndexOutOfBounds { message: String },

    #[error("Invalid topology invariant: {message}")]
    InvalidTopology { message: String },

    #[error("Internal runtime error: {message}")]
    Internal { message: String },

    #[error("Nul byte in string argument")]
    NulError,
}

impl AxmanError {
    fn from_status(status: axman_status_t) -> Self {
        let msg = unsafe {
            let p = axman_sys::axman_get_last_error_message();
            if p.is_null() {
                let fallback = axman_sys::axman_status_to_string(status);
                CStr::from_ptr(fallback).to_string_lossy().into_owned()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };

        match status {
            axman_status_t::AXMAN_ERR_INVALID_ARGUMENT => AxmanError::InvalidArgument { message: msg },
            axman_status_t::AXMAN_ERR_OUT_OF_MEMORY => AxmanError::OutOfMemory { message: msg },
            axman_status_t::AXMAN_ERR_DEVICE_UNAVAILABLE => AxmanError::DeviceUnavailable { message: msg },
            axman_status_t::AXMAN_ERR_INFERENCE_FAILURE => AxmanError::InferenceFailure { message: msg },
            axman_status_t::AXMAN_ERR_NULL_POINTER => AxmanError::NullPointer { message: msg },
            axman_status_t::AXMAN_ERR_ARITY_OVERFLOW => AxmanError::ArityOverflow { message: msg },
            axman_status_t::AXMAN_ERR_INDEX_OUT_OF_BOUNDS => AxmanError::IndexOutOfBounds { message: msg },
            axman_status_t::AXMAN_ERR_INVALID_TOPOLOGY => AxmanError::InvalidTopology { message: msg },
            _ => AxmanError::Internal { message: msg },
        }
    }
}

pub type Result<T> = std::result::Result<T, AxmanError>;

/// Safe view over an AXMAN tensor (e.g. logits or embeddings).
pub struct TensorView<'a, T> {
    data: &'a [T],
    shape: Vec<i64>,
}

impl<'a, T: fmt::Debug> fmt::Debug for TensorView<'a, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("shape", &self.shape)
            .field("len", &self.data.len())
            .finish()
    }
}

impl<'a, T> TensorView<'a, T> {
    #[inline]
    pub fn as_slice(&self) -> &'a [T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Access element in a 2D tensor by (row, col) coordinates.
    pub fn get_2d(&self, row: usize, col: usize) -> Option<&'a T> {
        if self.shape.len() != 2 {
            return None;
        }
        let rows = self.shape[0] as usize;
        let cols = self.shape[1] as usize;
        if row >= rows || col >= cols {
            return None;
        }
        let idx = row * cols + col;
        self.data.get(idx)
    }
}

impl<'a, T> std::ops::Deref for TensorView<'a, T> {
    type Target = [T];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<'a, T> std::ops::Index<usize> for TensorView<'a, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

/// Helper to safely construct a `TensorView<'a, f32>` from raw C-ABI tensor descriptor.
fn tensor_view_from_raw<'a>(raw: &'a axman_sys::axman_tensor_t) -> TensorView<'a, f32> {
    if raw.ndim == 0 || raw.shape.is_null() || raw.data.is_null() {
        let shape = if raw.shape.is_null() || raw.ndim == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(raw.shape, raw.ndim).to_vec() }
        };
        return TensorView { data: &[], shape };
    }

    let mut total = 1usize;
    for i in 0..raw.ndim {
        let dim = unsafe { *raw.shape.add(i) };
        if dim <= 0 {
            total = 0;
            break;
        }
        total = match total.checked_mul(dim as usize) {
            Some(v) => v,
            None => {
                total = 0;
                break;
            }
        };
    }

    let shape = unsafe { std::slice::from_raw_parts(raw.shape, raw.ndim).to_vec() };
    let slice = if total == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(raw.data as *const f32, total) }
    };

    TensorView { data: slice, shape }
}

/// Owned output of an AXMANet model inference execution.
/// Automatically releases internal heap buffers via `axman_output_free` on Drop.
pub struct AxmanOutput {
    raw: axman_sys::axman_output_t,
}

impl AxmanOutput {
    /// Distance-to-closure value scalar in [0.0, 1.0].
    pub fn value_estimate(&self) -> f32 {
        self.raw.value_estimate
    }

    /// View into morphism transition & premise selection policy logits (shape: [num_nodes, num_actions]).
    pub fn policy_logits(&self) -> TensorView<'_, f32> {
        tensor_view_from_raw(&self.raw.policy_logits)
    }

    /// View into latent invariant embedding (shape: [1, d_model]).
    pub fn latent_embedding(&self) -> TensorView<'_, f32> {
        tensor_view_from_raw(&self.raw.latent_embedding)
    }
}

impl Drop for AxmanOutput {
    fn drop(&mut self) {
        unsafe {
            axman_sys::axman_output_free(&mut self.raw);
        }
    }
}

/// Port-Hypergraph representation borrowed by the C-ABI during forward inference.
pub struct Hypergraph<'a> {
    num_nodes: usize,
    num_hyperedges: usize,
    node_features: &'a [f32],
    node_feat_dim: usize,
    edge_sources: &'a [i64],
    edge_ports: &'a [i32],
    edge_targets: &'a [i64],
    edge_offsets: &'a [usize],
}

impl<'a> Hypergraph<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_nodes: usize,
        num_hyperedges: usize,
        node_features: &'a [f32],
        node_feat_dim: usize,
        edge_sources: &'a [i64],
        edge_ports: &'a [i32],
        edge_targets: &'a [i64],
        edge_offsets: &'a [usize],
    ) -> Result<Self> {
        if node_features.len() != num_nodes * node_feat_dim {
            return Err(AxmanError::InvalidArgument {
                message: format!(
                    "node_features size mismatch: expected {}, got {}",
                    num_nodes * node_feat_dim,
                    node_features.len()
                ),
            });
        }
        if edge_targets.len() != num_hyperedges {
            return Err(AxmanError::InvalidArgument {
                message: format!(
                    "edge_targets size mismatch: expected {}, got {}",
                    num_hyperedges,
                    edge_targets.len()
                ),
            });
        }
        if edge_offsets.len() != num_hyperedges + 1 {
            return Err(AxmanError::InvalidArgument {
                message: format!(
                    "edge_offsets size mismatch: expected {}, got {}",
                    num_hyperedges + 1,
                    edge_offsets.len()
                ),
            });
        }
        if edge_sources.len() != edge_ports.len() {
            return Err(AxmanError::InvalidArgument {
                message: format!(
                    "edge_sources and edge_ports length mismatch: {} != {}",
                    edge_sources.len(),
                    edge_ports.len()
                ),
            });
        }
        if edge_offsets[0] != 0 {
            return Err(AxmanError::InvalidTopology {
                message: format!("edge_offsets[0] must be 0, got {}", edge_offsets[0]),
            });
        }
        if edge_offsets[num_hyperedges] != edge_sources.len() {
            return Err(AxmanError::InvalidTopology {
                message: format!(
                    "edge_offsets[{}] must match edge_sources length ({}), got {}",
                    num_hyperedges,
                    edge_sources.len(),
                    edge_offsets[num_hyperedges]
                ),
            });
        }

        let graph = Self {
            num_nodes,
            num_hyperedges,
            node_features,
            node_feat_dim,
            edge_sources,
            edge_ports,
            edge_targets,
            edge_offsets,
        };

        // Validate using C-ABI validator
        graph.validate()?;

        Ok(graph)
    }

    #[inline]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    #[inline]
    pub fn num_hyperedges(&self) -> usize {
        self.num_hyperedges
    }

    #[inline]
    pub fn node_feat_dim(&self) -> usize {
        self.node_feat_dim
    }

    #[inline]
    pub fn node_features(&self) -> &'a [f32] {
        self.node_features
    }

    #[inline]
    pub fn edge_sources(&self) -> &'a [i64] {
        self.edge_sources
    }

    #[inline]
    pub fn edge_ports(&self) -> &'a [i32] {
        self.edge_ports
    }

    #[inline]
    pub fn edge_targets(&self) -> &'a [i64] {
        self.edge_targets
    }

    #[inline]
    pub fn edge_offsets(&self) -> &'a [usize] {
        self.edge_offsets
    }

    /// Returns the arity (number of input premises) of the hyperedge at `edge_idx`.
    pub fn arity(&self, edge_idx: usize) -> Option<usize> {
        if edge_idx >= self.num_hyperedges {
            None
        } else {
            Some(self.edge_offsets[edge_idx + 1] - self.edge_offsets[edge_idx])
        }
    }

    /// Structural validation of graph topology.
    pub fn validate(&self) -> Result<()> {
        let mut shape = [self.num_nodes as i64, self.node_feat_dim as i64];
        let raw = self.as_raw(&mut shape);
        let status = unsafe { axman_sys::axman_hypergraph_validate(&raw) };
        if status != axman_status_t::AXMAN_SUCCESS {
            return Err(AxmanError::from_status(status));
        }
        Ok(())
    }

    /// Computes the 64-bit structural hash of the port-hypergraph.
    pub fn structural_hash(&self) -> Result<u64> {
        let mut shape = [self.num_nodes as i64, self.node_feat_dim as i64];
        let raw = self.as_raw(&mut shape);
        let mut hash = 0u64;
        let status = unsafe { axman_sys::axman_hypergraph_hash(&raw, &mut hash) };
        if status != axman_status_t::AXMAN_SUCCESS {
            return Err(AxmanError::from_status(status));
        }
        Ok(hash)
    }

    fn as_raw(&self, shape_buf: &mut [i64; 2]) -> axman_sys::axman_hypergraph_t {
        shape_buf[0] = self.num_nodes as i64;
        shape_buf[1] = self.node_feat_dim as i64;

        let node_features_tensor = if self.node_features.is_empty() || self.num_nodes == 0 {
            axman_sys::axman_tensor_t::default()
        } else {
            axman_sys::axman_tensor_t {
                data: self.node_features.as_ptr() as *mut std::os::raw::c_void,
                shape: shape_buf.as_mut_ptr(),
                ndim: 2,
                dtype: axman_sys::axman_dtype_t::AXMAN_DTYPE_FLOAT32 as i32,
                device: axman_sys::axman_device_t::AXMAN_DEVICE_CPU as i32,
            }
        };

        axman_sys::axman_hypergraph_t {
            num_nodes: self.num_nodes,
            num_hyperedges: self.num_hyperedges,
            node_features: node_features_tensor,
            edge_sources: if self.edge_sources.is_empty() {
                std::ptr::null()
            } else {
                self.edge_sources.as_ptr()
            },
            edge_ports: if self.edge_ports.is_empty() {
                std::ptr::null()
            } else {
                self.edge_ports.as_ptr()
            },
            edge_targets: if self.edge_targets.is_empty() {
                std::ptr::null()
            } else {
                self.edge_targets.as_ptr()
            },
            edge_offsets: if self.edge_offsets.is_empty() {
                std::ptr::null()
            } else {
                self.edge_offsets.as_ptr()
            },
        }
    }
}

/// Strongly-typed neural model configuration for AXMANet.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelConfig {
    pub d_model: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub port_dim: usize,
    pub num_kinds: usize,
    pub in_features: usize,
    pub num_actions: usize,
    pub max_ports: usize,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::base()
    }
}

impl ModelConfig {
    /// Small model configuration preset (d_model=256, layers=6, heads=8, port_dim=64).
    pub fn small() -> Self {
        Self {
            d_model: 256,
            num_layers: 6,
            num_heads: 8,
            port_dim: 64,
            num_kinds: 16,
            in_features: 64,
            num_actions: 128,
            max_ports: 16,
        }
    }

    /// Base model configuration preset (d_model=512, layers=12, heads=16, port_dim=128).
    pub fn base() -> Self {
        Self {
            d_model: 512,
            num_layers: 12,
            num_heads: 16,
            port_dim: 128,
            num_kinds: 16,
            in_features: 64,
            num_actions: 256,
            max_ports: 16,
        }
    }

    /// Large model configuration preset (d_model=1024, layers=24, heads=16, port_dim=256).
    pub fn large() -> Self {
        Self {
            d_model: 1024,
            num_layers: 24,
            num_heads: 16,
            port_dim: 256,
            num_kinds: 16,
            in_features: 128,
            num_actions: 512,
            max_ports: 16,
        }
    }

    /// Serialize configuration to JSON string accepted by C-ABI.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"d_model": {}, "num_layers": {}, "num_heads": {}, "port_dim": {}, "num_kinds": {}, "in_features": {}, "num_actions": {}, "max_ports": {}}}"#,
            self.d_model,
            self.num_layers,
            self.num_heads,
            self.port_dim,
            self.num_kinds,
            self.in_features,
            self.num_actions,
            self.max_ports
        )
    }
}

/// RAII wrapper over `axman_model_handle_t`.
/// Guaranteed thread-safe for concurrent read-only forward inference (`Send` + `Sync`).
pub struct AxmanModel {
    handle: axman_sys::axman_model_handle_t,
}

// SAFETY: axman_model_forward is thread-safe on read-only forward pass.
unsafe impl Send for AxmanModel {}
unsafe impl Sync for AxmanModel {}

impl AxmanModel {
    /// Load an AXMANet model from weights file (or None) and optional JSON configuration.
    pub fn load<P: AsRef<Path>>(model_path: Option<P>, config_json: Option<&str>) -> Result<Self> {
        let c_path = match model_path {
            Some(p) => {
                let s = p.as_ref().to_str().ok_or(AxmanError::InvalidArgument {
                    message: "Invalid non-UTF8 path".into(),
                })?;
                Some(CString::new(s).map_err(|_| AxmanError::NulError)?)
            }
            None => None,
        };

        let c_json = match config_json {
            Some(j) => Some(CString::new(j).map_err(|_| AxmanError::NulError)?),
            None => None,
        };

        let path_ptr = c_path.as_ref().map_or(std::ptr::null(), |p| p.as_ptr());
        let json_ptr = c_json.as_ref().map_or(std::ptr::null(), |j| j.as_ptr());

        let mut handle = std::ptr::null_mut();
        let status = unsafe { axman_sys::axman_model_create(path_ptr, json_ptr, &mut handle) };

        if status != axman_status_t::AXMAN_SUCCESS {
            return Err(AxmanError::from_status(status));
        }

        Ok(Self { handle })
    }

    /// Load model directly from weights file with optional configuration JSON.
    pub fn from_file<P: AsRef<Path>>(model_path: P, config_json: Option<&str>) -> Result<Self> {
        Self::load(Some(model_path), config_json)
    }

    /// Initialize model with default random weights according to configuration JSON.
    pub fn from_config(config_json: &str) -> Result<Self> {
        Self::load(None::<&Path>, Some(config_json))
    }

    /// Initialize model with a typed ModelConfig.
    pub fn from_model_config(config: &ModelConfig) -> Result<Self> {
        let json = config.to_json();
        Self::from_config(&json)
    }

    /// Initialize model with default configuration.
    pub fn default_model() -> Result<Self> {
        Self::load(None::<&Path>, None)
    }

    /// Execute forward inference on an input port-hypergraph.
    pub fn forward(&self, graph: &Hypergraph<'_>) -> Result<AxmanOutput> {
        let mut shape_buf = [0i64; 2];
        let raw_graph = graph.as_raw(&mut shape_buf);

        let mut out = axman_sys::axman_output_t::default();
        let status = unsafe { axman_sys::axman_model_forward(self.handle, &raw_graph, &mut out) };

        if status != axman_status_t::AXMAN_SUCCESS {
            return Err(AxmanError::from_status(status));
        }

        Ok(AxmanOutput { raw: out })
    }

    /// Returns the library version string.
    pub fn version(&self) -> &'static str {
        version()
    }
}

impl Drop for AxmanModel {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                axman_sys::axman_model_destroy(self.handle);
            }
            self.handle = std::ptr::null_mut();
        }
    }
}

/// Returns the library version string.
pub fn version() -> &'static str {
    unsafe {
        let p = axman_sys::axman_get_version();
        CStr::from_ptr(p).to_str().unwrap_or(AXMAN_VERSION_STRING)
    }
}
