/**
 * @file axman_c_api.h
 * @brief Axiomatic eXploration & Morphism Architecture Network (AXMANet)
 *        Stable Zero-Copy C-ABI Specification
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef AXMAN_C_API_H
#define AXMAN_C_API_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------------------------- */
/* Dynamic Library Export / Import Symbol Macro                              */
/* -------------------------------------------------------------------------- */

#if defined(_WIN32) || defined(__CYGWIN__)
  #if defined(AXMAN_EXPORTS) || defined(AXMAN_BUILDING_DLL)
    #define AXMAN_API __declspec(dllexport)
  #elif defined(AXMAN_STATIC) || defined(AXMAN_STATIC_LIB)
    #define AXMAN_API
  #else
    #define AXMAN_API __declspec(dllimport)
  #endif
#else
  #if defined(__GNUC__) && __GNUC__ >= 4
    #define AXMAN_API __attribute__((visibility("default")))
  #else
    #define AXMAN_API
  #endif
#endif

/* -------------------------------------------------------------------------- */
/* Version & Dimension Constants                                              */
/* -------------------------------------------------------------------------- */

#define AXMAN_VERSION_MAJOR 1
#define AXMAN_VERSION_MINOR 0
#define AXMAN_VERSION_PATCH 0
#define AXMAN_VERSION_STRING "1.0.0"

/**
 * @brief Maximum supported hyperedge arity (k_max = 16) in a static SRAM kernel tile.
 * Hyperedges with higher arity must undergo canonical currying decomposition.
 */
#define AXMAN_MAX_HYPEREDGE_ARITY 16

/* -------------------------------------------------------------------------- */
/* Status and Error Codes                                                     */
/* -------------------------------------------------------------------------- */

typedef enum {
    AXMAN_SUCCESS                 = 0,
    AXMAN_ERR_INVALID_ARGUMENT    = 1,
    AXMAN_ERR_OUT_OF_MEMORY       = 2,
    AXMAN_ERR_DEVICE_UNAVAILABLE  = 3,
    AXMAN_ERR_INFERENCE_FAILURE   = 4,
    AXMAN_ERR_NULL_POINTER        = 5,
    AXMAN_ERR_ARITY_OVERFLOW      = 6,
    AXMAN_ERR_INDEX_OUT_OF_BOUNDS = 7,
    AXMAN_ERR_INVALID_TOPOLOGY    = 8,
    AXMAN_ERR_INTERNAL            = 99
} axman_status_t;

/* -------------------------------------------------------------------------- */
/* Tensor Metadata & Data Types                                               */
/* -------------------------------------------------------------------------- */

typedef enum {
    AXMAN_DTYPE_FLOAT32   = 0,
    AXMAN_DTYPE_BFLOAT16  = 1,
    AXMAN_DTYPE_INT64     = 2,
    AXMAN_DTYPE_INT32     = 3,
    AXMAN_DTYPE_FLOAT16   = 4,
    AXMAN_DTYPE_INT8      = 5,
    AXMAN_DTYPE_FP8_E4M3  = 6
} axman_dtype_t;

typedef enum {
    AXMAN_PRECISION_FP32     = 0,
    AXMAN_PRECISION_BF16     = 1,
    AXMAN_PRECISION_FP8_E4M3 = 2,
    AXMAN_PRECISION_INT8     = 3
} axman_precision_t;

typedef enum {
    AXMAN_DEVICE_CPU  = 0,
    AXMAN_DEVICE_CUDA = 1
} axman_device_t;

/**
 * @brief Zero-copy descriptor for dense contiguous tensors.
 */
typedef struct {
    void* data;              /**< Pointer to contiguous memory buffer */
    int64_t* shape;          /**< Dimension array [ndim] */
    size_t ndim;             /**< Number of dimensions (rank) */
    int32_t dtype;           /**< Data type (axman_dtype_t) */
    int32_t device;          /**< Device type (axman_device_t) */
} axman_tensor_t;

/* -------------------------------------------------------------------------- */
/* Port-Hypergraph Topology (CSR / COO Layout)                                */
/* -------------------------------------------------------------------------- */

/**
 * @brief Directed Port-Hypergraph G = (V, E, P) in Compressed Sparse Row (CSR) format.
 *
 * Memory ownership contract:
 * All arrays (edge_sources, edge_ports, edge_targets, edge_offsets) and node_features
 * are borrowed views allocated and managed by the caller. AXMAN functions do not free
 * or reallocate these buffers.
 */
typedef struct {
    size_t num_nodes;                /**< Number of typed objects / AST nodes in V */
    size_t num_hyperedges;           /**< Number of morphisms / hyperedges in E */

    axman_tensor_t node_features;    /**< Node feature matrix: [num_nodes, node_feat_dim] */

    const int64_t* edge_sources;     /**< Flat contiguous array of source premise node indices */
    const int32_t* edge_ports;       /**< Positional port numbers (de Bruijn binding indices 0..k-1) */
    const int64_t* edge_targets;     /**< Target node index for each hyperedge [num_hyperedges] */
    const size_t* edge_offsets;      /**< CSR offsets into edge_sources [num_hyperedges + 1] */
} axman_hypergraph_t;

/* -------------------------------------------------------------------------- */
/* Model Output Structures                                                    */
/* -------------------------------------------------------------------------- */

/**
 * @brief Output container returned by model forward pass.
 *
 * Memory ownership contract:
 * Buffers are allocated by AXMAN runtime and must be released by calling
 * axman_output_free(&output).
 */
typedef struct {
    axman_tensor_t policy_logits;    /**< Morphism transition & premise selection logits */
    float value_estimate;            /**< Distance-to-closure value scalar in [0.0, 1.0] */
    axman_tensor_t latent_embedding; /**< Latent state invariant representation */
} axman_output_t;

/**
 * @brief Opaque handle to the instantiated AXMANet model instance.
 */
typedef struct axman_model_opaque* axman_model_handle_t;

/* -------------------------------------------------------------------------- */
/* Model Lifecycle Functions                                                  */
/* -------------------------------------------------------------------------- */

/**
 * @brief Create and initialize an AXMANet model instance from file weights and JSON configuration.
 *
 * @param[in]  model_path  Path to model weights file (.safetensors / checkpoint).
 * @param[in]  config_json JSON-formatted model configuration string (or NULL for default).
 * @param[out] out_handle  Pointer to receive the created model handle.
 * @return AXMAN_SUCCESS on success, or an error status code.
 */
AXMAN_API axman_status_t axman_model_create(
    const char* model_path,
    const char* config_json,
    axman_model_handle_t* out_handle
);

/**
 * @brief Execute forward inference on an input port-hypergraph.
 *
 * @param[in]  handle      Valid model handle.
 * @param[in]  input_graph Borrowed input hypergraph structure.
 * @param[out] out_result  Pointer to receive the model outputs.
 * @return AXMAN_SUCCESS on success, or an error status code.
 */
AXMAN_API axman_status_t axman_model_forward(
    axman_model_handle_t handle,
    const axman_hypergraph_t* input_graph,
    axman_output_t* out_result
);

/**
 * @brief Destroy a model instance and free all internal resources.
 *
 * @param[in] handle Model handle to destroy.
 * @return AXMAN_SUCCESS on success, or an error status code.
 */
AXMAN_API axman_status_t axman_model_destroy(
    axman_model_handle_t handle
);

/**
 * @brief Release heap memory allocated inside an axman_output_t container.
 * Idempotent: safe to call multiple times or on zeroed structs.
 *
 * @param[in,out] output Pointer to the output container to release.
 */
AXMAN_API void axman_output_free(
    axman_output_t* output
);

/* -------------------------------------------------------------------------- */
/* Tensor Memory Management & Helpers                                         */
/* -------------------------------------------------------------------------- */

/**
 * @brief Get byte size for a given tensor data type.
 */
AXMAN_API size_t axman_dtype_element_size(
    axman_dtype_t dtype
);

/**
 * @brief Allocate a dense contiguous tensor buffer on CPU memory.
 *
 * @param[in]  shape      Array of dimension extents [ndim].
 * @param[in]  ndim       Number of axes.
 * @param[in]  dtype      Data type enum.
 * @param[out] out_tensor Pointer to tensor structure to initialize.
 * @return AXMAN_SUCCESS on success, or error status code.
 */
AXMAN_API axman_status_t axman_tensor_alloc(
    const int64_t* shape,
    size_t ndim,
    axman_dtype_t dtype,
    axman_tensor_t* out_tensor
);

/**
 * @brief Release memory associated with an allocated tensor.
 * Idempotent: resets pointers to NULL and ndim to 0.
 *
 * @param[in,out] tensor Pointer to tensor to release.
 */
AXMAN_API void axman_tensor_free(
    axman_tensor_t* tensor
);

/* -------------------------------------------------------------------------- */
/* Graph Validation & Structural Hashing (Phase 0.4 API)                      */
/* -------------------------------------------------------------------------- */

/**
 * @brief Validate the structural integrity of a port-hypergraph.
 * Checks bounds, offset monotonicity, arity constraints (k <= AXMAN_MAX_HYPEREDGE_ARITY).
 *
 * @param[in] graph Pointer to the hypergraph structure to validate.
 * @return AXMAN_SUCCESS if valid, or corresponding error code.
 */
AXMAN_API axman_status_t axman_hypergraph_validate(
    const axman_hypergraph_t* graph
);

/**
 * @brief Compute a 64-bit deterministic structural hash of the port-hypergraph.
 *
 * @param[in]  graph    Pointer to the hypergraph structure.
 * @param[out] out_hash Pointer to receive the 64-bit hash.
 * @return AXMAN_SUCCESS on success, or an error status code.
 */
AXMAN_API axman_status_t axman_hypergraph_hash(
    const axman_hypergraph_t* graph,
    uint64_t* out_hash
);

/* -------------------------------------------------------------------------- */
/* Diagnostics & Metadata                                                     */
/* -------------------------------------------------------------------------- */

/**
 * @brief Convert a status code into a human-readable English string.
 */
AXMAN_API const char* axman_status_to_string(
    axman_status_t status
);

/**
 * @brief Retrieve thread-local last error diagnostic message.
 */
AXMAN_API const char* axman_get_last_error_message(void);

/**
 * @brief Retrieve library version string.
 */
AXMAN_API const char* axman_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* AXMAN_C_API_H */
