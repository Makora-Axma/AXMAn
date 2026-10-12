/**
 * @file memory.c
 * @brief Memory Management, Zero-Copy Views, and Tensor Lifecycles for AXMANet
 *
 * Implements strict ownership semantics:
 * - Input graphs are borrowed views owned by the caller.
 * - Outputs are allocated by runtime and released via axman_output_free.
 * - Idempotent, double-free safe memory reclamation.
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#include "axman_c_api.h"
#include <stdlib.h>
#include <string.h>

/* Thread-local last error buffer */
#if defined(__GNUC__) || defined(__clang__)
static __thread char g_last_error_msg[256] = "AXMAN: No error";
#elif defined(_MSC_VER)
static __declspec(thread) char g_last_error_msg[256] = "AXMAN: No error";
#else
static char g_last_error_msg[256] = "AXMAN: No error";
#endif

/* -------------------------------------------------------------------------- */
/* Tensor Element Size & Allocation                                           */
/* -------------------------------------------------------------------------- */

AXMAN_API size_t axman_dtype_element_size(axman_dtype_t dtype) {
    switch (dtype) {
        case AXMAN_DTYPE_FLOAT32:  return sizeof(float);
        case AXMAN_DTYPE_BFLOAT16: return 2;
        case AXMAN_DTYPE_INT64:    return sizeof(int64_t);
        case AXMAN_DTYPE_INT32:    return sizeof(int32_t);
        case AXMAN_DTYPE_FLOAT16:  return 2;
        case AXMAN_DTYPE_INT8:     return 1;
        case AXMAN_DTYPE_FP8_E4M3: return 1;
        default:                   return 0;
    }
}

AXMAN_API axman_status_t axman_tensor_alloc(
    const int64_t* shape,
    size_t ndim,
    axman_dtype_t dtype,
    axman_tensor_t* out_tensor
) {
    if (!out_tensor) {
        return AXMAN_ERR_NULL_POINTER;
    }

    memset(out_tensor, 0, sizeof(*out_tensor));

    if (ndim > 0 && !shape) {
        return AXMAN_ERR_INVALID_ARGUMENT;
    }

    size_t elem_size = axman_dtype_element_size(dtype);
    if (elem_size == 0) {
        return AXMAN_ERR_INVALID_ARGUMENT;
    }

    size_t total_elements = 1;
    if (ndim == 0) {
        total_elements = 1; /* Scalar tensor */
    } else {
        for (size_t i = 0; i < ndim; ++i) {
            if (shape[i] < 0) {
                return AXMAN_ERR_INVALID_ARGUMENT;
            }
            total_elements *= (size_t)shape[i];
        }
    }

    int64_t* shape_copy = NULL;
    if (ndim > 0) {
        shape_copy = (int64_t*)malloc(ndim * sizeof(int64_t));
        if (!shape_copy) {
            return AXMAN_ERR_OUT_OF_MEMORY;
        }
        memcpy(shape_copy, shape, ndim * sizeof(int64_t));
    }

    void* data_buf = malloc(total_elements * elem_size);
    if (!data_buf && total_elements > 0) {
        free(shape_copy);
        return AXMAN_ERR_OUT_OF_MEMORY;
    }

    if (data_buf) {
        memset(data_buf, 0, total_elements * elem_size);
    }

    out_tensor->data = data_buf;
    out_tensor->shape = shape_copy;
    out_tensor->ndim = ndim;
    out_tensor->dtype = (int32_t)dtype;
    out_tensor->device = (int32_t)AXMAN_DEVICE_CPU;

    return AXMAN_SUCCESS;
}

AXMAN_API void axman_tensor_free(axman_tensor_t* tensor) {
    if (!tensor) {
        return;
    }

    if (tensor->data) {
        free(tensor->data);
        tensor->data = NULL;
    }

    if (tensor->shape) {
        free(tensor->shape);
        tensor->shape = NULL;
    }

    tensor->ndim = 0;
    tensor->dtype = 0;
    tensor->device = 0;
}

AXMAN_API void axman_output_free(axman_output_t* output) {
    if (!output) {
        return;
    }

    axman_tensor_free(&output->policy_logits);
    axman_tensor_free(&output->latent_embedding);
    output->value_estimate = 0.0f;
}

/* -------------------------------------------------------------------------- */
/* Diagnostics & Version Info                                                 */
/* -------------------------------------------------------------------------- */

AXMAN_API const char* axman_status_to_string(axman_status_t status) {
    switch (status) {
        case AXMAN_SUCCESS:                 return "AXMAN_SUCCESS: Operation completed successfully";
        case AXMAN_ERR_INVALID_ARGUMENT:    return "AXMAN_ERR_INVALID_ARGUMENT: Invalid parameter supplied";
        case AXMAN_ERR_OUT_OF_MEMORY:       return "AXMAN_ERR_OUT_OF_MEMORY: Memory allocation failure";
        case AXMAN_ERR_DEVICE_UNAVAILABLE:  return "AXMAN_ERR_DEVICE_UNAVAILABLE: Requested accelerator is unavailable";
        case AXMAN_ERR_INFERENCE_FAILURE:   return "AXMAN_ERR_INFERENCE_FAILURE: Execution failure during forward pass";
        case AXMAN_ERR_NULL_POINTER:        return "AXMAN_ERR_NULL_POINTER: Unexpected NULL pointer passed";
        case AXMAN_ERR_ARITY_OVERFLOW:      return "AXMAN_ERR_ARITY_OVERFLOW: Hyperedge arity exceeds limit (k > 16)";
        case AXMAN_ERR_INDEX_OUT_OF_BOUNDS: return "AXMAN_ERR_INDEX_OUT_OF_BOUNDS: Node index out of bounds [0, num_nodes-1]";
        case AXMAN_ERR_INVALID_TOPOLOGY:    return "AXMAN_ERR_INVALID_TOPOLOGY: Malformed CSR offsets or cycle";
        case AXMAN_ERR_INTERNAL:            return "AXMAN_ERR_INTERNAL: Internal unrecoverable engine failure";
        default:                            return "AXMAN_ERR_UNKNOWN: Unrecognized status code";
    }
}

/* External hook to C++ engine error buffer */
#ifdef __cplusplus
extern "C" {
#endif
const char* axman_engine_get_last_error(void);
void axman_set_last_error_message(const char* msg);
#ifdef __cplusplus
}
#endif

void axman_set_last_error_message(const char* msg) {
    if (msg) {
        strncpy(g_last_error_msg, msg, sizeof(g_last_error_msg) - 1);
        g_last_error_msg[sizeof(g_last_error_msg) - 1] = '\0';
    }
}

AXMAN_API const char* axman_get_last_error_message(void) {
    if (g_last_error_msg[0] != '\0' && strcmp(g_last_error_msg, "AXMAN: No error") != 0) {
        return g_last_error_msg;
    }
    const char* engine_err = axman_engine_get_last_error();
    if (engine_err && engine_err[0] != '\0') {
        return engine_err;
    }
    return g_last_error_msg;
}

AXMAN_API const char* axman_get_version(void) {
    return AXMAN_VERSION_STRING;
}
