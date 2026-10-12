/**
 * @file test_memory.c
 * @brief Unit tests for AXMANet Memory Model, Zero-Copy Tensors, and axman_output_free
 */

#include "axman_c_api.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void test_dtype_element_sizes(void) {
    printf("Running test_dtype_element_sizes...\n");
    assert(axman_dtype_element_size(AXMAN_DTYPE_FLOAT32) == 4);
    assert(axman_dtype_element_size(AXMAN_DTYPE_BFLOAT16) == 2);
    assert(axman_dtype_element_size(AXMAN_DTYPE_INT64) == 8);
    assert(axman_dtype_element_size(AXMAN_DTYPE_INT32) == 4);
    assert(axman_dtype_element_size(AXMAN_DTYPE_FLOAT16) == 2);
    assert(axman_dtype_element_size((axman_dtype_t)999) == 0);
    printf("  -> PASS\n");
}

static void test_tensor_alloc_and_free(void) {
    printf("Running test_tensor_alloc_and_free...\n");

    int64_t shape[] = {4, 128};
    axman_tensor_t tensor;
    memset(&tensor, 0, sizeof(tensor));

    axman_status_t status = axman_tensor_alloc(shape, 2, AXMAN_DTYPE_FLOAT32, &tensor);
    assert(status == AXMAN_SUCCESS);
    assert(tensor.data != NULL);
    assert(tensor.shape != NULL);
    assert(tensor.ndim == 2);
    assert(tensor.shape[0] == 4);
    assert(tensor.shape[1] == 128);
    assert(tensor.dtype == (int32_t)AXMAN_DTYPE_FLOAT32);
    assert(tensor.device == (int32_t)AXMAN_DEVICE_CPU);

    /* Verify buffer is zeroed */
    const float* fbuf = (const float*)tensor.data;
    for (size_t i = 0; i < 4 * 128; ++i) {
        assert(fbuf[i] == 0.0f);
    }

    /* Free tensor */
    axman_tensor_free(&tensor);
    assert(tensor.data == NULL);
    assert(tensor.shape == NULL);
    assert(tensor.ndim == 0);

    /* Double free idempotency check */
    axman_tensor_free(&tensor);
    assert(tensor.data == NULL);

    /* NULL pointer safety */
    axman_tensor_free(NULL);

    printf("  -> PASS\n");
}

static void test_output_free_and_double_free_resilience(void) {
    printf("Running test_output_free_and_double_free_resilience...\n");

    axman_output_t output;
    memset(&output, 0, sizeof(output));

    /* Allocate realistic policy logits: [1, 256] */
    int64_t policy_shape[] = {1, 256};
    axman_status_t status = axman_tensor_alloc(policy_shape, 2, AXMAN_DTYPE_FLOAT32, &output.policy_logits);
    assert(status == AXMAN_SUCCESS);

    /* Allocate latent embedding: [1, 512] */
    int64_t latent_shape[] = {1, 512};
    status = axman_tensor_alloc(latent_shape, 2, AXMAN_DTYPE_FLOAT32, &output.latent_embedding);
    assert(status == AXMAN_SUCCESS);

    output.value_estimate = 0.85f;

    assert(output.policy_logits.data != NULL);
    assert(output.latent_embedding.data != NULL);

    /* First release */
    axman_output_free(&output);

    assert(output.policy_logits.data == NULL);
    assert(output.policy_logits.shape == NULL);
    assert(output.latent_embedding.data == NULL);
    assert(output.latent_embedding.shape == NULL);
    assert(output.value_estimate == 0.0f);

    /* Second release (Double-Free test): must not crash */
    axman_output_free(&output);

    /* NULL release: must not crash */
    axman_output_free(NULL);

    printf("  -> PASS\n");
}

static void test_diagnostics_and_version(void) {
    printf("Running test_diagnostics_and_version...\n");

    const char* ver = axman_get_version();
    assert(ver != NULL);
    assert(strcmp(ver, "1.0.0") == 0);

    assert(strstr(axman_status_to_string(AXMAN_SUCCESS), "successfully") != NULL);
    assert(strstr(axman_status_to_string(AXMAN_ERR_INVALID_ARGUMENT), "Invalid") != NULL);
    assert(strstr(axman_status_to_string(AXMAN_ERR_OUT_OF_MEMORY), "Memory") != NULL);
    assert(strstr(axman_status_to_string(AXMAN_ERR_ARITY_OVERFLOW), "arities") != NULL ||
           strstr(axman_status_to_string(AXMAN_ERR_ARITY_OVERFLOW), "arity") != NULL);

    printf("  -> PASS\n");
}

int main(void) {
    printf("=== Running AXMANet Memory Model & Output Free Test Suite ===\n");
    test_dtype_element_sizes();
    test_tensor_alloc_and_free();
    test_output_free_and_double_free_resilience();
    test_diagnostics_and_version();
    printf("=== ALL MEMORY MODEL TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
