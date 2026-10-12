/**
 * @file run_all_phase0_tests.c
 * @brief Master verification suite executing all Phase 0 tests (0.1 - 0.4)
 */

#include "axman_c_api.h"
#include "axman_currying.h"
#include "axman_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

/* Phase 0.1 Verification */
static void test_p0_1_c_abi(void) {
    printf("[Phase 0.1] Verifying C-ABI layouts and limits...\n");
    assert(sizeof(axman_tensor_t) == 32);
    assert(sizeof(axman_hypergraph_t) == 80);
    assert(sizeof(axman_output_t) == 72);
    assert(AXMAN_MAX_HYPEREDGE_ARITY == 16);
    assert(strcmp(axman_get_version(), "1.0.0") == 0);
    printf("  -> OK\n");
}

/* Phase 0.2 Verification */
static void test_p0_2_currying(void) {
    printf("[Phase 0.2] Verifying Currying Decomposition & CSR Invariants...\n");
    const size_t ARITY = 25;
    size_t offsets[] = {0, ARITY};
    int64_t targets[] = {28};
    int64_t sources[25];
    int32_t ports[25];
    for (size_t i = 0; i < ARITY; ++i) {
        sources[i] = (int64_t)i;
        ports[i] = (int32_t)i;
    }

    axman_hypergraph_t g;
    memset(&g, 0, sizeof(g));
    g.num_nodes = 30;
    g.num_hyperedges = 1;
    g.edge_offsets = offsets;
    g.edge_targets = targets;
    g.edge_sources = sources;
    g.edge_ports = ports;

    /* Arity > 16 must fail before decomposition */
    assert(axman_hypergraph_check_invariants(&g, 16) == AXMAN_ERR_ARITY_OVERFLOW);

    /* Decompose */
    axman_currying_result_t res;
    assert(axman_hypergraph_curry_decompose(&g, NULL, &res) == AXMAN_SUCCESS);
    assert(res.was_decomposed == true);
    assert(axman_hypergraph_check_invariants(&res.graph, 16) == AXMAN_SUCCESS);

    axman_currying_result_free(&res);
    printf("  -> OK\n");
}

/* Phase 0.3 Verification */
static void test_p0_3_memory_model(void) {
    printf("[Phase 0.3] Verifying Memory Ownership & Safe Output Freeing...\n");
    axman_output_t out;
    memset(&out, 0, sizeof(out));

    int64_t s_pol[] = {1, 64};
    int64_t s_lat[] = {1, 128};
    assert(axman_tensor_alloc(s_pol, 2, AXMAN_DTYPE_FLOAT32, &out.policy_logits) == AXMAN_SUCCESS);
    assert(axman_tensor_alloc(s_lat, 2, AXMAN_DTYPE_FLOAT32, &out.latent_embedding) == AXMAN_SUCCESS);
    out.value_estimate = 0.99f;

    /* Single and double free */
    axman_output_free(&out);
    assert(out.policy_logits.data == NULL);
    assert(out.latent_embedding.data == NULL);
    assert(out.value_estimate == 0.0f);

    axman_output_free(&out); /* Must not crash */
    printf("  -> OK\n");
}

/* Phase 0.4 Verification */
static void test_p0_4_structural_hashing(void) {
    printf("[Phase 0.4] Verifying Deterministic XXH64 Structural Hashing...\n");
    size_t offsets[] = {0, 3};
    int64_t targets[] = {3};
    int64_t sources[] = {0, 1, 2};
    int32_t ports[]   = {0, 1, 2};

    axman_hypergraph_t g;
    memset(&g, 0, sizeof(g));
    g.num_nodes = 5;
    g.num_hyperedges = 1;
    g.edge_offsets = offsets;
    g.edge_targets = targets;
    g.edge_sources = sources;
    g.edge_ports = ports;

    uint64_t h1 = 0, h2 = 0;
    assert(axman_hypergraph_hash(&g, &h1) == AXMAN_SUCCESS);
    assert(axman_hypergraph_hash(&g, &h2) == AXMAN_SUCCESS);
    assert(h1 != 0 && h1 == h2);

    /* Perturb port */
    ports[0] = 1; ports[1] = 0;
    uint64_t h3 = 0;
    assert(axman_hypergraph_hash(&g, &h3) == AXMAN_SUCCESS);
    assert(h3 != h1);

    printf("  Deterministic Hash: 0x%016llX (Perturbed: 0x%016llX)\n",
           (unsigned long long)h1, (unsigned long long)h3);
    printf("  -> OK\n");
}

int main(void) {
    printf("=====================================================\n");
    printf("   AXMANet PHASE 0 (0.1 - 0.4) MASTER VERIFICATION   \n");
    printf("=====================================================\n");
    test_p0_1_c_abi();
    test_p0_2_currying();
    test_p0_3_memory_model();
    test_p0_4_structural_hashing();
    printf("=====================================================\n");
    printf("   >>> ALL PHASE 0 VERIFICATIONS PASSED (100%%) <<<   \n");
    printf("=====================================================\n");
    return 0;
}
