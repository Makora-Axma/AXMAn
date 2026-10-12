/**
 * @file test_currying.c
 * @brief Unit tests for Port-Hypergraph CSR invariant validation & Currying Decomposition Pass
 */

#include "axman_currying.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void test_normal_graph_within_arity_limit(void) {
    printf("Running test_normal_graph_within_arity_limit...\n");

    /* Create graph with 10 nodes, 2 hyperedges with arities 3 and 4 (<= 16) */
    size_t offsets[] = {0, 3, 7};
    int64_t targets[] = {3, 9};
    int64_t sources[] = {0, 1, 2, 4, 5, 6, 7};
    int32_t ports[]   = {0, 1, 2, 0, 1, 2, 3};

    axman_hypergraph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 10;
    graph.num_hyperedges = 2;
    graph.edge_offsets = offsets;
    graph.edge_targets = targets;
    graph.edge_sources = sources;
    graph.edge_ports = ports;

    axman_status_t status = axman_hypergraph_check_invariants(&graph, AXMAN_MAX_HYPEREDGE_ARITY);
    assert(status == AXMAN_SUCCESS);

    axman_currying_result_t result;
    status = axman_hypergraph_curry_decompose(&graph, NULL, &result);
    assert(status == AXMAN_SUCCESS);
    assert(result.was_decomposed == false);
    assert(result.num_auxiliary_nodes == 0);
    assert(result.graph.num_hyperedges == 2);
    assert(result.graph.num_nodes == 10);

    axman_currying_result_free(&result);
    printf("  -> PASS\n");
}

static void test_high_arity_currying_decomposition(void) {
    printf("Running test_high_arity_currying_decomposition...\n");

    /* Create a graph with 35 nodes and 1 large hyperedge with arity 28 (k = 28 > 16) */
    const size_t ARITY = 28;
    size_t offsets[] = {0, ARITY};
    int64_t targets[] = {30};

    int64_t sources[28];
    int32_t ports[28];
    for (size_t i = 0; i < ARITY; ++i) {
        sources[i] = (int64_t)i;
        ports[i] = (int32_t)i;
    }

    axman_hypergraph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 35;
    graph.num_hyperedges = 1;
    graph.edge_offsets = offsets;
    graph.edge_targets = targets;
    graph.edge_sources = sources;
    graph.edge_ports = ports;

    /* Direct check should fail with arity overflow */
    axman_status_t status = axman_hypergraph_check_invariants(&graph, AXMAN_MAX_HYPEREDGE_ARITY);
    assert(status == AXMAN_ERR_ARITY_OVERFLOW);

    /* Run currying decomposition */
    axman_currying_result_t result;
    status = axman_hypergraph_curry_decompose(&graph, NULL, &result);
    assert(status == AXMAN_SUCCESS);
    assert(result.was_decomposed == true);
    assert(result.num_auxiliary_nodes > 0);
    assert(result.graph.num_hyperedges > 1);

    /* Verify that every decomposed hyperedge satisfies k <= 16 */
    for (size_t e = 0; e < result.graph.num_hyperedges; ++e) {
        size_t k = result.graph.edge_offsets[e + 1] - result.graph.edge_offsets[e];
        assert(k <= AXMAN_MAX_HYPEREDGE_ARITY);
    }

    /* Verify final target connects to node 30 */
    assert(result.graph.edge_targets[result.graph.num_hyperedges - 1] == 30);

    /* Invariants of resulting graph must strictly pass */
    status = axman_hypergraph_check_invariants(&result.graph, AXMAN_MAX_HYPEREDGE_ARITY);
    assert(status == AXMAN_SUCCESS);

    printf("  Decomposed arity %zu into %zu hyperedges with %zu auxiliary nodes (all k <= 16)\n",
           ARITY, result.graph.num_hyperedges, result.num_auxiliary_nodes);

    axman_currying_result_free(&result);
    printf("  -> PASS\n");
}

static void test_index_out_of_bounds_detection(void) {
    printf("Running test_index_out_of_bounds_detection...\n");

    size_t offsets[] = {0, 2};
    int64_t targets[] = {15}; /* 15 >= num_nodes (10) -> OUT OF BOUNDS */
    int64_t sources[] = {0, 1};
    int32_t ports[]   = {0, 1};

    axman_hypergraph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 10;
    graph.num_hyperedges = 1;
    graph.edge_offsets = offsets;
    graph.edge_targets = targets;
    graph.edge_sources = sources;
    graph.edge_ports = ports;

    axman_status_t status = axman_hypergraph_check_invariants(&graph, AXMAN_MAX_HYPEREDGE_ARITY);
    assert(status == AXMAN_ERR_INDEX_OUT_OF_BOUNDS);
    printf("  -> PASS\n");
}

static void test_invalid_topology_offsets(void) {
    printf("Running test_invalid_topology_offsets...\n");

    /* Non-monotonic offsets: 0, 3, 2 */
    size_t offsets[] = {0, 3, 2};
    int64_t targets[] = {1, 2};
    int64_t sources[] = {0, 1, 2};
    int32_t ports[]   = {0, 1, 2};

    axman_hypergraph_t graph;
    memset(&graph, 0, sizeof(graph));
    graph.num_nodes = 10;
    graph.num_hyperedges = 2;
    graph.edge_offsets = offsets;
    graph.edge_targets = targets;
    graph.edge_sources = sources;
    graph.edge_ports = ports;

    axman_status_t status = axman_hypergraph_check_invariants(&graph, AXMAN_MAX_HYPEREDGE_ARITY);
    assert(status == AXMAN_ERR_INVALID_TOPOLOGY);
    printf("  -> PASS\n");
}

int main(void) {
    printf("=== Running Port-Hypergraph & Currying Pass Test Suite ===\n");
    test_normal_graph_within_arity_limit();
    test_high_arity_currying_decomposition();
    test_index_out_of_bounds_detection();
    test_invalid_topology_offsets();
    printf("=== ALL UNIT TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
