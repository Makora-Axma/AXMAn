/**
 * @file test_hash.c
 * @brief Unit tests for XXH64 structural hashing and deterministic graph validation
 */

#include "axman_hash.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

static void test_hash_determinism(void) {
    printf("Running test_hash_determinism...\n");

    size_t offsets[] = {0, 2, 5};
    int64_t targets[] = {3, 7};
    int64_t sources[] = {0, 1, 2, 3, 4};
    int32_t ports[]   = {0, 1, 0, 1, 2};

    axman_hypergraph_t g1;
    memset(&g1, 0, sizeof(g1));
    g1.num_nodes = 8;
    g1.num_hyperedges = 2;
    g1.edge_offsets = offsets;
    g1.edge_targets = targets;
    g1.edge_sources = sources;
    g1.edge_ports = ports;

    axman_hypergraph_t g2 = g1;

    uint64_t h1 = 0, h2 = 0;
    axman_status_t s1 = axman_hypergraph_hash(&g1, &h1);
    axman_status_t s2 = axman_hypergraph_hash(&g2, &h2);

    assert(s1 == AXMAN_SUCCESS);
    assert(s2 == AXMAN_SUCCESS);
    assert(h1 != 0);
    assert(h1 == h2);

    printf("  Hash is strictly deterministic: 0x%016llX\n", (unsigned long long)h1);
    printf("  -> PASS\n");
}

static void test_hash_sensitivity_and_avalanche(void) {
    printf("Running test_hash_sensitivity_and_avalanche...\n");

    size_t offsets[] = {0, 2, 5};
    int64_t targets[] = {3, 7};
    int64_t sources1[] = {0, 1, 2, 3, 4};
    int32_t ports1[]   = {0, 1, 0, 1, 2};

    axman_hypergraph_t base_graph;
    memset(&base_graph, 0, sizeof(base_graph));
    base_graph.num_nodes = 8;
    base_graph.num_hyperedges = 2;
    base_graph.edge_offsets = offsets;
    base_graph.edge_targets = targets;
    base_graph.edge_sources = sources1;
    base_graph.edge_ports = ports1;

    uint64_t base_hash = 0;
    assert(axman_hypergraph_hash(&base_graph, &base_hash) == AXMAN_SUCCESS);

    /* 1. Modify one source node */
    int64_t sources2[] = {0, 1, 2, 5, 4}; /* source[3] changed 3 -> 5 */
    axman_hypergraph_t g_src = base_graph;
    g_src.edge_sources = sources2;
    uint64_t hash_src = 0;
    assert(axman_hypergraph_hash(&g_src, &hash_src) == AXMAN_SUCCESS);
    assert(hash_src != base_hash);

    /* 2. Modify one target node */
    int64_t targets2[] = {3, 6}; /* target[1] changed 7 -> 6 */
    axman_hypergraph_t g_tgt = base_graph;
    g_tgt.edge_targets = targets2;
    uint64_t hash_tgt = 0;
    assert(axman_hypergraph_hash(&g_tgt, &hash_tgt) == AXMAN_SUCCESS);
    assert(hash_tgt != base_hash);
    assert(hash_tgt != hash_src);

    /* 3. Modify one de Bruijn port */
    int32_t ports2[] = {0, 1, 0, 2, 1}; /* port permutation */
    axman_hypergraph_t g_port = base_graph;
    g_port.edge_ports = ports2;
    uint64_t hash_port = 0;
    assert(axman_hypergraph_hash(&g_port, &hash_port) == AXMAN_SUCCESS);
    assert(hash_port != base_hash);
    assert(hash_port != hash_src);
    assert(hash_port != hash_tgt);

    printf("  Base: 0x%016llX | AltSrc: 0x%016llX | AltTgt: 0x%016llX | AltPort: 0x%016llX\n",
           (unsigned long long)base_hash, (unsigned long long)hash_src,
           (unsigned long long)hash_tgt, (unsigned long long)hash_port);
    printf("  -> PASS\n");
}

static void test_empty_and_isolated_graph_hashes(void) {
    printf("Running test_empty_and_isolated_graph_hashes...\n");

    /* Empty graph (0 nodes, 0 edges) */
    axman_hypergraph_t empty_graph;
    memset(&empty_graph, 0, sizeof(empty_graph));
    uint64_t h_empty = 0;
    assert(axman_hypergraph_hash(&empty_graph, &h_empty) == AXMAN_SUCCESS);
    assert(h_empty != 0);

    /* Isolated nodes graph (10 nodes, 0 edges) */
    axman_hypergraph_t iso_graph;
    memset(&iso_graph, 0, sizeof(iso_graph));
    iso_graph.num_nodes = 10;
    uint64_t h_iso = 0;
    assert(axman_hypergraph_hash(&iso_graph, &h_iso) == AXMAN_SUCCESS);
    assert(h_iso != 0);
    assert(h_iso != h_empty);

    printf("  Empty: 0x%016llX | 10-isolated: 0x%016llX\n",
           (unsigned long long)h_empty, (unsigned long long)h_iso);
    printf("  -> PASS\n");
}

static void test_invalid_graph_hash_rejection(void) {
    printf("Running test_invalid_graph_hash_rejection...\n");

    /* Null checks */
    uint64_t h = 0;
    assert(axman_hypergraph_hash(NULL, &h) == AXMAN_ERR_NULL_POINTER);

    axman_hypergraph_t g;
    memset(&g, 0, sizeof(g));
    assert(axman_hypergraph_hash(&g, NULL) == AXMAN_ERR_NULL_POINTER);

    /* Out of bounds target node */
    size_t offsets[] = {0, 1};
    int64_t targets[] = {99}; /* 99 >= num_nodes (5) */
    int64_t sources[] = {0};
    int32_t ports[]   = {0};

    g.num_nodes = 5;
    g.num_hyperedges = 1;
    g.edge_offsets = offsets;
    g.edge_targets = targets;
    g.edge_sources = sources;
    g.edge_ports = ports;

    assert(axman_hypergraph_hash(&g, &h) == AXMAN_ERR_INDEX_OUT_OF_BOUNDS);

    /* Arity overflow: 18 > 16 */
    size_t big_offsets[] = {0, 18};
    int64_t big_sources[18] = {0};
    int32_t big_ports[18] = {0};
    int64_t big_targets[] = {1};

    g.num_nodes = 5;
    g.num_hyperedges = 1;
    g.edge_offsets = big_offsets;
    g.edge_targets = big_targets;
    g.edge_sources = big_sources;
    g.edge_ports = big_ports;

    assert(axman_hypergraph_hash(&g, &h) == AXMAN_ERR_ARITY_OVERFLOW);

    printf("  -> PASS\n");
}

static void test_streaming_hasher_equivalence(void) {
    printf("Running test_streaming_hasher_equivalence...\n");

    const char* text = "Curry-Howard-Lambek Topological Hypergraph Attention";
    size_t len = strlen(text);

    uint64_t one_pass = axman_hash64(text, len, 42);

    axman_hasher_t hasher;
    axman_hasher_init(&hasher, 42);
    /* Ingest in split parts */
    axman_hasher_update(&hasher, text, 10);
    axman_hasher_update(&hasher, text + 10, 20);
    axman_hasher_update(&hasher, text + 30, len - 30);
    uint64_t streamed = axman_hasher_digest(&hasher);

    assert(one_pass == streamed);
    printf("  Streamed matches one-pass: 0x%016llX\n", (unsigned long long)streamed);
    printf("  -> PASS\n");
}

int main(void) {
    printf("=== Running Port-Hypergraph XXH64 Hashing & Validation Test Suite ===\n");
    test_hash_determinism();
    test_hash_sensitivity_and_avalanche();
    test_empty_and_isolated_graph_hashes();
    test_invalid_graph_hash_rejection();
    test_streaming_hasher_equivalence();
    printf("=== ALL STRUCTURAL HASHING TESTS PASSED SUCCESSFULLY! ===\n");
    return 0;
}
