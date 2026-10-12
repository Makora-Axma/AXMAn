/**
 * @file hash.c
 * @brief Deterministic 64-bit Structural Topology Hashing & Validation for AXMANet
 *
 * Implements canonical XXH64 hashing over Port-Hypergraph CSR structures
 * to support graph deduplication, LRU caching, and invariant checking.
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#include "axman_hash.h"
#include <string.h>

/* -------------------------------------------------------------------------- */
/* XXH64 Constants & Primitives                                              */
/* -------------------------------------------------------------------------- */

static const uint64_t PRIME64_1 = 11400714785074694791ULL;
static const uint64_t PRIME64_2 = 14029467366897019727ULL;
static const uint64_t PRIME64_3 =  1609587929392839161ULL;
static const uint64_t PRIME64_4 =  9650029242287828579ULL;
static const uint64_t PRIME64_5 =  2870177450012600261ULL;

static inline uint64_t rotl64(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

static inline uint64_t xxh64_round(uint64_t acc, uint64_t input) {
    acc += input * PRIME64_2;
    acc = rotl64(acc, 31);
    acc *= PRIME64_1;
    return acc;
}

static inline uint64_t xxh64_merge_round(uint64_t acc, uint64_t val) {
    val = xxh64_round(0, val);
    acc ^= val;
    acc = acc * PRIME64_1 + PRIME64_4;
    return acc;
}

static inline uint64_t read64_le(const void* ptr) {
    uint64_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

static inline uint32_t read32_le(const void* ptr) {
    uint32_t val;
    memcpy(&val, ptr, sizeof(val));
    return val;
}

/* -------------------------------------------------------------------------- */
/* Streaming Hasher Implementation                                            */
/* -------------------------------------------------------------------------- */

AXMAN_API void axman_hasher_init(axman_hasher_t* hasher, uint64_t seed) {
    if (!hasher) return;
    hasher->total_len = 0;
    hasher->v[0] = seed + PRIME64_1 + PRIME64_2;
    hasher->v[1] = seed + PRIME64_2;
    hasher->v[2] = seed + 0;
    hasher->v[3] = seed - PRIME64_1;
    hasher->memsize = 0;
    memset(hasher->mem64, 0, sizeof(hasher->mem64));
}

AXMAN_API void axman_hasher_update(
    axman_hasher_t* hasher,
    const void* input,
    size_t length
) {
    if (!hasher || !input || length == 0) return;

    const uint8_t* p = (const uint8_t*)input;
    const uint8_t* const bEnd = p + length;
    hasher->total_len += length;

    if (hasher->memsize + length < 32) {
        memcpy((uint8_t*)hasher->mem64 + hasher->memsize, p, length);
        hasher->memsize += (uint32_t)length;
        return;
    }

    if (hasher->memsize > 0) {
        size_t fill = 32 - hasher->memsize;
        memcpy((uint8_t*)hasher->mem64 + hasher->memsize, p, fill);
        hasher->v[0] = xxh64_round(hasher->v[0], read64_le((const uint8_t*)hasher->mem64 + 0));
        hasher->v[1] = xxh64_round(hasher->v[1], read64_le((const uint8_t*)hasher->mem64 + 8));
        hasher->v[2] = xxh64_round(hasher->v[2], read64_le((const uint8_t*)hasher->mem64 + 16));
        hasher->v[3] = xxh64_round(hasher->v[3], read64_le((const uint8_t*)hasher->mem64 + 24));
        p += fill;
        hasher->memsize = 0;
    }

    if (p <= bEnd - 32) {
        const uint8_t* const limit = bEnd - 32;
        uint64_t v1 = hasher->v[0];
        uint64_t v2 = hasher->v[1];
        uint64_t v3 = hasher->v[2];
        uint64_t v4 = hasher->v[3];

        do {
            v1 = xxh64_round(v1, read64_le(p + 0));
            v2 = xxh64_round(v2, read64_le(p + 8));
            v3 = xxh64_round(v3, read64_le(p + 16));
            v4 = xxh64_round(v4, read64_le(p + 24));
            p += 32;
        } while (p <= limit);

        hasher->v[0] = v1;
        hasher->v[1] = v2;
        hasher->v[2] = v3;
        hasher->v[3] = v4;
    }

    if (p < bEnd) {
        memcpy(hasher->mem64, p, (size_t)(bEnd - p));
        hasher->memsize = (uint32_t)(bEnd - p);
    }
}

AXMAN_API uint64_t axman_hasher_digest(const axman_hasher_t* hasher) {
    if (!hasher) return 0;

    uint64_t h64;

    if (hasher->total_len >= 32) {
        uint64_t v1 = hasher->v[0];
        uint64_t v2 = hasher->v[1];
        uint64_t v3 = hasher->v[2];
        uint64_t v4 = hasher->v[3];

        h64 = rotl64(v1, 1) + rotl64(v2, 7) + rotl64(v3, 12) + rotl64(v4, 18);
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = hasher->v[2] + PRIME64_5;
    }

    h64 += hasher->total_len;

    const uint8_t* p = (const uint8_t*)hasher->mem64;
    const uint8_t* const bEnd = p + hasher->memsize;

    while (p + 8 <= bEnd) {
        uint64_t k1 = xxh64_round(0, read64_le(p));
        h64 ^= k1;
        h64 = rotl64(h64, 27) * PRIME64_1 + PRIME64_4;
        p += 8;
    }

    if (p + 4 <= bEnd) {
        h64 ^= (uint64_t)(read32_le(p)) * PRIME64_1;
        h64 = rotl64(h64, 23) * PRIME64_2 + PRIME64_3;
        p += 4;
    }

    while (p < bEnd) {
        h64 ^= (*p) * PRIME64_5;
        h64 = rotl64(h64, 11) * PRIME64_1;
        p++;
    }

    /* Final avalanche */
    h64 ^= h64 >> 33;
    h64 *= PRIME64_2;
    h64 ^= h64 >> 29;
    h64 *= PRIME64_3;
    h64 ^= h64 >> 32;

    return h64;
}

AXMAN_API uint64_t axman_hash64(const void* data, size_t length, uint64_t seed) {
    axman_hasher_t hasher;
    axman_hasher_init(&hasher, seed);
    axman_hasher_update(&hasher, data, length);
    return axman_hasher_digest(&hasher);
}

/* -------------------------------------------------------------------------- */
/* Port-Hypergraph Structural Topology Hashing                                */
/* -------------------------------------------------------------------------- */

AXMAN_API axman_status_t axman_hypergraph_hash(
    const axman_hypergraph_t* graph,
    uint64_t* out_hash
) {
    if (!graph || !out_hash) {
        return AXMAN_ERR_NULL_POINTER;
    }

    /* 1. Strict validation of graph invariants first */
    axman_status_t val_status = axman_hypergraph_validate(graph);
    if (val_status != AXMAN_SUCCESS) {
        return val_status;
    }

    /* 2. Compute canonical 64-bit structural hash */
    axman_hasher_t hasher;
    axman_hasher_init(&hasher, 0x41584D414EULL); /* "AXMAN" seed */

    uint64_t header[3];
    header[0] = (uint64_t)graph->num_nodes;
    header[1] = (uint64_t)graph->num_hyperedges;
    header[2] = (graph->num_hyperedges > 0 && graph->edge_offsets)
                ? (uint64_t)graph->edge_offsets[graph->num_hyperedges]
                : 0ULL;
    axman_hasher_update(&hasher, header, sizeof(header));

    if (graph->num_hyperedges > 0) {
        /* Ingest CSR offsets */
        axman_hasher_update(&hasher,
                            graph->edge_offsets,
                            (graph->num_hyperedges + 1) * sizeof(size_t));

        /* Ingest target nodes */
        axman_hasher_update(&hasher,
                            graph->edge_targets,
                            graph->num_hyperedges * sizeof(int64_t));

        size_t total_sources = graph->edge_offsets[graph->num_hyperedges];
        if (total_sources > 0) {
            /* Ingest source premises */
            axman_hasher_update(&hasher,
                                graph->edge_sources,
                                total_sources * sizeof(int64_t));

            /* Ingest positional port indices (de Bruijn binding order) */
            axman_hasher_update(&hasher,
                                graph->edge_ports,
                                total_sources * sizeof(int32_t));
        }
    }

    /* Ingest feature tensor signature if present */
    if (graph->node_features.data != NULL && graph->node_features.shape != NULL) {
        int64_t feat_meta[4];
        feat_meta[0] = (int64_t)graph->node_features.ndim;
        feat_meta[1] = graph->node_features.shape[0];
        feat_meta[2] = (graph->node_features.ndim > 1) ? graph->node_features.shape[1] : 0;
        feat_meta[3] = (int64_t)graph->node_features.dtype;
        axman_hasher_update(&hasher, feat_meta, sizeof(feat_meta));
    }

    *out_hash = axman_hasher_digest(&hasher);
    return AXMAN_SUCCESS;
}
