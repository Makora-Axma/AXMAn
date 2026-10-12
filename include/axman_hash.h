/**
 * @file axman_hash.h
 * @brief Deterministic 64-bit Structural Topology Hashing for AXMANet
 *
 * Implements high-performance XXH64-based non-cryptographic structural hashing
 * for port-hypergraphs to enable graph deduplication, LRU inference caching,
 * and de Bruijn topology equivalence matching.
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef AXMAN_HASH_H
#define AXMAN_HASH_H

#include "axman_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Streaming 64-bit hasher state (XXH64-compatible).
 */
typedef struct {
    uint64_t total_len;
    uint64_t v[4];
    uint64_t mem64[4];
    uint32_t memsize;
} axman_hasher_t;

/**
 * @brief Initialize a streaming 64-bit hasher with a seed.
 */
AXMAN_API void axman_hasher_init(
    axman_hasher_t* hasher,
    uint64_t seed
);

/**
 * @brief Ingest a chunk of memory into the streaming hasher.
 */
AXMAN_API void axman_hasher_update(
    axman_hasher_t* hasher,
    const void* input,
    size_t length
);

/**
 * @brief Finalize and produce the 64-bit hash digest.
 */
AXMAN_API uint64_t axman_hasher_digest(
    const axman_hasher_t* hasher
);

/**
 * @brief Compute a 64-bit hash of a contiguous memory block in a single pass.
 */
AXMAN_API uint64_t axman_hash64(
    const void* data,
    size_t length,
    uint64_t seed
);

#ifdef __cplusplus
}
#endif

#endif /* AXMAN_HASH_H */
