/**
 * @file currying.c
 * @brief Currying Decomposition Pass & Port-Hypergraph Topology Validation
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#include "axman_currying.h"
#include <stdlib.h>
#include <string.h>

/* -------------------------------------------------------------------------- */
/* Hypergraph Invariant Validation                                            */
/* -------------------------------------------------------------------------- */

AXMAN_API axman_status_t axman_hypergraph_check_invariants(
    const axman_hypergraph_t* graph,
    size_t max_arity
) {
    if (!graph) {
        return AXMAN_ERR_NULL_POINTER;
    }

    if (max_arity == 0) {
        max_arity = AXMAN_MAX_HYPEREDGE_ARITY;
    }

    /* An empty graph (0 nodes, 0 edges) is considered vacuously valid */
    if (graph->num_nodes == 0 && graph->num_hyperedges == 0) {
        return AXMAN_SUCCESS;
    }

    /* Edges cannot reference nodes if there are no nodes */
    if (graph->num_nodes == 0 && graph->num_hyperedges > 0) {
        return AXMAN_ERR_INDEX_OUT_OF_BOUNDS;
    }

    if (graph->num_hyperedges > 0) {
        if (!graph->edge_offsets || !graph->edge_targets) {
            return AXMAN_ERR_NULL_POINTER;
        }

        if (graph->edge_offsets[0] != 0) {
            return AXMAN_ERR_INVALID_TOPOLOGY;
        }

        size_t total_sources = graph->edge_offsets[graph->num_hyperedges];
        if (total_sources > 0) {
            if (!graph->edge_sources || !graph->edge_ports) {
                return AXMAN_ERR_NULL_POINTER;
            }
        }

        for (size_t i = 0; i < graph->num_hyperedges; ++i) {
            size_t start = graph->edge_offsets[i];
            size_t end = graph->edge_offsets[i + 1];

            /* Monotonicity check */
            if (end < start) {
                return AXMAN_ERR_INVALID_TOPOLOGY;
            }

            /* Arity constraint check */
            size_t arity = end - start;
            if (arity > max_arity) {
                return AXMAN_ERR_ARITY_OVERFLOW;
            }

            /* Target node index bounds */
            int64_t target = graph->edge_targets[i];
            if (target < 0 || (size_t)target >= graph->num_nodes) {
                return AXMAN_ERR_INDEX_OUT_OF_BOUNDS;
            }

            /* Source nodes and ports checks */
            for (size_t s = start; s < end; ++s) {
                int64_t source = graph->edge_sources[s];
                if (source < 0 || (size_t)source >= graph->num_nodes) {
                    return AXMAN_ERR_INDEX_OUT_OF_BOUNDS;
                }

                int32_t port = graph->edge_ports[s];
                if (port < 0) {
                    return AXMAN_ERR_INVALID_ARGUMENT;
                }
            }
        }
    }

    /* Validate node_features if provided */
    if (graph->node_features.data != NULL) {
        if (!graph->node_features.shape) {
            return AXMAN_ERR_INVALID_ARGUMENT;
        }
        if (graph->node_features.ndim != 2) {
            return AXMAN_ERR_INVALID_ARGUMENT;
        }
        if (graph->node_features.shape[0] != (int64_t)graph->num_nodes) {
            return AXMAN_ERR_INVALID_ARGUMENT;
        }
        if (graph->node_features.shape[1] <= 0) {
            return AXMAN_ERR_INVALID_ARGUMENT;
        }
        if (graph->node_features.dtype < 0 || graph->node_features.dtype > 4) {
            return AXMAN_ERR_INVALID_ARGUMENT;
        }
    }

    return AXMAN_SUCCESS;
}

AXMAN_API axman_status_t axman_hypergraph_validate(
    const axman_hypergraph_t* graph
) {
    return axman_hypergraph_check_invariants(graph, AXMAN_MAX_HYPEREDGE_ARITY);
}

/* -------------------------------------------------------------------------- */
/* Currying Decomposition Pass Implementation                                 */
/* -------------------------------------------------------------------------- */

static void calculate_splits_for_edge(
    size_t arity,
    size_t max_arity,
    size_t* out_edges,
    size_t* out_aux,
    size_t* out_sources
) {
    if (arity <= max_arity) {
        *out_edges = 1;
        *out_aux = 0;
        *out_sources = arity;
        return;
    }

    /* Arity > max_arity: decomposition required.
     * We chunk sources. First chunk takes (max_arity - 1) sources -> aux_1.
     * Intermediate chunks take prev_aux + (max_arity - 2) sources -> aux_{next}.
     * Final chunk takes prev_aux + remaining sources -> original_target.
     * All chunks have arity <= max_arity.
     */
    size_t chunk_first = (max_arity > 1) ? (max_arity - 1) : 1;
    size_t chunk_mid = (max_arity > 2) ? (max_arity - 2) : 1;

    size_t edges = 1;
    size_t aux = 1;
    size_t sources = chunk_first;
    size_t rem = arity - chunk_first;

    while (1 + rem > max_arity) {
        edges++;
        aux++;
        sources += (1 + chunk_mid); /* 1 for prev_aux + chunk_mid fresh sources */
        rem -= chunk_mid;
    }

    /* Final edge */
    edges++;
    sources += (1 + rem); /* 1 for prev_aux + rem remaining sources */

    *out_edges = edges;
    *out_aux = aux;
    *out_sources = sources;
}

AXMAN_API axman_status_t axman_hypergraph_curry_decompose(
    const axman_hypergraph_t* input_graph,
    const axman_currying_config_t* config,
    axman_currying_result_t* out_result
) {
    if (!input_graph || !out_result) {
        return AXMAN_ERR_NULL_POINTER;
    }

    memset(out_result, 0, sizeof(*out_result));

    size_t max_arity = AXMAN_MAX_HYPEREDGE_ARITY;
    bool synthesize_features = false;
    if (config) {
        if (config->max_arity > 0) {
            max_arity = config->max_arity;
        }
        synthesize_features = config->synthesize_node_features;
    }

    if (max_arity < 2) {
        return AXMAN_ERR_INVALID_ARGUMENT;
    }

    /* Basic sanity check on input graph without arity constraint */
    if (input_graph->num_hyperedges > 0) {
        if (!input_graph->edge_offsets || !input_graph->edge_targets) {
            return AXMAN_ERR_NULL_POINTER;
        }
        if (input_graph->edge_offsets[0] != 0) {
            return AXMAN_ERR_INVALID_TOPOLOGY;
        }
        for (size_t i = 0; i < input_graph->num_hyperedges; ++i) {
            if (input_graph->edge_offsets[i + 1] < input_graph->edge_offsets[i]) {
                return AXMAN_ERR_INVALID_TOPOLOGY;
            }
            int64_t target = input_graph->edge_targets[i];
            if (target < 0 || (size_t)target >= input_graph->num_nodes) {
                return AXMAN_ERR_INDEX_OUT_OF_BOUNDS;
            }
        }
    }

    /* Check if decomposition is actually required */
    bool needs_decomposition = false;
    size_t total_decomposed_edges = 0;
    size_t total_aux_nodes = 0;
    size_t total_decomposed_sources = 0;

    for (size_t i = 0; i < input_graph->num_hyperedges; ++i) {
        size_t arity = input_graph->edge_offsets[i + 1] - input_graph->edge_offsets[i];
        if (arity > max_arity) {
            needs_decomposition = true;
        }
        size_t e_count = 0, a_count = 0, s_count = 0;
        calculate_splits_for_edge(arity, max_arity, &e_count, &a_count, &s_count);
        total_decomposed_edges += e_count;
        total_aux_nodes += a_count;
        total_decomposed_sources += s_count;
    }

    if (!needs_decomposition) {
        /* No decomposition needed, return borrowed view */
        out_result->graph = *input_graph;
        out_result->was_decomposed = false;
        out_result->num_auxiliary_nodes = 0;
        out_result->num_added_hyperedges = 0;
        return AXMAN_SUCCESS;
    }

    /* Allocate normalized topological CSR buffers */
    size_t* new_offsets = (size_t*)malloc((total_decomposed_edges + 1) * sizeof(size_t));
    int64_t* new_targets = (int64_t*)malloc(total_decomposed_edges * sizeof(int64_t));
    int64_t* new_sources = NULL;
    int32_t* new_ports = NULL;

    if (total_decomposed_sources > 0) {
        new_sources = (int64_t*)malloc(total_decomposed_sources * sizeof(int64_t));
        new_ports = (int32_t*)malloc(total_decomposed_sources * sizeof(int32_t));
    }

    if (!new_offsets || !new_targets || (total_decomposed_sources > 0 && (!new_sources || !new_ports))) {
        free(new_offsets);
        free(new_targets);
        free(new_sources);
        free(new_ports);
        return AXMAN_ERR_OUT_OF_MEMORY;
    }

    size_t cur_edge = 0;
    size_t cur_src = 0;
    size_t next_aux_node = input_graph->num_nodes;
    new_offsets[0] = 0;

    size_t chunk_first = (max_arity > 1) ? (max_arity - 1) : 1;
    size_t chunk_mid = (max_arity > 2) ? (max_arity - 2) : 1;

    for (size_t i = 0; i < input_graph->num_hyperedges; ++i) {
        size_t orig_start = input_graph->edge_offsets[i];
        size_t orig_end = input_graph->edge_offsets[i + 1];
        size_t arity = orig_end - orig_start;
        int64_t final_target = input_graph->edge_targets[i];

        if (arity <= max_arity) {
            /* Copy unchanged */
            for (size_t s = orig_start; s < orig_end; ++s) {
                new_sources[cur_src] = input_graph->edge_sources[s];
                new_ports[cur_src] = input_graph->edge_ports[s];
                cur_src++;
            }
            new_targets[cur_edge] = final_target;
            cur_edge++;
            new_offsets[cur_edge] = cur_src;
        } else {
            /* Decompose into curried chain */
            size_t read_idx = orig_start;

            /* First edge: chunk_first sources -> aux_node */
            int64_t cur_aux = (int64_t)(next_aux_node++);
            for (size_t k = 0; k < chunk_first; ++k) {
                new_sources[cur_src] = input_graph->edge_sources[read_idx];
                new_ports[cur_src] = input_graph->edge_ports[read_idx];
                read_idx++;
                cur_src++;
            }
            new_targets[cur_edge] = cur_aux;
            cur_edge++;
            new_offsets[cur_edge] = cur_src;

            size_t rem = orig_end - read_idx;

            /* Intermediate edges */
            while (1 + rem > max_arity) {
                int64_t next_aux = (int64_t)(next_aux_node++);
                /* Source 0: previous aux */
                new_sources[cur_src] = cur_aux;
                new_ports[cur_src] = 0;
                cur_src++;

                /* Remaining sources in chunk */
                for (size_t k = 0; k < chunk_mid; ++k) {
                    new_sources[cur_src] = input_graph->edge_sources[read_idx];
                    new_ports[cur_src] = (int32_t)(k + 1);
                    read_idx++;
                    cur_src++;
                }

                new_targets[cur_edge] = next_aux;
                cur_edge++;
                new_offsets[cur_edge] = cur_src;

                cur_aux = next_aux;
                rem = orig_end - read_idx;
            }

            /* Final edge of chain */
            new_sources[cur_src] = cur_aux;
            new_ports[cur_src] = 0;
            cur_src++;

            int32_t p = 1;
            while (read_idx < orig_end) {
                new_sources[cur_src] = input_graph->edge_sources[read_idx];
                new_ports[cur_src] = p++;
                read_idx++;
                cur_src++;
            }

            new_targets[cur_edge] = final_target;
            cur_edge++;
            new_offsets[cur_edge] = cur_src;
        }
    }

    out_result->graph.num_nodes = input_graph->num_nodes + total_aux_nodes;
    out_result->graph.num_hyperedges = total_decomposed_edges;
    out_result->graph.edge_offsets = new_offsets;
    out_result->graph.edge_sources = new_sources;
    out_result->graph.edge_ports = new_ports;
    out_result->graph.edge_targets = new_targets;

    /* Handle node_features if synthesizing is requested */
    if (synthesize_features && input_graph->node_features.data && input_graph->node_features.ndim == 2) {
        int64_t feat_dim = input_graph->node_features.shape[1];
        size_t old_nodes = input_graph->num_nodes;
        size_t new_nodes = out_result->graph.num_nodes;

        int64_t* new_shape = (int64_t*)malloc(2 * sizeof(int64_t));
        if (new_shape) {
            new_shape[0] = (int64_t)new_nodes;
            new_shape[1] = feat_dim;

            size_t elem_size = sizeof(float);
            if (input_graph->node_features.dtype == AXMAN_DTYPE_FLOAT32) {
                elem_size = sizeof(float);
            } else if (input_graph->node_features.dtype == AXMAN_DTYPE_INT64) {
                elem_size = sizeof(int64_t);
            }

            void* new_data = calloc(new_nodes * feat_dim, elem_size);
            if (new_data) {
                memcpy(new_data, input_graph->node_features.data, old_nodes * feat_dim * elem_size);
                out_result->graph.node_features.data = new_data;
                out_result->graph.node_features.shape = new_shape;
                out_result->graph.node_features.ndim = 2;
                out_result->graph.node_features.dtype = input_graph->node_features.dtype;
                out_result->graph.node_features.device = input_graph->node_features.device;
            } else {
                free(new_shape);
                out_result->graph.node_features = input_graph->node_features;
            }
        } else {
            out_result->graph.node_features = input_graph->node_features;
        }
    } else {
        out_result->graph.node_features = input_graph->node_features;
    }

    out_result->was_decomposed = true;
    out_result->num_auxiliary_nodes = total_aux_nodes;
    out_result->num_added_hyperedges = total_decomposed_edges - input_graph->num_hyperedges;

    /* Verify invariants of the decomposed graph */
    axman_status_t status = axman_hypergraph_check_invariants(&out_result->graph, max_arity);
    if (status != AXMAN_SUCCESS) {
        axman_currying_result_free(out_result);
        return status;
    }

    return AXMAN_SUCCESS;
}

AXMAN_API void axman_currying_result_free(
    axman_currying_result_t* result
) {
    if (!result) return;

    if (result->was_decomposed) {
        free((void*)result->graph.edge_offsets);
        free((void*)result->graph.edge_targets);
        free((void*)result->graph.edge_sources);
        free((void*)result->graph.edge_ports);

        /* If synthesized node_features were allocated */
        if (result->graph.node_features.shape && result->graph.node_features.data) {
            /* Only free if it was a newly allocated buffer distinct from original */
            if (result->graph.num_nodes > 0 && result->num_auxiliary_nodes > 0) {
                free(result->graph.node_features.shape);
                free(result->graph.node_features.data);
            }
        }
    }

    memset(result, 0, sizeof(*result));
}
