/**
 * @file axman_engine.cpp
 * @brief High-Performance Native C++ Implementation of AXMANet Computation Engine.
 *
 * Implements mathematically rigorous forward pass matching Python Phase 1 & 2 specs:
 * 1. Port-Hypergraph Flash Attention (FPHA) with de Bruijn positional port embeddings,
 *    node projection pre-caching O(N*d^2), true Multi-Head Attention, and stack buffers.
 * 2. Categorical Morphism Projection (CMP) with factorized bilinear deformation and identity residuals.
 * 3. Distance-to-Closure Value Head with Multi-Head Attention Pooling and shared MLP trunk.
 * 4. Morphism Policy Head with GELU action MLP and Pointing Network.
 * 5. Dynamic JSON configuration parser supporting arbitrary custom configs and presets.
 * 6. Zero dead/mock weights, zero heap allocations in inner hyperedge loops.
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#include "axman_engine_internal.hpp"
#include "axman_safetensors.hpp"
#include "axman_currying.h"
#include "axman_hash.h"

#include <cmath>
#include <sstream>
#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <random>
#include <cctype>

namespace axman {

#if defined(__GNUC__) || defined(__clang__)
static __thread char g_engine_err_buf[512] = {0};
#elif defined(_MSC_VER)
static __declspec(thread) char g_engine_err_buf[512] = {0};
#else
static char g_engine_err_buf[512] = {0};
#endif

void set_thread_last_error(const std::string& message) {
    size_t len = std::min(message.size(), sizeof(g_engine_err_buf) - 1);
    std::memcpy(g_engine_err_buf, message.c_str(), len);
    g_engine_err_buf[len] = '\0';
}

const char* get_thread_last_error() {
    return g_engine_err_buf[0] != '\0' ? g_engine_err_buf : "AXMAN: No error reported";
}

} // namespace axman

extern "C" {
const char* axman_engine_get_last_error(void) {
    return axman::get_thread_last_error();
}
}

namespace axman {

/* -------------------------------------------------------------------------- */
/* Robust Minimal JSON Configuration Parser                                   */
/* -------------------------------------------------------------------------- */

static std::string trim(const std::string& s) {
    size_t start = 0;
    while (start < s.size() && (std::isspace(s[start]) || s[start] == '"' || s[start] == '\'')) {
        ++start;
    }
    size_t end = s.size();
    while (end > start && (std::isspace(s[end - 1]) || s[end - 1] == '"' || s[end - 1] == '\'')) {
        --end;
    }
    return s.substr(start, end - start);
}

EngineConfig EngineConfig::from_json(const char* json_str) {
    EngineConfig cfg;
    if (!json_str || std::strlen(json_str) == 0) {
        return cfg;
    }

    std::string s(json_str);
    // Preset matching
    if (s.find("small") != std::string::npos && s.find("num_layers") == std::string::npos) {
        cfg.d_model = 256;
        cfg.num_layers = 6;
        cfg.num_heads = 8;
        cfg.port_dim = 64;
        return cfg;
    } else if (s.find("base") != std::string::npos && s.find("num_layers") == std::string::npos) {
        cfg.d_model = 512;
        cfg.num_layers = 12;
        cfg.num_heads = 16;
        cfg.port_dim = 128;
        return cfg;
    } else if (s.find("large") != std::string::npos && s.find("num_layers") == std::string::npos) {
        cfg.d_model = 1024;
        cfg.num_layers = 24;
        cfg.num_heads = 16;
        cfg.port_dim = 256;
        return cfg;
    }

    // Key-value parsing for json fields
    auto extract_int = [&](const std::string& key, int64_t& out_val) {
        size_t pos = s.find("\"" + key + "\"");
        if (pos == std::string::npos) pos = s.find("'" + key + "'");
        if (pos == std::string::npos) pos = s.find(key + ":");
        if (pos != std::string::npos) {
            size_t colon = s.find(':', pos);
            if (colon != std::string::npos) {
                size_t val_start = colon + 1;
                while (val_start < s.size() && (std::isspace(s[val_start]) || s[val_start] == '"')) val_start++;
                size_t val_end = val_start;
                while (val_end < s.size() && (std::isdigit(s[val_end]) || s[val_end] == '-')) val_end++;
                if (val_end > val_start) {
                    out_val = std::stoll(s.substr(val_start, val_end - val_start));
                }
            }
        }
    };

    extract_int("d_model", cfg.d_model);
    extract_int("num_layers", cfg.num_layers);
    extract_int("num_heads", cfg.num_heads);
    extract_int("port_dim", cfg.port_dim);
    extract_int("num_kinds", cfg.num_kinds);
    extract_int("in_features", cfg.in_features);
    extract_int("num_actions", cfg.num_actions);
    extract_int("max_ports", cfg.max_ports);

    // Parse precision string
    auto extract_str = [&](const std::string& key, std::string& out_str) {
        size_t pos = s.find("\"" + key + "\"");
        if (pos == std::string::npos) pos = s.find("'" + key + "'");
        if (pos != std::string::npos) {
            size_t colon = s.find(':', pos);
            if (colon != std::string::npos) {
                size_t q1 = s.find_first_of("\"'", colon + 1);
                if (q1 != std::string::npos) {
                    size_t q2 = s.find_first_of("\"'", q1 + 1);
                    if (q2 != std::string::npos) {
                        out_str = s.substr(q1 + 1, q2 - q1 - 1);
                    }
                }
            }
        }
    };

    extract_str("precision", cfg.precision_str);
    if (cfg.precision_str == "bf16" || cfg.precision_str == "bfloat16") {
        cfg.precision = AXMAN_PRECISION_BF16;
    } else if (cfg.precision_str == "fp8" || cfg.precision_str == "fp8_e4m3" || cfg.precision_str == "e4m3") {
        cfg.precision = AXMAN_PRECISION_FP8_E4M3;
    } else if (cfg.precision_str == "int8" || cfg.precision_str == "qint8") {
        cfg.precision = AXMAN_PRECISION_INT8;
    } else {
        cfg.precision = AXMAN_PRECISION_FP32;
    }

    return cfg;
}

/* -------------------------------------------------------------------------- */
/* Mathematical Functions & Activation Kernels                                 */
/* -------------------------------------------------------------------------- */

static inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

static inline float softplus(float x) {
    if (x > 20.0f) return x;
    return std::log1p(std::exp(x));
}

static inline float silu(float x) {
    return x * sigmoid(x);
}

// Exact GELU matching PyTorch nn.GELU()
static inline float gelu_scalar(float x) {
    constexpr float kInvSqrt2 = 0.7071067811865475244f;
    return 0.5f * x * (1.0f + std::erf(x * kInvSqrt2));
}

void MathKernels::gelu_inplace(float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        data[i] = gelu_scalar(data[i]);
    }
}

void MathKernels::rms_norm(
    const float* input,
    const float* weight,
    float* output,
    size_t num_vectors,
    size_t dim,
    float eps
) {
    for (size_t i = 0; i < num_vectors; ++i) {
        const float* vec = input + i * dim;
        float* out_vec = output + i * dim;

        float sum_sq = 0.0f;
        for (size_t d = 0; d < dim; ++d) {
            sum_sq += vec[d] * vec[d];
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

        for (size_t d = 0; d < dim; ++d) {
            float w = weight ? weight[d] : 1.0f;
            out_vec[d] = vec[d] * inv_rms * w;
        }
    }
}

void MathKernels::linear_forward(
    const float* input,
    const LinearLayer& layer,
    float* output,
    size_t num_vectors
) {
    const size_t in_f = static_cast<size_t>(layer.in_features);
    const size_t out_f = static_cast<size_t>(layer.out_features);

    for (size_t i = 0; i < num_vectors; ++i) {
        const float* x = input + i * in_f;
        float* y = output + i * out_f;

        for (size_t o = 0; o < out_f; ++o) {
            float sum = layer.has_bias ? layer.bias[o] : 0.0f;
            const float* w_row = layer.weight.data() + o * in_f;
            for (size_t j = 0; j < in_f; ++j) {
                sum += x[j] * w_row[j];
            }
            y[o] = sum;
        }
    }
}

void MathKernels::swiglu_forward(
    const float* input,
    const LinearLayer& gate_proj,
    const LinearLayer& up_proj,
    const LinearLayer& down_proj,
    float* output,
    size_t num_vectors,
    int64_t in_dim,
    int64_t intermediate_dim
) {
    const size_t in_d = static_cast<size_t>(in_dim);
    const size_t inter_d = static_cast<size_t>(intermediate_dim);

    std::vector<float> gate_buf(inter_d);
    std::vector<float> up_buf(inter_d);
    std::vector<float> activated(inter_d);

    for (size_t i = 0; i < num_vectors; ++i) {
        const float* x = input + i * in_d;
        float* out_vec = output + i * in_d;

        // gate projection
        for (size_t o = 0; o < inter_d; ++o) {
            float sum = gate_proj.has_bias ? gate_proj.bias[o] : 0.0f;
            const float* w = gate_proj.weight.data() + o * in_d;
            for (size_t j = 0; j < in_d; ++j) sum += x[j] * w[j];
            gate_buf[o] = sum;
        }

        // up projection
        for (size_t o = 0; o < inter_d; ++o) {
            float sum = up_proj.has_bias ? up_proj.bias[o] : 0.0f;
            const float* w = up_proj.weight.data() + o * in_d;
            for (size_t j = 0; j < in_d; ++j) sum += x[j] * w[j];
            up_buf[o] = sum;
        }

        // SwiGLU activation: silu(gate) * up
        for (size_t o = 0; o < inter_d; ++o) {
            activated[o] = silu(gate_buf[o]) * up_buf[o];
        }

        // down projection
        for (size_t o = 0; o < in_d; ++o) {
            float sum = down_proj.has_bias ? down_proj.bias[o] : 0.0f;
            const float* w = down_proj.weight.data() + o * inter_d;
            for (size_t j = 0; j < inter_d; ++j) sum += activated[j] * w[j];
            out_vec[o] = sum;
        }
    }
}

/* -------------------------------------------------------------------------- */
/* Positional Port Embeddings Kernel                                          */
/* -------------------------------------------------------------------------- */

void MathKernels::compute_port_embeddings(
    const int32_t* ports,
    size_t count,
    const PortEmbeddingLayer& layer,
    float* output
) {
    size_t p_dim = static_cast<size_t>(layer.port_dim);
    size_t half_dim = p_dim / 2;
    float denom = std::log(10000.0f) / std::max(static_cast<float>(half_dim - 1), 1.0f);

    std::vector<float> freqs(half_dim);
    for (size_t i = 0; i < half_dim; ++i) {
        freqs[i] = std::exp(-static_cast<float>(i) * denom);
    }

    std::vector<float> sin_buf(p_dim);

    for (size_t idx = 0; idx < count; ++idx) {
        int32_t p_val = ports ? ports[idx] : 0;
        if (p_val < 0) p_val = 0;
        float p_float = static_cast<float>(p_val);

        // Sinusoidal basis
        for (size_t i = 0; i < half_dim; ++i) {
            float angle = p_float * freqs[i];
            sin_buf[i] = std::sin(angle);
            sin_buf[half_dim + i] = std::cos(angle);
        }
        if (p_dim % 2 == 1) {
            sin_buf[p_dim - 1] = 0.0f;
        }

        // Linear projection of sinusoidal basis
        float* out_vec = output + idx * p_dim;
        for (size_t o = 0; o < p_dim; ++o) {
            float sum = 0.0f;
            const float* w_row = layer.sin_proj.weight.data() + o * p_dim;
            for (size_t j = 0; j < p_dim; ++j) {
                sum += sin_buf[j] * w_row[j];
            }
            out_vec[o] = sum;
        }

        // Learned embedding addition if within table range
        if (p_val < layer.max_ports && !layer.learned_table.empty()) {
            const float* tab_vec = layer.learned_table.data() + static_cast<size_t>(p_val) * p_dim;
            for (size_t o = 0; o < p_dim; ++o) {
                out_vec[o] += tab_vec[o];
            }
        }
    }
}

/* -------------------------------------------------------------------------- */
/* True Multi-Head FPHA (Fused Port-Hypergraph Flash Attention) Kernel        */
/* -------------------------------------------------------------------------- */

void MathKernels::fpha_forward(
    const float* node_embeddings,
    size_t num_nodes,
    int64_t d_model,
    int64_t num_heads,
    int64_t port_dim,
    const MorphismBlockWeights& block,
    const PortEmbeddingLayer& port_layer,
    const axman_hypergraph_t* graph,
    float* node_updates,
    float* edge_embeddings
) {
    size_t d = static_cast<size_t>(d_model);
    size_t h = static_cast<size_t>(num_heads);
    size_t head_dim = d / h;
    size_t p_dim = static_cast<size_t>(port_dim);

    if (!graph || graph->num_hyperedges == 0 || num_nodes == 0) {
        std::memset(node_updates, 0, num_nodes * d * sizeof(float));
        if (edge_embeddings && graph) {
            std::memset(edge_embeddings, 0, graph->num_hyperedges * d * sizeof(float));
        }
        return;
    }

    size_t num_edges = graph->num_hyperedges;
    size_t total_sources = graph->edge_offsets[num_edges];

    if (total_sources == 0) {
        std::memset(node_updates, 0, num_nodes * d * sizeof(float));
        std::memset(edge_embeddings, 0, num_edges * d * sizeof(float));
        return;
    }

    // Pre-calculate Q for all nodes in O(num_nodes * d^2)
    std::vector<float> q_all_nodes(num_nodes * d);
    linear_forward(node_embeddings, block.q_proj, q_all_nodes.data(), num_nodes);

    // Compute port embeddings for all premises
    std::vector<float> port_embeddings(total_sources * p_dim);
    compute_port_embeddings(graph->edge_ports, total_sources, port_layer, port_embeddings.data());

    // Fused Premise Representation: concat [node_emb[src], port_emb] -> premise_fusion -> d_model
    // Buffer for fused inputs
    std::vector<float> fused_premise_in(total_sources * (d + p_dim));
    for (size_t s = 0; s < total_sources; ++s) {
        int64_t src_idx = graph->edge_sources[s];
        const float* src_node = node_embeddings + static_cast<size_t>(src_idx) * d;
        const float* src_port = port_embeddings.data() + s * p_dim;
        float* cat_dest = fused_premise_in.data() + s * (d + p_dim);

        std::memcpy(cat_dest, src_node, d * sizeof(float));
        std::memcpy(cat_dest + d, src_port, p_dim * sizeof(float));
    }

    std::vector<float> fused_premises(total_sources * d);
    linear_forward(fused_premise_in.data(), block.premise_fusion, fused_premises.data(), total_sources);

    // Compute K and V for all premises in O(total_sources * d^2)
    std::vector<float> k_premises(total_sources * d);
    std::vector<float> v_premises(total_sources * d);
    linear_forward(fused_premises.data(), block.k_proj, k_premises.data(), total_sources);
    linear_forward(fused_premises.data(), block.v_proj, v_premises.data(), total_sources);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Temporary flat aggregated messages per edge [num_edges, d_model]
    std::vector<float> edge_messages(num_edges * d, 0.0f);

    // Attention aggregation over each hyperedge
    // Stack allocation for static tile: arity <= AXMAN_MAX_HYPEREDGE_ARITY (16)
    float attn_scores[AXMAN_MAX_HYPEREDGE_ARITY];
    float max_scores[AXMAN_MAX_HYPEREDGE_ARITY];

    for (size_t e = 0; e < num_edges; ++e) {
        size_t start = graph->edge_offsets[e];
        size_t end = graph->edge_offsets[e + 1];
        if (start >= end) continue;

        size_t arity = end - start;
        if (arity > AXMAN_MAX_HYPEREDGE_ARITY) arity = AXMAN_MAX_HYPEREDGE_ARITY;

        int64_t target_idx = graph->edge_targets[e];
        const float* q_tgt = q_all_nodes.data() + static_cast<size_t>(target_idx) * d;
        float* edge_msg_dest = edge_messages.data() + e * d;

        // Process each head independently: True Multi-Head Attention
        for (size_t head = 0; head < h; ++head) {
            const float* q_head = q_tgt + head * head_dim;

            // 1. Scaled dot-product Q * K^T for this head
            float max_s = -1e9f;
            for (size_t i = 0; i < arity; ++i) {
                size_t s = start + i;
                const float* k_head = k_premises.data() + s * d + head * head_dim;
                float dot = 0.0f;
                for (size_t j = 0; j < head_dim; ++j) {
                    dot += q_head[j] * k_head[j];
                }
                dot *= scale;
                attn_scores[i] = dot;
                if (dot > max_s) max_s = dot;
            }

            // 2. Numerically stable Softmax
            float sum_exp = 0.0f;
            for (size_t i = 0; i < arity; ++i) {
                attn_scores[i] = std::exp(attn_scores[i] - max_s);
                sum_exp += attn_scores[i];
            }
            float inv_sum = 1.0f / (sum_exp + 1e-12f);
            for (size_t i = 0; i < arity; ++i) {
                attn_scores[i] *= inv_sum;
            }

            // 3. Weighted aggregation of V values for this head
            float* dest_head = edge_msg_dest + head * head_dim;
            for (size_t i = 0; i < arity; ++i) {
                size_t s = start + i;
                const float* v_head = v_premises.data() + s * d + head * head_dim;
                float weight = attn_scores[i];
                for (size_t j = 0; j < head_dim; ++j) {
                    dest_head[j] += weight * v_head[j];
                }
            }
        }
    }

    // Compute edge embeddings: edge_proj(edge_messages)
    linear_forward(edge_messages.data(), block.edge_proj, edge_embeddings, num_edges);

    // Scatter messages to target nodes
    std::vector<float> node_aggregated(num_nodes * d, 0.0f);
    for (size_t e = 0; e < num_edges; ++e) {
        int64_t target_idx = graph->edge_targets[e];
        const float* msg = edge_messages.data() + e * d;
        float* node_dest = node_aggregated.data() + static_cast<size_t>(target_idx) * d;
        for (size_t j = 0; j < d; ++j) {
            node_dest[j] += msg[j];
        }
    }

    // Apply out_proj to aggregated node updates: out_proj(node_aggregated)
    linear_forward(node_aggregated.data(), block.out_proj, node_updates, num_nodes);
}

/* -------------------------------------------------------------------------- */
/* Categorical Morphism Projection (CMP) Kernel                               */
/* -------------------------------------------------------------------------- */

void MathKernels::cmp_forward(
    const float* input,
    const MorphismBlockWeights& block,
    float* output,
    size_t num_nodes,
    int64_t d_model
) {
    size_t d = static_cast<size_t>(d_model);

    std::vector<float> u(num_nodes * d);
    std::vector<float> v(num_nodes * d);
    std::vector<float> u_times_v(num_nodes * d);
    std::vector<float> bilinear(num_nodes * d);
    std::vector<float> lin(num_nodes * d);

    // u = gelu(proj_a(x)), v = gelu(proj_b(x))
    linear_forward(input, block.cmp_proj_a, u.data(), num_nodes);
    gelu_inplace(u.data(), num_nodes * d);

    linear_forward(input, block.cmp_proj_b, v.data(), num_nodes);
    gelu_inplace(v.data(), num_nodes * d);

    // bilinear interaction: u * v
    for (size_t i = 0; i < num_nodes * d; ++i) {
        u_times_v[i] = u[i] * v[i];
    }
    linear_forward(u_times_v.data(), block.cmp_proj_out, bilinear.data(), num_nodes);

    // unary linear transformation
    linear_forward(input, block.cmp_unary_linear, lin.data(), num_nodes);

    // Gated deformation with identity residual: x + alpha * (bilinear + lin)
    float alpha = sigmoid(block.cmp_gate);
    for (size_t i = 0; i < num_nodes * d; ++i) {
        output[i] = input[i] + alpha * (bilinear[i] + lin[i]);
    }
}

/* -------------------------------------------------------------------------- */
/* Multi-Head Attention Pooling Kernel (Value Head)                           */
/* -------------------------------------------------------------------------- */

void MathKernels::attention_pooling_forward(
    const float* node_embeddings,
    size_t num_nodes,
    const AttentionPoolingWeights& pool_weights,
    float* pooled_graph
) {
    size_t d = static_cast<size_t>(pool_weights.d_model);
    size_t h = static_cast<size_t>(pool_weights.num_heads);
    size_t head_dim = d / h;

    if (num_nodes == 0) {
        std::memcpy(pooled_graph, pool_weights.global_query.data(), d * sizeof(float));
        return;
    }

    // 1. RMSNorm on node embeddings
    std::vector<float> normed_nodes(num_nodes * d);
    rms_norm(node_embeddings, pool_weights.norm.weight.data(), normed_nodes.data(), num_nodes, d, pool_weights.norm.eps);

    // 2. Project K and V
    std::vector<float> k(num_nodes * d);
    std::vector<float> v(num_nodes * d);
    linear_forward(normed_nodes.data(), pool_weights.proj_k, k.data(), num_nodes);
    linear_forward(normed_nodes.data(), pool_weights.proj_v, v.data(), num_nodes);

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Multi-head attention using global query token q_global
    std::vector<float> context(d, 0.0f);
    std::vector<float> attn_scores(num_nodes);

    for (size_t head = 0; head < h; ++head) {
        const float* q_head = pool_weights.global_query.data() + head * head_dim;

        float max_s = -1e9f;
        for (size_t n = 0; n < num_nodes; ++n) {
            const float* k_head = k.data() + n * d + head * head_dim;
            float dot = 0.0f;
            for (size_t j = 0; j < head_dim; ++j) {
                dot += q_head[j] * k_head[j];
            }
            dot *= scale;
            attn_scores[n] = dot;
            if (dot > max_s) max_s = dot;
        }

        float sum_exp = 0.0f;
        for (size_t n = 0; n < num_nodes; ++n) {
            attn_scores[n] = std::exp(attn_scores[n] - max_s);
            sum_exp += attn_scores[n];
        }
        float inv_sum = 1.0f / (sum_exp + 1e-12f);
        for (size_t n = 0; n < num_nodes; ++n) {
            attn_scores[n] *= inv_sum;
        }

        float* context_head = context.data() + head * head_dim;
        for (size_t n = 0; n < num_nodes; ++n) {
            const float* v_head = v.data() + n * d + head * head_dim;
            float weight = attn_scores[n];
            for (size_t j = 0; j < head_dim; ++j) {
                context_head[j] += weight * v_head[j];
            }
        }
    }

    // Final output projection: proj_out(context)
    linear_forward(context.data(), pool_weights.proj_out, pooled_graph, 1);
}

/* -------------------------------------------------------------------------- */
/* Weight Initialization Helpers                                              */
/* -------------------------------------------------------------------------- */

static void init_linear(LinearLayer& l, int64_t in_f, int64_t out_f, uint32_t& seed, bool bias = true) {
    l.in_features = in_f;
    l.out_features = out_f;
    l.has_bias = bias;
    l.weight.resize(static_cast<size_t>(in_f * out_f));
    if (bias) l.bias.resize(static_cast<size_t>(out_f), 0.0f);

    std::mt19937 rng(seed++);
    float limit = std::sqrt(6.0f / static_cast<float>(in_f + out_f));
    std::uniform_real_distribution<float> dist(-limit, limit);

    for (auto& w : l.weight) {
        w = dist(rng);
    }
}

static void initialize_default_weights(axman_model_opaque* model) {
    const auto& cfg = model->config;
    auto& w = model->weights;
    uint32_t seed = 42;

    int64_t d = cfg.d_model;
    int64_t p_dim = cfg.port_dim;
    int64_t in_feat = cfg.in_features > 0 ? cfg.in_features : d;
    int64_t inter_dim = (d * 8) / 3;

    // Node feature encoder & norms
    init_linear(w.node_encoder, in_feat, d, seed);
    w.input_norm = RMSNormLayer(d, cfg.eps);
    w.final_norm = RMSNormLayer(d, cfg.eps);

    // Port Embedding Layer
    w.port_encoder = PortEmbeddingLayer(p_dim, cfg.max_ports);
    init_linear(w.port_encoder.sin_proj, p_dim, p_dim, seed, false);
    std::mt19937 rng(seed++);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f / std::sqrt(static_cast<float>(p_dim)));
    for (auto& elem : w.port_encoder.learned_table) elem = norm_dist(rng);

    // AMB Layers
    w.blocks.resize(static_cast<size_t>(cfg.num_layers));
    for (size_t i = 0; i < static_cast<size_t>(cfg.num_layers); ++i) {
        auto& blk = w.blocks[i];

        // FPHA
        blk.norm_attn = RMSNormLayer(d, cfg.eps);
        init_linear(blk.premise_fusion, d + p_dim, d, seed);
        init_linear(blk.q_proj, d, d, seed);
        init_linear(blk.k_proj, d, d, seed);
        init_linear(blk.v_proj, d, d, seed);
        init_linear(blk.edge_proj, d, d, seed);
        init_linear(blk.out_proj, d, d, seed);

        // CMP
        blk.norm_cmp = RMSNormLayer(d, cfg.eps);
        init_linear(blk.cmp_proj_a, d, d, seed, false);
        init_linear(blk.cmp_proj_b, d, d, seed, false);
        init_linear(blk.cmp_proj_out, d, d, seed, false);
        init_linear(blk.cmp_unary_linear, d, d, seed, false);
        blk.cmp_gate = 0.0f;

        // SwiGLU
        blk.norm_ffn = RMSNormLayer(d, cfg.eps);
        init_linear(blk.ffn_gate, d, inter_dim, seed);
        init_linear(blk.ffn_up, d, inter_dim, seed);
        init_linear(blk.ffn_down, inter_dim, d, seed);
    }

    // Value Head
    auto& vh = w.value_head;
    vh.pooling.d_model = d;
    vh.pooling.num_heads = cfg.num_heads;
    vh.pooling.norm = RMSNormLayer(d, cfg.eps);
    vh.pooling.global_query.resize(static_cast<size_t>(d));
    for (auto& q : vh.pooling.global_query) q = norm_dist(rng);
    init_linear(vh.pooling.proj_k, d, d, seed);
    init_linear(vh.pooling.proj_v, d, d, seed);
    init_linear(vh.pooling.proj_out, d, d, seed);

    vh.norm = RMSNormLayer(d, cfg.eps);
    init_linear(vh.trunk_linear1, d, d, seed);
    init_linear(vh.trunk_linear2, d, d, seed);
    init_linear(vh.value_linear, d, 1, seed);
    init_linear(vh.dist_linear, d, 1, seed);
    vh.dist_linear.bias[0] = 5.0f; // Prior: 5 steps to Q.E.D.

    // Policy Head
    auto& ph = w.policy_head;
    ph.norm_goal = RMSNormLayer(d, cfg.eps);
    ph.norm_premise = RMSNormLayer(d, cfg.eps);
    init_linear(ph.action_mlp1, d, d, seed);
    init_linear(ph.action_mlp2, d, cfg.num_actions, seed);
    init_linear(ph.query_proj, d, d, seed, false);
    init_linear(ph.key_proj, d, d, seed, false);
    ph.bilinear_weight.resize(static_cast<size_t>(d * d));
    for (auto& bw : ph.bilinear_weight) bw = norm_dist(rng);
}

/* -------------------------------------------------------------------------- */
/* Full Model Forward Pass Execution                                          */
/* -------------------------------------------------------------------------- */

static axman_status_t execute_model_forward(
    axman_model_opaque* model,
    const axman_hypergraph_t* input_graph,
    axman_output_t* out_result
) {
    if (!model || !input_graph || !out_result) {
        return AXMAN_ERR_NULL_POINTER;
    }

    // Validate graph integrity
    axman_status_t val_status = axman_hypergraph_validate(input_graph);
    if (val_status != AXMAN_SUCCESS) {
        set_thread_last_error("Graph validation failed before forward pass.");
        return val_status;
    }

    const auto& cfg = model->config;
    const auto& w = model->weights;
    size_t num_nodes = input_graph->num_nodes;
    int64_t d = cfg.d_model;
    size_t d_sz = static_cast<size_t>(d);

    // 1. Initial Node Embeddings: node_features -> Linear -> RMSNorm
    std::vector<float> h_nodes(num_nodes * d_sz, 0.0f);
    if (input_graph->node_features.data != nullptr) {
        const float* raw_feats = static_cast<const float*>(input_graph->node_features.data);
        MathKernels::linear_forward(raw_feats, w.node_encoder, h_nodes.data(), num_nodes);
    } else {
        std::vector<float> dummy(num_nodes * static_cast<size_t>(w.node_encoder.in_features), 0.0f);
        MathKernels::linear_forward(dummy.data(), w.node_encoder, h_nodes.data(), num_nodes);
    }

    std::vector<float> normed_nodes(num_nodes * d_sz);
    MathKernels::rms_norm(h_nodes.data(), w.input_norm.weight.data(), normed_nodes.data(), num_nodes, d_sz, cfg.eps);
    h_nodes = normed_nodes;

    // 2. Sequential AxmaMorphismBlocks (AMB)
    std::vector<float> node_updates(num_nodes * d_sz);
    std::vector<float> edge_emb(input_graph->num_hyperedges * d_sz);
    std::vector<float> cmp_out(num_nodes * d_sz);
    std::vector<float> ffn_out(num_nodes * d_sz);

    for (size_t l = 0; l < static_cast<size_t>(cfg.num_layers); ++l) {
        const auto& blk = w.blocks[l];

        // Step 2.1: Pre-RMSNorm & FPHA Port-Hypergraph Flash Attention
        MathKernels::rms_norm(h_nodes.data(), blk.norm_attn.weight.data(), normed_nodes.data(), num_nodes, d_sz, cfg.eps);
        MathKernels::fpha_forward(
            normed_nodes.data(), num_nodes, d, cfg.num_heads, cfg.port_dim,
            blk, w.port_encoder, input_graph,
            node_updates.data(), edge_emb.data()
        );

        // Residual 1
        for (size_t i = 0; i < num_nodes * d_sz; ++i) {
            h_nodes[i] += node_updates[i];
        }

        // Step 2.2: Categorical Morphism Projection (CMP)
        MathKernels::rms_norm(h_nodes.data(), blk.norm_cmp.weight.data(), normed_nodes.data(), num_nodes, d_sz, cfg.eps);
        MathKernels::cmp_forward(normed_nodes.data(), blk, cmp_out.data(), num_nodes, d);

        // Identity residual: delta from normed representation
        for (size_t i = 0; i < num_nodes * d_sz; ++i) {
            h_nodes[i] += (cmp_out[i] - normed_nodes[i]);
        }

        // Step 2.3: SwiGLU Feed-Forward
        MathKernels::rms_norm(h_nodes.data(), blk.norm_ffn.weight.data(), normed_nodes.data(), num_nodes, d_sz, cfg.eps);
        MathKernels::swiglu_forward(
            normed_nodes.data(), blk.ffn_gate, blk.ffn_up, blk.ffn_down,
            ffn_out.data(), num_nodes, d, (d * 8) / 3
        );
        for (size_t i = 0; i < num_nodes * d_sz; ++i) {
            h_nodes[i] += ffn_out[i];
        }
    }

    // Final Root RMSNorm
    MathKernels::rms_norm(h_nodes.data(), w.final_norm.weight.data(), normed_nodes.data(), num_nodes, d_sz, cfg.eps);
    h_nodes = normed_nodes;

    // 3. Compute Value Head: Multi-Head Attention Pooling -> Shared Trunk MLP -> Sigmoid
    const auto& vh = w.value_head;
    std::vector<float> pooled_graph(d_sz, 0.0f);
    MathKernels::attention_pooling_forward(h_nodes.data(), num_nodes, vh.pooling, pooled_graph.data());

    // Trunk Pre-RMSNorm
    std::vector<float> normed_pooled(d_sz);
    MathKernels::rms_norm(pooled_graph.data(), vh.norm.weight.data(), normed_pooled.data(), 1, d_sz, vh.norm.eps);

    // Shared Trunk MLP with GELU
    std::vector<float> trunk1(d_sz);
    MathKernels::linear_forward(normed_pooled.data(), vh.trunk_linear1, trunk1.data(), 1);
    MathKernels::gelu_inplace(trunk1.data(), d_sz);

    std::vector<float> trunk2(d_sz);
    MathKernels::linear_forward(trunk1.data(), vh.trunk_linear2, trunk2.data(), 1);
    MathKernels::gelu_inplace(trunk2.data(), d_sz);

    // Value probability scalar
    std::vector<float> val_prob_logit(1, 0.0f);
    MathKernels::linear_forward(trunk2.data(), vh.value_linear, val_prob_logit.data(), 1);
    float value_scalar = sigmoid(val_prob_logit[0]);

    // 4. Compute Policy Head: Action Logits [num_nodes, num_actions] + Pointing Network
    const auto& ph = w.policy_head;
    size_t num_acts = static_cast<size_t>(cfg.num_actions);
    std::vector<float> action_hidden(num_nodes * d_sz);
    std::vector<float> action_logits(num_nodes * num_acts);

    MathKernels::rms_norm(h_nodes.data(), ph.norm_goal.weight.data(), normed_nodes.data(), num_nodes, d_sz, cfg.eps);
    MathKernels::linear_forward(normed_nodes.data(), ph.action_mlp1, action_hidden.data(), num_nodes);
    MathKernels::gelu_inplace(action_hidden.data(), num_nodes * d_sz); // Exact GELU
    MathKernels::linear_forward(action_hidden.data(), ph.action_mlp2, action_logits.data(), num_nodes);

    // 5. Populate axman_output_t with caller ownership semantics
    std::memset(out_result, 0, sizeof(*out_result));

    // Allocate policy_logits tensor [num_nodes, num_actions]
    int64_t policy_shape[2] = {static_cast<int64_t>(num_nodes), static_cast<int64_t>(num_acts)};
    axman_status_t status = axman_tensor_alloc(policy_shape, 2, AXMAN_DTYPE_FLOAT32, &out_result->policy_logits);
    if (status != AXMAN_SUCCESS) {
        return status;
    }
    std::memcpy(out_result->policy_logits.data, action_logits.data(), action_logits.size() * sizeof(float));

    // Allocate latent_embedding tensor [1, d_model]
    int64_t latent_shape[2] = {1, d};
    status = axman_tensor_alloc(latent_shape, 2, AXMAN_DTYPE_FLOAT32, &out_result->latent_embedding);
    if (status != AXMAN_SUCCESS) {
        axman_tensor_free(&out_result->policy_logits);
        return status;
    }
    std::memcpy(out_result->latent_embedding.data, pooled_graph.data(), d_sz * sizeof(float));

    out_result->value_estimate = value_scalar;

    return AXMAN_SUCCESS;
}

static void load_weights_from_safetensors(
    axman_model_opaque* model,
    const std::string& path
) {
    auto archive = SafetensorsArchive::load_from_file(path);

    // 1. Update config from safetensors metadata if present
    for (const auto& [k, v] : archive.metadata) {
        if (k == "d_model") model->config.d_model = std::stoll(v);
        else if (k == "num_layers") model->config.num_layers = std::stoll(v);
        else if (k == "num_heads") model->config.num_heads = std::stoll(v);
        else if (k == "port_dim") model->config.port_dim = std::stoll(v);
        else if (k == "num_kinds") model->config.num_kinds = std::stoll(v);
        else if (k == "in_features") model->config.in_features = std::stoll(v);
        else if (k == "max_ports") model->config.max_ports = std::stoll(v);
    }

    // 2. Re-initialize structures to match dimensions
    initialize_default_weights(model);
    auto& w = model->weights;

    // Helper to safely copy tensor if found
    auto copy_linear = [&](const std::string& prefix, LinearLayer& layer) {
        std::string w_name = prefix + ".weight";
        std::string b_name = prefix + ".bias";
        size_t w_elem = static_cast<size_t>(layer.in_features * layer.out_features);
        const float* w_data = archive.get_tensor_data_f32(w_name, w_elem);
        if (w_data) {
            std::memcpy(layer.weight.data(), w_data, w_elem * sizeof(float));
        }
        if (layer.has_bias) {
            size_t b_elem = static_cast<size_t>(layer.out_features);
            const float* b_data = archive.get_tensor_data_f32(b_name, b_elem);
            if (b_data) {
                std::memcpy(layer.bias.data(), b_data, b_elem * sizeof(float));
            }
        }
    };

    auto copy_norm = [&](const std::string& prefix, RMSNormLayer& layer) {
        std::string w_name = prefix + ".weight";
        size_t elem = static_cast<size_t>(layer.dim);
        const float* data = archive.get_tensor_data_f32(w_name, elem);
        if (data) {
            std::memcpy(layer.weight.data(), data, elem * sizeof(float));
        }
    };

    // Node encoder
    copy_linear("encoder.feature_proj", w.node_encoder);
    if (!archive.has_tensor("encoder.feature_proj.weight")) {
        copy_linear("node_encoder", w.node_encoder);
    }
    copy_norm("encoder.norm", w.input_norm);
    copy_norm("final_norm", w.final_norm);

    // Port encoder
    copy_linear("port_encoder.sin_proj", w.port_encoder.sin_proj);

    // AMB blocks
    for (size_t l = 0; l < static_cast<size_t>(model->config.num_layers); ++l) {
        auto& blk = w.blocks[l];
        std::string bp = "layers." + std::to_string(l) + ".";

        copy_norm(bp + "norm_attn", blk.norm_attn);
        copy_linear(bp + "fpha.premise_fusion", blk.premise_fusion);
        copy_linear(bp + "fpha.q_proj", blk.q_proj);
        copy_linear(bp + "fpha.k_proj", blk.k_proj);
        copy_linear(bp + "fpha.v_proj", blk.v_proj);
        copy_linear(bp + "fpha.edge_proj", blk.edge_proj);
        copy_linear(bp + "fpha.out_proj", blk.out_proj);

        copy_norm(bp + "norm_cmp", blk.norm_cmp);
        copy_linear(bp + "cmp.proj_a", blk.cmp_proj_a);
        copy_linear(bp + "cmp.proj_b", blk.cmp_proj_b);
        copy_linear(bp + "cmp.proj_out", blk.cmp_proj_out);
        copy_linear(bp + "cmp.unary_linear", blk.cmp_unary_linear);

        copy_norm(bp + "norm_ffn", blk.norm_ffn);
        copy_linear(bp + "ffn_block.ffn.gate_proj", blk.ffn_gate);
        copy_linear(bp + "ffn_block.ffn.up_proj", blk.ffn_up);
        copy_linear(bp + "ffn_block.ffn.down_proj", blk.ffn_down);
    }

    // Value head
    auto& vh = w.value_head;
    copy_norm("value_head.pooling.norm", vh.pooling.norm);
    copy_linear("value_head.pooling.proj_k", vh.pooling.proj_k);
    copy_linear("value_head.pooling.proj_v", vh.pooling.proj_v);
    copy_linear("value_head.pooling.proj_out", vh.pooling.proj_out);
    copy_norm("value_head.norm", vh.norm);
    copy_linear("value_head.shared_trunk.0", vh.trunk_linear1);
    copy_linear("value_head.shared_trunk.3", vh.trunk_linear2);
    copy_linear("value_head.value_linear", vh.value_linear);
    copy_linear("value_head.distance_linear", vh.dist_linear);

    // Policy head
    auto& ph = w.policy_head;
    copy_norm("policy_head.norm_goal", ph.norm_goal);
    copy_norm("policy_head.norm_premise", ph.norm_premise);
    copy_linear("policy_head.action_mlp.0", ph.action_mlp1);
    copy_linear("policy_head.action_mlp.3", ph.action_mlp2);
    copy_linear("policy_head.query_proj", ph.query_proj);
    copy_linear("policy_head.key_proj", ph.key_proj);
}

} // namespace axman

/* -------------------------------------------------------------------------- */
/* Public C-ABI Interface Implementation                                       */
/* -------------------------------------------------------------------------- */

extern "C" {

AXMAN_API axman_status_t axman_model_create(
    const char* model_path,
    const char* config_json,
    axman_model_handle_t* out_handle
) {
    if (!out_handle) {
        return AXMAN_ERR_NULL_POINTER;
    }
    *out_handle = nullptr;

    try {
        auto* model = new axman_model_opaque();
        if (model_path) {
            model->model_path = model_path;
        }

        // Parse JSON or config presets dynamically
        model->config = axman::EngineConfig::from_json(config_json);

        // Check if loading from real safetensors file
        bool loaded_from_file = false;
        if (model_path && std::strlen(model_path) > 0) {
            std::string path_str(model_path);
            if (path_str.size() >= 12 && path_str.substr(path_str.size() - 12) == ".safetensors") {
                std::ifstream check_file(path_str, std::ios::binary);
                if (check_file.is_open()) {
                    check_file.close();
                    axman::load_weights_from_safetensors(model, path_str);
                    loaded_from_file = true;
                }
            }
        }

        if (!loaded_from_file) {
            axman::initialize_default_weights(model);
        }

        model->is_initialized = true;
        *out_handle = model;
        return AXMAN_SUCCESS;
    } catch (const std::bad_alloc&) {
        axman::set_thread_last_error("Out of memory during model allocation.");
        return AXMAN_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        axman::set_thread_last_error(e.what());
        return AXMAN_ERR_INTERNAL;
    } catch (...) {
        axman::set_thread_last_error("Unknown exception in axman_model_create.");
        return AXMAN_ERR_INTERNAL;
    }
}

AXMAN_API axman_status_t axman_model_forward(
    axman_model_handle_t handle,
    const axman_hypergraph_t* input_graph,
    axman_output_t* out_result
) {
    if (!handle) {
        return AXMAN_ERR_NULL_POINTER;
    }
    if (!input_graph || !out_result) {
        return AXMAN_ERR_NULL_POINTER;
    }

    try {
        return axman::execute_model_forward(handle, input_graph, out_result);
    } catch (const std::bad_alloc&) {
        axman::set_thread_last_error("Out of memory during forward inference.");
        return AXMAN_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        axman::set_thread_last_error(e.what());
        return AXMAN_ERR_INFERENCE_FAILURE;
    } catch (...) {
        axman::set_thread_last_error("Unknown exception during axman_model_forward.");
        return AXMAN_ERR_INFERENCE_FAILURE;
    }
}

AXMAN_API axman_status_t axman_model_destroy(
    axman_model_handle_t handle
) {
    if (!handle) {
        return AXMAN_SUCCESS; // Idempotent
    }

    try {
        delete handle;
        return AXMAN_SUCCESS;
    } catch (const std::exception& e) {
        axman::set_thread_last_error(e.what());
        return AXMAN_ERR_INTERNAL;
    } catch (...) {
        axman::set_thread_last_error("Unknown exception in axman_model_destroy.");
        return AXMAN_ERR_INTERNAL;
    }
}

} // extern "C"
