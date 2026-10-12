/**
 * @file axman_engine_internal.hpp
 * @brief Internal C++ definitions for the AXMANet native inference engine.
 *
 * Implements struct axman_model_opaque, full configuration parser,
 * exact weight structures matching Phase 1 & Phase 2 specs, and
 * mathematically accurate, high-performance C++ kernels.
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef AXMAN_ENGINE_INTERNAL_HPP
#define AXMAN_ENGINE_INTERNAL_HPP

#include "axman_c_api.h"
#include <string>
#include <vector>
#include <memory>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace axman {

/**
 * @brief Runtime engine configuration parsed from JSON string or presets.
 */
struct EngineConfig {
    int64_t d_model = 256;
    int64_t num_layers = 6;
    int64_t num_heads = 8;
    int64_t port_dim = 64;
    int64_t num_kinds = 32;
    int64_t in_features = 64;
    int64_t num_actions = 32;
    int64_t max_ports = 64;
    std::string port_mode = "hybrid";
    float eps = 1e-6f;
    std::string device_str = "cpu";
    std::string precision_str = "fp32";
    axman_precision_t precision = AXMAN_PRECISION_FP32;

    static EngineConfig from_json(const char* json_str);
};

/**
 * @brief Thread-safe exception formatting and error message storage.
 */
void set_thread_last_error(const std::string& message);
const char* get_thread_last_error();

/**
 * @brief Dense linear layer weights & biases.
 */
struct LinearLayer {
    int64_t in_features = 0;
    int64_t out_features = 0;
    std::vector<float> weight; // [out_features, in_features] row-major
    std::vector<float> bias;   // [out_features]
    bool has_bias = true;

    LinearLayer() = default;
    LinearLayer(int64_t in_f, int64_t out_f, bool bias_flag = true)
        : in_features(in_f), out_features(out_f), has_bias(bias_flag) {
        weight.assign(static_cast<size_t>(in_f * out_f), 0.0f);
        if (has_bias) {
            bias.assign(static_cast<size_t>(out_f), 0.0f);
        }
    }
};

/**
 * @brief RMSNorm layer parameters.
 */
struct RMSNormLayer {
    int64_t dim = 0;
    float eps = 1e-6f;
    std::vector<float> weight; // [dim]

    RMSNormLayer() = default;
    RMSNormLayer(int64_t d, float epsilon = 1e-6f)
        : dim(d), eps(epsilon) {
        weight.assign(static_cast<size_t>(d), 1.0f);
    }
};

/**
 * @brief Port Positional Embedding Layer (DeBruijnPortEmbedding).
 * Hybrid mode: Sinusoidal basis + learned linear projection + learned table.
 */
struct PortEmbeddingLayer {
    int64_t port_dim = 64;
    int64_t max_ports = 64;
    LinearLayer sin_proj;       // [port_dim, port_dim]
    std::vector<float> learned_table; // [max_ports, port_dim]

    PortEmbeddingLayer() = default;
    PortEmbeddingLayer(int64_t p_dim, int64_t m_ports)
        : port_dim(p_dim), max_ports(m_ports), sin_proj(p_dim, p_dim, false) {
        learned_table.assign(static_cast<size_t>(p_dim * m_ports), 0.0f);
    }
};

/**
 * @brief AXMA Morphism Block (AMB) weights (FPHA + CMP + SwiGLU).
 */
struct MorphismBlockWeights {
    // 1. FPHA (Fused Port-Hypergraph Flash Attention)
    RMSNormLayer norm_attn;
    LinearLayer premise_fusion; // [d_model + port_dim] -> d_model
    LinearLayer q_proj;         // [d_model] -> d_model
    LinearLayer k_proj;         // [d_model] -> d_model
    LinearLayer v_proj;         // [d_model] -> d_model
    LinearLayer out_proj;       // [d_model] -> d_model (applied after node scatter)
    LinearLayer edge_proj;      // [d_model] -> d_model (maps aggregated message to hyperedge embedding)

    // 2. CMP (Categorical Morphism Projection)
    RMSNormLayer norm_cmp;
    LinearLayer cmp_proj_a;     // [d_model] -> rank (d_model)
    LinearLayer cmp_proj_b;     // [d_model] -> rank (d_model)
    LinearLayer cmp_proj_out;   // rank -> [d_model]
    LinearLayer cmp_unary_linear; // [d_model] -> [d_model]
    float cmp_gate = 0.0f;      // Learnable gate parameter alpha = sigmoid(cmp_gate)

    // 3. SwiGLU Feed-Forward Network
    RMSNormLayer norm_ffn;
    LinearLayer ffn_gate;       // [d_model] -> intermediate_dim
    LinearLayer ffn_up;         // [d_model] -> intermediate_dim
    LinearLayer ffn_down;       // intermediate_dim -> [d_model]
};

/**
 * @brief Multi-Head Attention Pooling layer weights (for Value Head).
 */
struct AttentionPoolingWeights {
    int64_t d_model = 256;
    int64_t num_heads = 8;
    RMSNormLayer norm;
    std::vector<float> global_query; // [d_model]
    LinearLayer proj_k;              // [d_model] -> d_model
    LinearLayer proj_v;              // [d_model] -> d_model
    LinearLayer proj_out;            // [d_model] -> d_model
};

/**
 * @brief Value Head weights (Attention Pooling + Shared Trunk MLP + Sigmoid/Softplus).
 */
struct ValueHeadWeights {
    AttentionPoolingWeights pooling;
    RMSNormLayer norm;
    LinearLayer trunk_linear1; // [d_model] -> h_dim
    LinearLayer trunk_linear2; // h_dim -> h_dim
    LinearLayer value_linear;  // h_dim -> 1
    LinearLayer dist_linear;   // h_dim -> 1
};

/**
 * @brief Policy Head weights (Tactic Action MLP Classifier + Bilinear Pointing Network).
 */
struct PolicyHeadWeights {
    RMSNormLayer norm_goal;
    RMSNormLayer norm_premise;
    LinearLayer action_mlp1;    // [d_model] -> act_hidden
    LinearLayer action_mlp2;    // act_hidden -> num_actions
    LinearLayer query_proj;     // [d_model] -> d_model
    LinearLayer key_proj;       // [d_model] -> d_model
    std::vector<float> bilinear_weight; // [d_model, d_model] row-major
};

/**
 * @brief Complete model weights container.
 */
struct ModelWeights {
    LinearLayer node_encoder;
    RMSNormLayer input_norm;
    PortEmbeddingLayer port_encoder;
    std::vector<MorphismBlockWeights> blocks;
    RMSNormLayer final_norm;

    ValueHeadWeights value_head;
    PolicyHeadWeights policy_head;
};

/**
 * @brief High-performance math kernel operations on CPU.
 */
class MathKernels {
public:
    static void rms_norm(
        const float* input,
        const float* weight,
        float* output,
        size_t num_vectors,
        size_t dim,
        float eps
    );

    static void linear_forward(
        const float* input,
        const LinearLayer& layer,
        float* output,
        size_t num_vectors
    );

    static void gelu_inplace(float* data, size_t count);

    static void swiglu_forward(
        const float* input,
        const LinearLayer& gate_proj,
        const LinearLayer& up_proj,
        const LinearLayer& down_proj,
        float* output,
        size_t num_vectors,
        int64_t in_dim,
        int64_t intermediate_dim
    );

    static void compute_port_embeddings(
        const int32_t* ports,
        size_t count,
        const PortEmbeddingLayer& layer,
        float* output
    );

    static void fpha_forward(
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
    );

    static void cmp_forward(
        const float* input,
        const MorphismBlockWeights& block,
        float* output,
        size_t num_nodes,
        int64_t d_model
    );

    static void attention_pooling_forward(
        const float* node_embeddings,
        size_t num_nodes,
        const AttentionPoolingWeights& pool_weights,
        float* pooled_graph
    );
};

} // namespace axman

/**
 * @brief Concrete definition of the opaque model struct.
 */
struct axman_model_opaque {
    axman::EngineConfig config;
    std::string model_path;
    axman_device_t device;
    axman::ModelWeights weights;
    bool is_initialized = false;

    axman_model_opaque() : device(AXMAN_DEVICE_CPU) {}
};

#endif // AXMAN_ENGINE_INTERNAL_HPP
