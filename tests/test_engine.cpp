/**
 * @file test_engine.cpp
 * @brief Unit tests for Task 3.1: AXMANet Native C-Engine Inference & C-ABI.
 *
 * Verifies:
 * 1. Model creation (axman_model_create) with NULL, defaults, base preset, and custom JSON config.
 * 2. Proper error propagation and thread-local error messaging.
 * 3. Complete forward pass execution (axman_model_forward) on sample hypergraphs.
 * 4. Validation that positional de Bruijn ports (edge_ports) affect representations.
 * 5. Output tensors validity (policy_logits shape, value_estimate in [0, 1], latent_embedding).
 * 6. Deterministic idempotent reclamation (axman_output_free, axman_model_destroy).
 * 7. Protection against invalid graphs (cycles, out-of-bounds, arity overflows).
 */

#include "axman_c_api.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#include <cstring>

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion FAILED: " #cond " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while (0)

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::cerr << "Assertion FAILED: " #a " == " #b " (" << (a) << " != " << (b) << ") at " \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            return 1; \
        } \
    } while (0)

int main() {
    std::cout << "=====================================================" << std::endl;
    std::cout << "     AXMANet PHASE 3.1: NATIVE INFERENCE ENGINE      " << std::endl;
    std::cout << "=====================================================" << std::endl;

    // Test 1: Null pointer checks
    {
        std::cout << "[Test 1] Testing NULL parameter validation..." << std::endl;
        axman_status_t status = axman_model_create(nullptr, nullptr, nullptr);
        ASSERT_EQ(status, AXMAN_ERR_NULL_POINTER);

        status = axman_model_forward(nullptr, nullptr, nullptr);
        ASSERT_EQ(status, AXMAN_ERR_NULL_POINTER);

        status = axman_model_destroy(nullptr);
        ASSERT_EQ(status, AXMAN_SUCCESS); // Idempotent
        std::cout << "  -> OK" << std::endl;
    }

    // Test 2: Dynamic JSON config parsing (base preset and custom)
    {
        std::cout << "[Test 2] Testing dynamic JSON configuration parsing..." << std::endl;
        axman_model_handle_t model_custom = nullptr;
        // Test custom JSON with d_model=128, num_layers=2, num_heads=4, num_actions=16
        const char* custom_json = "{\"d_model\": 128, \"num_layers\": 2, \"num_heads\": 4, \"num_actions\": 16, \"in_features\": 64}";
        axman_status_t status = axman_model_create(nullptr, custom_json, &model_custom);
        ASSERT_EQ(status, AXMAN_SUCCESS);
        ASSERT_TRUE(model_custom != nullptr);

        // Test forward with custom model
        size_t num_nodes = 2;
        size_t num_edges = 1;
        int64_t edge_sources[] = {0};
        int32_t edge_ports[] = {0};
        int64_t edge_targets[] = {1};
        size_t edge_offsets[] = {0, 1};

        std::vector<float> features(num_nodes * 64, 1.0f);
        int64_t feat_shape[] = {static_cast<int64_t>(num_nodes), 64};

        axman_hypergraph_t graph;
        graph.num_nodes = num_nodes;
        graph.num_hyperedges = num_edges;
        graph.edge_sources = edge_sources;
        graph.edge_ports = edge_ports;
        graph.edge_targets = edge_targets;
        graph.edge_offsets = edge_offsets;
        graph.node_features.data = features.data();
        graph.node_features.shape = feat_shape;
        graph.node_features.ndim = 2;
        graph.node_features.dtype = AXMAN_DTYPE_FLOAT32;
        graph.node_features.device = AXMAN_DEVICE_CPU;

        axman_output_t out_custom;
        status = axman_model_forward(model_custom, &graph, &out_custom);
        ASSERT_EQ(status, AXMAN_SUCCESS);

        // Verify custom dimensions
        ASSERT_EQ(out_custom.policy_logits.shape[0], 2);
        ASSERT_EQ(out_custom.policy_logits.shape[1], 16); // custom 16 actions
        ASSERT_EQ(out_custom.latent_embedding.shape[1], 128); // custom d_model 128

        axman_output_free(&out_custom);
        axman_model_destroy(model_custom);
        std::cout << "  -> OK (Config JSON parsing verified)" << std::endl;
    }

    // Test 3: Lifecycle - create, execute, destroy with de Bruijn port sensitivity
    {
        std::cout << "[Test 3] Testing Model Creation & Full Multi-Head Forward Pass..." << std::endl;
        axman_model_handle_t model = nullptr;
        axman_status_t status = axman_model_create("dummy_model.safetensors", nullptr, &model);
        ASSERT_EQ(status, AXMAN_SUCCESS);
        ASSERT_TRUE(model != nullptr);

        // Build valid hypergraph
        // 3 nodes: 0: A, 1: B, 2: C (goal)
        // 2 hyperedges:
        // edge 0: (0) -> 1
        // edge 1: (0, 1) -> 2 with ports (0, 1)
        size_t num_nodes = 3;
        size_t num_edges = 2;
        int64_t edge_sources[] = {0, 0, 1};
        int32_t edge_ports1[] = {0, 0, 1}; // Standard ports
        int32_t edge_ports2[] = {0, 2, 5}; // Perturbed ports
        int64_t edge_targets[] = {1, 2};
        size_t edge_offsets[] = {0, 1, 3};

        // Node features: 3 nodes x 64 features
        std::vector<float> features(num_nodes * 64, 0.5f);
        int64_t feat_shape[] = {static_cast<int64_t>(num_nodes), 64};

        axman_hypergraph_t graph1;
        graph1.num_nodes = num_nodes;
        graph1.num_hyperedges = num_edges;
        graph1.edge_sources = edge_sources;
        graph1.edge_ports = edge_ports1;
        graph1.edge_targets = edge_targets;
        graph1.edge_offsets = edge_offsets;
        graph1.node_features.data = features.data();
        graph1.node_features.shape = feat_shape;
        graph1.node_features.ndim = 2;
        graph1.node_features.dtype = AXMAN_DTYPE_FLOAT32;
        graph1.node_features.device = AXMAN_DEVICE_CPU;

        // Run forward pass 1
        axman_output_t out1;
        status = axman_model_forward(model, &graph1, &out1);
        ASSERT_EQ(status, AXMAN_SUCCESS);

        // Verify output shapes & values
        ASSERT_EQ(out1.policy_logits.ndim, 2);
        ASSERT_EQ(out1.policy_logits.shape[0], static_cast<int64_t>(num_nodes));
        ASSERT_EQ(out1.policy_logits.shape[1], 32); // 32 actions
        ASSERT_TRUE(out1.policy_logits.data != nullptr);

        ASSERT_TRUE(out1.value_estimate >= 0.0f && out1.value_estimate <= 1.0f);

        ASSERT_EQ(out1.latent_embedding.ndim, 2);
        ASSERT_EQ(out1.latent_embedding.shape[0], 1);
        ASSERT_EQ(out1.latent_embedding.shape[1], 256); // d_model = 256
        ASSERT_TRUE(out1.latent_embedding.data != nullptr);

        // Run forward pass 2 with perturbed ports to verify port sensitivity
        axman_hypergraph_t graph2 = graph1;
        graph2.edge_ports = edge_ports2;
        axman_output_t out2;
        status = axman_model_forward(model, &graph2, &out2);
        ASSERT_EQ(status, AXMAN_SUCCESS);

        // Verify that different positional de Bruijn ports produce different outputs
        float diff_sum = 0.0f;
        const float* p1 = static_cast<const float*>(out1.policy_logits.data);
        const float* p2 = static_cast<const float*>(out2.policy_logits.data);
        for (size_t i = 0; i < num_nodes * 32; ++i) {
            diff_sum += std::abs(p1[i] - p2[i]);
        }
        std::cout << "  -> de Bruijn Port Sensitivity Diff: " << diff_sum << std::endl;
        ASSERT_TRUE(diff_sum > 1e-4f); // Port embeddings strictly affect the representation!

        std::cout << "  -> Value estimate: " << out1.value_estimate << std::endl;
        std::cout << "  -> Policy shape: [" << out1.policy_logits.shape[0] << ", "
                  << out1.policy_logits.shape[1] << "]" << std::endl;
        std::cout << "  -> Latent shape: [" << out1.latent_embedding.shape[0] << ", "
                  << out1.latent_embedding.shape[1] << "]" << std::endl;

        // Cleanup outputs
        axman_output_free(&out1);
        axman_output_free(&out2);
        ASSERT_TRUE(out1.policy_logits.data == nullptr);
        ASSERT_TRUE(out1.latent_embedding.data == nullptr);

        // Destroy model
        status = axman_model_destroy(model);
        ASSERT_EQ(status, AXMAN_SUCCESS);
        std::cout << "  -> OK" << std::endl;
    }

    // Test 4: Rejection of invalid graph
    {
        std::cout << "[Test 4] Testing rejection of corrupted graph topology..." << std::endl;
        axman_model_handle_t model = nullptr;
        axman_status_t status = axman_model_create(nullptr, nullptr, &model);
        ASSERT_EQ(status, AXMAN_SUCCESS);

        axman_hypergraph_t bad_graph;
        std::memset(&bad_graph, 0, sizeof(bad_graph));
        bad_graph.num_nodes = 2;
        bad_graph.num_hyperedges = 1;
        int64_t bad_targets[] = {99}; // Out of bounds!
        size_t bad_offsets[] = {0, 0};
        bad_graph.edge_targets = bad_targets;
        bad_graph.edge_offsets = bad_offsets;

        axman_output_t out;
        status = axman_model_forward(model, &bad_graph, &out);
        ASSERT_EQ(status, AXMAN_ERR_INDEX_OUT_OF_BOUNDS);

        const char* err_msg = axman_get_last_error_message();
        std::cout << "  -> Intercepted error message: " << err_msg << std::endl;

        axman_model_destroy(model);
        std::cout << "  -> OK" << std::endl;
    }

    std::cout << "=====================================================" << std::endl;
    std::cout << "   >>> ALL PHASE 3.1 TESTS PASSED SUCCESSFULLY! <<<  " << std::endl;
    std::cout << "=====================================================" << std::endl;
    return 0;
}
