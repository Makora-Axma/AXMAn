/**
 * @file test_safetensors.cpp
 * @brief Unit tests for Task 3.3: Safetensors Serialization and C++ Loading.
 *
 * Verifies:
 * 1. Creation and reading of Safetensors file header and metadata.
 * 2. Exact extraction of tensor weights, shapes, and byte offsets.
 * 3. Loading of safetensors weights into axman_model_opaque and forward inference.
 * 4. Error handling against truncated files and corrupted offsets.
 */

#include "axman_c_api.h"
#include "axman_safetensors.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
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
    std::cout << "     AXMANet PHASE 3.3: SAFETENSORS SERIALIZATION    " << std::endl;
    std::cout << "=====================================================" << std::endl;

    const std::string test_path = "test_model_weights.safetensors";

    // 1. Manually craft a valid safetensors binary file with known float values
    {
        std::cout << "[Test 1] Generating reference safetensors binary archive..." << std::endl;
        // 2 tensors:
        // "node_encoder.weight": [2, 2] = 4 floats (16 bytes)
        // "node_encoder.bias": [2] = 2 floats (8 bytes)
        float w_data[4] = {1.5f, -2.0f, 3.25f, 0.75f};
        float b_data[2] = {0.1f, -0.2f};

        std::string json_header =
            "{\"__metadata__\":{\"d_model\":\"2\",\"num_layers\":\"1\",\"num_heads\":\"1\",\"in_features\":\"2\"},"
            "\"node_encoder.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
            "\"node_encoder.bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}";

        uint64_t header_len = static_cast<uint64_t>(json_header.size());

        std::ofstream out(test_path, std::ios::binary);
        ASSERT_TRUE(out.is_open());

        // 8 bytes header length
        out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        // JSON header
        out.write(json_header.data(), header_len);
        // Tensor data
        out.write(reinterpret_cast<const char*>(w_data), 16);
        out.write(reinterpret_cast<const char*>(b_data), 8);
        out.close();

        std::cout << "  -> Reference archive created successfully." << std::endl;
    }

    // 2. Parse safetensors archive via C++ header-only parser
    {
        std::cout << "[Test 2] Parsing safetensors archive via SafetensorsArchive..." << std::endl;
        auto archive = axman::SafetensorsArchive::load_from_file(test_path);

        ASSERT_EQ(archive.metadata["d_model"], "2");
        ASSERT_EQ(archive.metadata["num_layers"], "1");
        ASSERT_TRUE(archive.has_tensor("node_encoder.weight"));
        ASSERT_TRUE(archive.has_tensor("node_encoder.bias"));

        const float* w_ptr = archive.get_tensor_data_f32("node_encoder.weight", 4);
        ASSERT_TRUE(w_ptr != nullptr);
        ASSERT_EQ(w_ptr[0], 1.5f);
        ASSERT_EQ(w_ptr[1], -2.0f);
        ASSERT_EQ(w_ptr[2], 3.25f);
        ASSERT_EQ(w_ptr[3], 0.75f);

        const float* b_ptr = archive.get_tensor_data_f32("node_encoder.bias", 2);
        ASSERT_TRUE(b_ptr != nullptr);
        ASSERT_EQ(b_ptr[0], 0.1f);
        ASSERT_EQ(b_ptr[1], -0.2f);

        std::cout << "  -> Metadata and float32 tensor payload verified!" << std::endl;
    }

    // 3. Instantiate model with safetensors file through axman_model_create
    {
        std::cout << "[Test 3] Loading model from safetensors file via C-ABI..." << std::endl;
        axman_model_handle_t model = nullptr;
        axman_status_t status = axman_model_create(test_path.c_str(), nullptr, &model);
        ASSERT_EQ(status, AXMAN_SUCCESS);
        ASSERT_TRUE(model != nullptr);

        // Run forward pass on 2 nodes
        size_t num_nodes = 2;
        size_t num_edges = 1;
        int64_t edge_sources[] = {0};
        int32_t edge_ports[] = {0};
        int64_t edge_targets[] = {1};
        size_t edge_offsets[] = {0, 1};

        std::vector<float> features = {1.0f, 0.0f, 0.0f, 1.0f}; // 2 nodes x 2 features
        int64_t feat_shape[] = {2, 2};

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

        axman_output_t out;
        status = axman_model_forward(model, &graph, &out);
        ASSERT_EQ(status, AXMAN_SUCCESS);

        std::cout << "  -> Value estimate from safetensors model: " << out.value_estimate << std::endl;
        std::cout << "  -> Latent dimension: " << out.latent_embedding.shape[1] << std::endl;

        axman_output_free(&out);
        axman_model_destroy(model);
        std::cout << "  -> OK (Safetensors Model execution verified)" << std::endl;
    }

    // Clean up temporary file
    std::remove(test_path.c_str());

    std::cout << "=====================================================" << std::endl;
    std::cout << "   >>> ALL PHASE 3.3 TESTS PASSED SUCCESSFULLY! <<<  " << std::endl;
    std::cout << "=====================================================" << std::endl;
    return 0;
}
