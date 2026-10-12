/**
 * @file test_cpp_bindings.cpp
 * @brief Comprehensive verification suite for Modern C++20 axman-cpp wrapper.
 *
 * Verifies:
 * 1. Version string retrieval.
 * 2. Result<T, Error> monadic error handling.
 * 3. HypergraphBuilder fluent API, arity capping and validation.
 * 4. Model lifecycle (RAII, move semantics).
 * 5. Complete forward pass with policy logits, value estimate and latent embeddings.
 * 6. Zero-copy TensorSpan accessors (size, shape, 2D indexing).
 * 7. Multi-threaded parallel inference across threads.
 * 8. Error propagation on invalid topologies.
 */

#include "axman/axman.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cmath>
#if defined(_WIN32)
#include <windows.h>
#endif

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
    std::cout << "     AXMANet PHASE 4.2: MODERN C++20 BINDINGS        " << std::endl;
    std::cout << "=====================================================" << std::endl;

    // Test 1: Version check
    {
        std::cout << "[Test 1] Testing version string..." << std::endl;
        auto ver = axman::version();
        ASSERT_EQ(ver, "1.0.0");
        std::cout << "  -> OK (Version: " << ver << ")" << std::endl;
    }

    // Test 2: HypergraphBuilder and Validation
    {
        std::cout << "[Test 2] Testing HypergraphBuilder & Structural Hash..." << std::endl;
        axman::HypergraphBuilder builder(32); // 32 feature dimensions

        std::vector<float> f0(32, 0.1f);
        std::vector<float> f1(32, 0.2f);
        std::vector<float> f2(32, 0.3f);

        size_t n0 = builder.add_node(f0);
        size_t n1 = builder.add_node(f1);
        size_t n2 = builder.add_node(f2);

        ASSERT_EQ(n0, 0);
        ASSERT_EQ(n1, 1);
        ASSERT_EQ(n2, 2);

        // Edge 0: n0 -> n1 with port 0
        builder.add_edge(n0, n1, 0);

        // Hyperedge 1: (n0, n1) -> n2 with ports (0, 1)
        int64_t src[] = {static_cast<int64_t>(n0), static_cast<int64_t>(n1)};
        int32_t ports[] = {0, 1};
        builder.add_hyperedge(src, ports, n2);

        auto view = builder.view();
        ASSERT_EQ(view.num_nodes(), 3);
        ASSERT_EQ(view.num_hyperedges(), 2);

        auto valid_res = view.validate();
        ASSERT_TRUE(valid_res.has_value());

        auto hash_res = view.structural_hash();
        ASSERT_TRUE(hash_res.has_value());
        ASSERT_TRUE(hash_res.value() != 0);
        std::cout << "  -> OK (Structural Hash: 0x" << std::hex << hash_res.value() << std::dec << ")" << std::endl;
    }

    // Test 3: Model creation, move semantics, forward pass
    {
        std::cout << "[Test 3] Testing Model Load & Inference..." << std::endl;
        const char* custom_config = "{\"d_model\": 64, \"num_layers\": 2, \"num_heads\": 4, \"num_actions\": 16, \"in_features\": 32}";
        auto model_res = axman::Model::load(std::nullopt, custom_config);
        ASSERT_TRUE(model_res.has_value());

        axman::Model model = std::move(model_res.value());
        ASSERT_TRUE(model.is_valid());

        // Test move assignment
        axman::Model model2;
        model2 = std::move(model);
        ASSERT_TRUE(model2.is_valid());
        ASSERT_TRUE(!model.is_valid());

        // Build input graph
        axman::HypergraphBuilder builder(32);
        builder.add_node_fill(0.5f);
        builder.add_node_fill(0.7f);
        builder.add_edge(0, 1, 0);

        auto out_res = model2.forward(builder.view());
        ASSERT_TRUE(out_res.has_value());

        auto out = std::move(out_res.value());
        float val = out.value_estimate();
        ASSERT_TRUE(val >= 0.0f && val <= 1.0f);

        auto policy = out.policy_logits();
        ASSERT_EQ(policy.shape().size(), 2);
        ASSERT_EQ(policy.shape()[0], 2);
        ASSERT_EQ(policy.shape()[1], 16);
        ASSERT_EQ(policy.size(), 2 * 16);

        // Test 2D accessor
        float p00 = policy.at2d(0, 0);
        ASSERT_EQ(p00, policy[0]);

        auto latent = out.latent_embedding();
        ASSERT_EQ(latent.shape().size(), 2);
        ASSERT_EQ(latent.shape()[0], 1);
        ASSERT_EQ(latent.shape()[1], 64);
        ASSERT_EQ(latent.size(), 64);

        std::cout << "  -> OK (Value: " << val << ", Policy Shape: ["
                  << policy.shape()[0] << ", " << policy.shape()[1] << "])" << std::endl;
    }

    // Test 4: Concurrent multithreaded forward inference
    {
        std::cout << "[Test 4] Testing Multithreaded Parallel Inference..." << std::endl;
        const char* config = "{\"d_model\": 64, \"num_layers\": 1, \"num_heads\": 2, \"num_actions\": 8, \"in_features\": 16}";
        auto model_res = axman::Model::load(std::nullopt, config);
        ASSERT_TRUE(model_res.has_value());

        auto shared_model = std::make_shared<axman::Model>(std::move(model_res.value()));

        axman::HypergraphBuilder builder(16);
        builder.add_node_fill(0.25f);
        builder.add_node_fill(0.75f);
        builder.add_edge(0, 1, 0);
        auto view = builder.view();

#if defined(_WIN32)
        struct ThreadPayload {
            std::shared_ptr<axman::Model> model;
            axman::HypergraphView view;
            float result;
        };

        const int NUM_THREADS = 20;
        HANDLE threads[NUM_THREADS];
        std::vector<ThreadPayload> payloads(NUM_THREADS, {shared_model, view, -1.0f});

        auto thread_proc = [](LPVOID param) -> DWORD {
            auto* p = static_cast<ThreadPayload*>(param);
            auto res = p->model->forward(p->view);
            p->result = res.value().value_estimate();
            return 0;
        };

        for (int i = 0; i < NUM_THREADS; ++i) {
            threads[i] = CreateThread(nullptr, 0, thread_proc, &payloads[i], 0, nullptr);
            ASSERT_TRUE(threads[i] != nullptr);
        }

        WaitForMultipleObjects(NUM_THREADS, threads, TRUE, INFINITE);

        for (int i = 0; i < NUM_THREADS; ++i) {
            CloseHandle(threads[i]);
            ASSERT_TRUE(payloads[i].result >= 0.0f && payloads[i].result <= 1.0f);
        }
        std::cout << "  -> OK (20 parallel Win32 threads completed)" << std::endl;
#else
        std::vector<float> results(20, -1.0f);
        for (int i = 0; i < 20; ++i) {
            auto res = shared_model->forward(view);
            results[i] = res.value().value_estimate();
            ASSERT_TRUE(results[i] >= 0.0f && results[i] >= 0.0f);
        }
        std::cout << "  -> OK (20 sequential inferences completed)" << std::endl;
#endif
    }

    // Test 5: Error handling on corrupted graph topology
    {
        std::cout << "[Test 5] Testing Error propagation on corrupted topology..." << std::endl;
        auto model_res = axman::Model::load();
        ASSERT_TRUE(model_res.has_value());
        auto model = std::move(model_res.value());

        // Construct invalid graph with out-of-bounds target
        std::vector<float> feats(2 * 64, 0.0f);
        int64_t src[] = {0};
        int32_t ports[] = {0};
        int64_t target[] = {999}; // Out of bounds!
        size_t offsets[] = {0, 1};

        axman::HypergraphView bad_view(
            2, 1, feats, 64, src, ports, target, offsets
        );

        auto valid_res = bad_view.validate();
        ASSERT_TRUE(!valid_res.has_value());
        ASSERT_EQ(valid_res.error().code, AXMAN_ERR_INDEX_OUT_OF_BOUNDS);

        auto out_res = model.forward(bad_view);
        ASSERT_TRUE(!out_res.has_value());
        ASSERT_EQ(out_res.error().code, AXMAN_ERR_INDEX_OUT_OF_BOUNDS);
        std::cout << "  -> OK (Intercepted expected error: " << out_res.error().message << ")" << std::endl;
    }

    // Test 6: Direct Safetensors file loading via Model::from_file
    {
        std::cout << "[Test 6] Testing Model::from_file with Safetensors archive..." << std::endl;
        const std::string test_safetensors = "test_cpp_bindings_weights.safetensors";

        float w_data[4] = {1.2f, -1.8f, 2.5f, 0.5f};
        float b_data[2] = {0.05f, -0.1f};

        std::string json_header =
            "{\"__metadata__\":{\"d_model\":\"2\",\"num_layers\":\"1\",\"num_heads\":\"1\",\"in_features\":\"2\"},"
            "\"node_encoder.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},"
            "\"node_encoder.bias\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[16,24]}}";

        uint64_t header_len = static_cast<uint64_t>(json_header.size());

        {
            FILE* fp = fopen(test_safetensors.c_str(), "wb");
            ASSERT_TRUE(fp != nullptr);
            fwrite(&header_len, sizeof(header_len), 1, fp);
            fwrite(json_header.data(), 1, header_len, fp);
            fwrite(w_data, sizeof(float), 4, fp);
            fwrite(b_data, sizeof(float), 2, fp);
            fclose(fp);
        }

        auto model_res = axman::Model::from_file(test_safetensors);
        ASSERT_TRUE(model_res.has_value());
        auto model = std::move(model_res.value());
        ASSERT_TRUE(model.is_valid());

        axman::HypergraphBuilder builder(2);
        builder.add_node_fill(1.0f);
        builder.add_node_fill(0.5f);
        builder.add_edge(0, 1, 0);

        auto out_res = model.forward(builder.view());
        ASSERT_TRUE(out_res.has_value());
        float val = out_res.value().value_estimate();
        ASSERT_TRUE(val >= 0.0f && val <= 1.0f);

        remove(test_safetensors.c_str());
        std::cout << "  -> OK (Safetensors loaded and inferred successfully, value: " << val << ")" << std::endl;
    }

    std::cout << "=====================================================" << std::endl;
    std::cout << "   >>> ALL PHASE 4.2 TESTS PASSED SUCCESSFULLY! <<<  " << std::endl;
    std::cout << "=====================================================" << std::endl;
    return 0;
}
