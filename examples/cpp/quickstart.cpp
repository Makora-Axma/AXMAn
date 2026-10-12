/**
 * @file quickstart.cpp
 * @brief AXMANet Modern C++20 Quickstart Example.
 *
 * Demonstrates:
 * 1. HypergraphBuilder fluent construction.
 * 2. Topological invariant verification and 64-bit structural hash computation.
 * 3. RAII Model instantiation and zero-copy forward pass.
 * 4. Safe inspection of output tensors via TensorSpan views.
 */

#include "axman/axman.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << "============================================================\n";
    std::cout << "       AXMANet C++20 Quickstart: Formal Reasoning Inference  \n";
    std::cout << "============================================================\n";

    constexpr size_t feat_dim = 256;

    // 1. Construct a Port-Hypergraph using HypergraphBuilder
    axman::HypergraphBuilder builder(feat_dim);

    // Add 4 nodes with unit/zero features
    std::vector<float> f_hyp(feat_dim, 0.1f);
    std::vector<float> f_p1(feat_dim, 0.2f);
    std::vector<float> f_p2(feat_dim, 0.3f);
    std::vector<float> f_goal(feat_dim, 0.4f);

    size_t h_idx = builder.add_node(f_hyp);
    size_t p1_idx = builder.add_node(f_p1);
    size_t p2_idx = builder.add_node(f_p2);
    size_t goal_idx = builder.add_node(f_goal);

    // Binary hyperedge: (Premise 1, Premise 2) -> Goal
    int64_t sources[2] = {static_cast<int64_t>(p1_idx), static_cast<int64_t>(p2_idx)};
    int32_t ports[2] = {0, 1};
    builder.add_hyperedge(sources, ports, static_cast<int64_t>(goal_idx));

    // Unary hyperedge: Hypothesis -> Premise 1
    builder.add_edge(static_cast<int64_t>(h_idx), static_cast<int64_t>(p1_idx), 0);

    auto graph_view = builder.view();
    std::cout << "[1] Hypergraph assembled: " << graph_view.num_nodes() 
              << " nodes, " << graph_view.num_hyperedges() << " hyperedges.\n";

    // 2. Validate topology and compute structural hash
    auto val_res = graph_view.validate();
    if (!val_res) {
        std::cerr << "[ERROR] Validation failed: " << val_res.error().to_string() << "\n";
        return 1;
    }

    auto hash_res = graph_view.structural_hash();
    if (!hash_res) {
        std::cerr << "[ERROR] Hash failed: " << hash_res.error().to_string() << "\n";
        return 1;
    }
    std::cout << "[2] Topology valid. Structural Hash: 0x" 
              << std::hex << std::setw(16) << std::setfill('0') << hash_res.value() 
              << std::dec << "\n";

    // 3. Model Configuration (JSON)
    std::string config_json = R"({
        "d_model": 256,
        "num_layers": 4,
        "num_heads": 8,
        "port_dim": 64,
        "num_actions": 17,
        "device": "cpu"
    })";

    // 4. Instantiate Model and execute zero-copy forward pass
    std::cout << "[3] Running zero-copy forward inference...\n";
    auto model_res = axman::Model::from_config(config_json);
    if (!model_res) {
        std::cerr << "[ERROR] Failed to load model: " << model_res.error().to_string() << "\n";
        return 1;
    }

    axman::Model model = std::move(model_res).value();
    auto out_res = model.forward(graph_view);
    if (!out_res) {
        std::cerr << "[ERROR] Forward pass failed: " << out_res.error().to_string() << "\n";
        return 1;
    }

    const axman::Output& output = out_res.value();
    std::cout << "  -> Value Estimate (Distance to Closure): " << output.value_estimate() << "\n";
    std::cout << "  -> Policy Logits size: " << output.policy_logits().size() << " floats\n";
    std::cout << "  -> Latent Invariant size: " << output.latent_embedding().size() << " floats\n";

    std::cout << "\n[SUCCESS] AXMANet C++20 Quickstart completed successfully!\n";
    return 0;
}
