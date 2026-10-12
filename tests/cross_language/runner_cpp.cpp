/**
 * @file runner_cpp.cpp
 * @brief Cross-language inference runner for Modern C++20 bindings.
 *
 * Reads golden_graph.json and golden_config.json, runs axman::Model,
 * and dumps outputs to golden_output_cpp.json.
 */

#include "axman/axman.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <iomanip>

// Minimal JSON parser/serializer for golden dataset
static std::string read_file_string(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open " << path << std::endl;
        std::exit(1);
    }
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    return buffer.str();
}

int main(int argc, char** argv) {
    std::string base_dir = (argc > 1) ? argv[1] : ".";
    std::string config_path = base_dir + "/golden_config.json";
    std::string graph_bin_path = base_dir + "/golden_sample.bin";
    std::string out_path = base_dir + "/golden_output_cpp.json";

    std::string config_json = read_file_string(config_path);

    // Read binary sample
    std::ifstream bin_ifs(graph_bin_path, std::ios::binary);
    if (!bin_ifs.is_open()) {
        std::cerr << "Failed to open binary sample " << graph_bin_path << std::endl;
        return 1;
    }

    char magic[4];
    bin_ifs.read(magic, 4);
    uint64_t num_nodes = 0, feat_dim = 0, num_edges = 0, num_sources = 0;
    bin_ifs.read(reinterpret_cast<char*>(&num_nodes), sizeof(num_nodes));
    bin_ifs.read(reinterpret_cast<char*>(&feat_dim), sizeof(feat_dim));
    bin_ifs.read(reinterpret_cast<char*>(&num_edges), sizeof(num_edges));
    bin_ifs.read(reinterpret_cast<char*>(&num_sources), sizeof(num_sources));

    std::vector<float> features(num_nodes * feat_dim);
    bin_ifs.read(reinterpret_cast<char*>(features.data()), features.size() * sizeof(float));

    std::vector<int64_t> sources(num_sources);
    bin_ifs.read(reinterpret_cast<char*>(sources.data()), sources.size() * sizeof(int64_t));

    std::vector<int32_t> ports(num_sources);
    bin_ifs.read(reinterpret_cast<char*>(ports.data()), ports.size() * sizeof(int32_t));

    std::vector<int64_t> targets(num_edges);
    bin_ifs.read(reinterpret_cast<char*>(targets.data()), targets.size() * sizeof(int64_t));

    std::vector<size_t> offsets(num_edges + 1);
    bin_ifs.read(reinterpret_cast<char*>(offsets.data()), offsets.size() * sizeof(size_t));

    // Construct Model & Graph
    auto model_res = axman::Model::load(std::nullopt, config_json);
    if (!model_res.has_value()) {
        std::cerr << "Failed to load C++ model: " << model_res.error().message << std::endl;
        return 1;
    }
    auto model = std::move(model_res.value());

    axman::HypergraphView graph(
        num_nodes,
        num_edges,
        features,
        feat_dim,
        sources,
        ports,
        targets,
        offsets
    );

    auto hash_res = graph.structural_hash();
    if (!hash_res.has_value()) {
        std::cerr << "Hash calculation failed in C++" << std::endl;
        return 1;
    }
    uint64_t structural_hash = hash_res.value();

    auto out_res = model.forward(graph);
    if (!out_res.has_value()) {
        std::cerr << "Forward failed in C++: " << out_res.error().message << std::endl;
        return 1;
    }

    auto out = std::move(out_res.value());
    float val = out.value_estimate();
    auto policy = out.policy_logits();
    auto latent = out.latent_embedding();

    // Write output JSON with full single-precision float accuracy (9 decimal digits)
    std::ofstream ofs(out_path);
    ofs << std::setprecision(9);
    ofs << "{\n";
    ofs << "  \"language\": \"cpp\",\n";
    ofs << "  \"structural_hash\": " << structural_hash << ",\n";
    ofs << "  \"value_estimate\": " << val << ",\n";
    ofs << "  \"policy_logits\": [";
    for (size_t i = 0; i < policy.size(); ++i) {
        ofs << (i > 0 ? ", " : "") << policy[i];
    }
    ofs << "],\n";
    ofs << "  \"latent_embedding\": [";
    for (size_t i = 0; i < latent.size(); ++i) {
        ofs << (i > 0 ? ", " : "") << latent[i];
    }
    ofs << "]\n";
    ofs << "}\n";

    std::cout << "[C++] Executed successfully. Structural hash: " << structural_hash << ", value: " << val << std::endl;
    return 0;
}
