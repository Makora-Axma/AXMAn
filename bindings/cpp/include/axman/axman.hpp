/**
 * @file axman.hpp
 * @brief Axiomatic eXploration & Morphism Architecture Network (AXMANet)
 *        Modern C++20 / C++23 High-Level Wrapper.
 *
 * Provides:
 * - Idiomatic RAII wrappers (axman::Model, axman::Output)
 * - Safe zero-copy views via std::span
 * - Fluent graph construction via axman::HypergraphBuilder
 * - Result<T, Error> error-handling abstraction (analogous to std::expected)
 * - Move-only semantics, thread safety (read-only concurrent forward)
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef AXMAN_CPP_AXMAN_HPP
#define AXMAN_CPP_AXMAN_HPP

#include "axman_c_api.h"

#include <span>
#include <vector>
#include <string>
#include <string_view>
#include <optional>
#include <memory>
#include <variant>
#include <stdexcept>
#include <sstream>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <iostream>

namespace axman {

/**
 * @brief Strong error type capturing status code and diagnostic text.
 */
struct Error {
    axman_status_t code{AXMAN_ERR_INTERNAL};
    std::string message;

    [[nodiscard]] std::string to_string() const {
        std::ostringstream oss;
        oss << "AXMAN Error [" << axman_status_to_string(code) << " (" << static_cast<int>(code) << ")]: "
            << (message.empty() ? axman_status_to_string(code) : message);
        return oss.str();
    }
};

/**
 * @brief Lightweight Result type modeled after C++23 std::expected<T, Error>.
 */
template <typename T>
class Result {
public:
    Result(T value) : data_(std::move(value)) {}
    Result(Error error) : data_(std::move(error)) {}

    [[nodiscard]] bool has_value() const noexcept {
        return std::holds_alternative<T>(data_);
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return has_value();
    }

    [[nodiscard]] const T& value() const & {
        if (!has_value()) {
            throw std::runtime_error(std::get<Error>(data_).to_string());
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T& value() & {
        if (!has_value()) {
            throw std::runtime_error(std::get<Error>(data_).to_string());
        }
        return std::get<T>(data_);
    }

    [[nodiscard]] T&& value() && {
        if (!has_value()) {
            throw std::runtime_error(std::get<Error>(data_).to_string());
        }
        return std::get<T>(std::move(data_));
    }

    [[nodiscard]] const Error& error() const noexcept {
        return std::get<Error>(data_);
    }

    [[nodiscard]] const T* operator->() const noexcept {
        return &std::get<T>(data_);
    }

    [[nodiscard]] T* operator->() noexcept {
        return &std::get<T>(data_);
    }

    [[nodiscard]] const T& operator*() const & noexcept {
        return std::get<T>(data_);
    }

    [[nodiscard]] T& operator*() & noexcept {
        return std::get<T>(data_);
    }

private:
    std::variant<T, Error> data_;
};

/**
 * @brief Specialization of Result for void returns.
 */
template <>
class Result<void> {
public:
    Result() : error_(std::nullopt) {}
    Result(Error error) : error_(std::move(error)) {}

    [[nodiscard]] bool has_value() const noexcept {
        return !error_.has_value();
    }

    [[nodiscard]] explicit operator bool() const noexcept {
        return has_value();
    }

    void value() const {
        if (error_.has_value()) {
            throw std::runtime_error(error_->to_string());
        }
    }

    [[nodiscard]] const Error& error() const noexcept {
        return error_.value();
    }

private:
    std::optional<Error> error_;
};

/**
 * @brief Helper to create Error from last C-ABI runtime diagnostic.
 */
inline Error make_error(axman_status_t status) {
    const char* raw_msg = axman_get_last_error_message();
    std::string msg;
    if (raw_msg != nullptr && raw_msg[0] != '\0') {
        msg = raw_msg;
    } else {
        msg = axman_status_to_string(status);
    }
    return Error{status, std::move(msg)};
}

/**
 * @brief Zero-copy immutable multi-dimensional view over a C-ABI tensor.
 */
template <typename T>
class TensorSpan {
public:
    TensorSpan() : data_{}, shape_{} {}

    TensorSpan(std::span<const T> data, std::vector<int64_t> shape)
        : data_(data), shape_(std::move(shape)) {}

    [[nodiscard]] std::span<const T> as_span() const noexcept { return data_; }
    [[nodiscard]] const std::vector<int64_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] size_t size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }

    [[nodiscard]] const T& operator[](size_t index) const {
        return data_[index];
    }

    /**
     * @brief 2D indexing convenience: matrix(row, col).
     */
    [[nodiscard]] const T& at2d(size_t row, size_t col) const {
        if (shape_.size() < 2) {
            throw std::out_of_range("TensorSpan rank is less than 2");
        }
        size_t cols = static_cast<size_t>(shape_[1]);
        size_t idx = row * cols + col;
        if (idx >= data_.size()) {
            throw std::out_of_range("TensorSpan index out of bounds");
        }
        return data_[idx];
    }

    auto begin() const noexcept { return data_.begin(); }
    auto end() const noexcept { return data_.end(); }

private:
    std::span<const T> data_;
    std::vector<int64_t> shape_;
};

/**
 * @brief RAII managed output returned by model forward pass.
 * Automatically frees internal buffers via axman_output_free on destruction.
 */
class Output {
public:
    Output() {
        std::memset(&raw_, 0, sizeof(raw_));
    }

    ~Output() {
        reset();
    }

    // Move-only semantics
    Output(const Output&) = delete;
    Output& operator=(const Output&) = delete;

    Output(Output&& other) noexcept {
        raw_ = other.raw_;
        std::memset(&other.raw_, 0, sizeof(other.raw_));
    }

    Output& operator=(Output&& other) noexcept {
        if (this != &other) {
            reset();
            raw_ = other.raw_;
            std::memset(&other.raw_, 0, sizeof(other.raw_));
        }
        return *this;
    }

    void reset() noexcept {
        if (raw_.policy_logits.data != nullptr || raw_.latent_embedding.data != nullptr) {
            axman_output_free(&raw_);
        }
        std::memset(&raw_, 0, sizeof(raw_));
    }

    [[nodiscard]] float value_estimate() const noexcept {
        return raw_.value_estimate;
    }

    [[nodiscard]] TensorSpan<float> policy_logits() const {
        return make_tensor_span<float>(raw_.policy_logits);
    }

    [[nodiscard]] TensorSpan<float> latent_embedding() const {
        return make_tensor_span<float>(raw_.latent_embedding);
    }

    [[nodiscard]] axman_output_t* raw_ptr() noexcept {
        return &raw_;
    }

private:
    template <typename T>
    static TensorSpan<T> make_tensor_span(const axman_tensor_t& t) {
        if (t.data == nullptr || t.ndim == 0 || t.shape == nullptr) {
            return TensorSpan<T>{};
        }
        std::vector<int64_t> shape(t.shape, t.shape + t.ndim);
        size_t total = 1;
        for (auto s : shape) {
            if (s <= 0) {
                total = 0;
                break;
            }
            total *= static_cast<size_t>(s);
        }
        if (total == 0) {
            return TensorSpan<T>{std::span<const T>{}, std::move(shape)};
        }
        return TensorSpan<T>{std::span<const T>(static_cast<const T*>(t.data), total), std::move(shape)};
    }

    axman_output_t raw_{};
};

/**
 * @brief Non-owning view of a Port-Hypergraph structure conforming to C-ABI layout.
 */
class HypergraphView {
public:
    HypergraphView(
        size_t num_nodes,
        size_t num_hyperedges,
        std::span<const float> node_features,
        size_t node_feat_dim,
        std::span<const int64_t> edge_sources,
        std::span<const int32_t> edge_ports,
        std::span<const int64_t> edge_targets,
        std::span<const size_t> edge_offsets
    ) : num_nodes_(num_nodes),
        num_hyperedges_(num_hyperedges),
        node_features_(node_features),
        node_feat_dim_(node_feat_dim),
        edge_sources_(edge_sources),
        edge_ports_(edge_ports),
        edge_targets_(edge_targets),
        edge_offsets_(edge_offsets)
    {
        if (node_features_.size() != num_nodes_ * node_feat_dim_) {
            throw std::invalid_argument("node_features size mismatch with num_nodes * node_feat_dim");
        }
        if (edge_targets_.size() != num_hyperedges_) {
            throw std::invalid_argument("edge_targets size mismatch with num_hyperedges");
        }
        if (edge_offsets_.size() != num_hyperedges_ + 1) {
            throw std::invalid_argument("edge_offsets size mismatch with num_hyperedges + 1");
        }
        if (edge_sources_.size() != edge_ports_.size()) {
            throw std::invalid_argument("edge_sources and edge_ports size mismatch");
        }
        if (num_hyperedges_ > 0) {
            if (edge_offsets_[0] != 0) {
                throw std::invalid_argument("edge_offsets[0] must be 0");
            }
            if (edge_offsets_[num_hyperedges_] != edge_sources_.size()) {
                throw std::invalid_argument("edge_offsets[num_hyperedges] must match edge_sources size");
            }
        }
    }

    [[nodiscard]] size_t num_nodes() const noexcept { return num_nodes_; }
    [[nodiscard]] size_t num_hyperedges() const noexcept { return num_hyperedges_; }
    [[nodiscard]] size_t node_feat_dim() const noexcept { return node_feat_dim_; }

    [[nodiscard]] Result<void> validate() const {
        int64_t shape[2];
        axman_hypergraph_t raw = to_c_abi(shape);
        axman_status_t status = axman_hypergraph_validate(&raw);
        if (status != AXMAN_SUCCESS) {
            return make_error(status);
        }
        return {};
    }

    [[nodiscard]] Result<uint64_t> structural_hash() const {
        int64_t shape[2];
        axman_hypergraph_t raw = to_c_abi(shape);
        uint64_t hash_val = 0;
        axman_status_t status = axman_hypergraph_hash(&raw, &hash_val);
        if (status != AXMAN_SUCCESS) {
            return make_error(status);
        }
        return hash_val;
    }

    [[nodiscard]] axman_hypergraph_t to_c_abi(int64_t shape_out[2]) const noexcept {
        axman_tensor_t feat{};
        if (!node_features_.empty() && num_nodes_ > 0) {
            shape_out[0] = static_cast<int64_t>(num_nodes_);
            shape_out[1] = static_cast<int64_t>(node_feat_dim_);
            feat.data = const_cast<float*>(node_features_.data());
            feat.shape = shape_out;
            feat.ndim = 2;
            feat.dtype = AXMAN_DTYPE_FLOAT32;
            feat.device = AXMAN_DEVICE_CPU;
        }

        axman_hypergraph_t g{};
        g.num_nodes = num_nodes_;
        g.num_hyperedges = num_hyperedges_;
        g.node_features = feat;
        g.edge_sources = edge_sources_.empty() ? nullptr : edge_sources_.data();
        g.edge_ports = edge_ports_.empty() ? nullptr : edge_ports_.data();
        g.edge_targets = edge_targets_.empty() ? nullptr : edge_targets_.data();
        g.edge_offsets = edge_offsets_.empty() ? nullptr : edge_offsets_.data();
        return g;
    }

private:
    size_t num_nodes_{0};
    size_t num_hyperedges_{0};
    std::span<const float> node_features_;
    size_t node_feat_dim_{0};
    std::span<const int64_t> edge_sources_;
    std::span<const int32_t> edge_ports_;
    std::span<const int64_t> edge_targets_;
    std::span<const size_t> edge_offsets_;
};

/**
 * @brief Fluent Builder for assembling Port-Hypergraphs with validation.
 */
class HypergraphBuilder {
public:
    explicit HypergraphBuilder(size_t node_feat_dim)
        : node_feat_dim_(node_feat_dim) {
        edge_offsets_.push_back(0);
    }

    /**
     * @brief Add a node with specified feature vector.
     * @return 0-based node index.
     */
    size_t add_node(std::span<const float> features) {
        if (features.size() != node_feat_dim_) {
            throw std::invalid_argument("features size does not match node_feat_dim");
        }
        node_features_.insert(node_features_.end(), features.begin(), features.end());
        size_t idx = num_nodes_++;
        return idx;
    }

    /**
     * @brief Add a uniform initialized node.
     */
    size_t add_node_fill(float value = 0.0f) {
        std::vector<float> f(node_feat_dim_, value);
        return add_node(f);
    }

    /**
     * @brief Add a directed hyperedge connecting ordered premises via de Bruijn ports to target.
     * @param sources Ordered list of premise node indices.
     * @param ports Ordered list of positional port numbers (0..k-1).
     * @param target Target conclusion node index.
     */
    HypergraphBuilder& add_hyperedge(
        std::span<const int64_t> sources,
        std::span<const int32_t> ports,
        int64_t target
    ) {
        if (sources.size() != ports.size()) {
            throw std::invalid_argument("sources and ports must have the same length");
        }
        if (sources.size() > AXMAN_MAX_HYPEREDGE_ARITY) {
            throw std::invalid_argument("Hyperedge arity exceeds k_max = 16");
        }

        for (auto s : sources) {
            edge_sources_.push_back(s);
        }
        for (auto p : ports) {
            edge_ports_.push_back(p);
        }
        edge_targets_.push_back(target);
        edge_offsets_.push_back(edge_sources_.size());
        return *this;
    }

    /**
     * @brief Add a unary hyperedge (standard edge source -> target with port 0).
     */
    HypergraphBuilder& add_edge(int64_t source, int64_t target, int32_t port = 0) {
        int64_t s[1] = {source};
        int32_t p[1] = {port};
        return add_hyperedge(std::span<const int64_t>(s, 1), std::span<const int32_t>(p, 1), target);
    }

    [[nodiscard]] size_t num_nodes() const noexcept { return num_nodes_; }
    [[nodiscard]] size_t num_hyperedges() const noexcept { return edge_targets_.size(); }

    [[nodiscard]] HypergraphView view() const {
        return HypergraphView(
            num_nodes_,
            edge_targets_.size(),
            node_features_,
            node_feat_dim_,
            edge_sources_,
            edge_ports_,
            edge_targets_,
            edge_offsets_
        );
    }

private:
    size_t node_feat_dim_{0};
    size_t num_nodes_{0};
    std::vector<float> node_features_;
    std::vector<int64_t> edge_sources_;
    std::vector<int32_t> edge_ports_;
    std::vector<int64_t> edge_targets_;
    std::vector<size_t> edge_offsets_;
};

/**
 * @brief High-level RAII wrapper for an instantiated AXMANet model.
 * Thread-safe for concurrent read-only forward passes.
 */
class Model {
public:
    Model() : handle_(nullptr) {}

    ~Model() {
        reset();
    }

    // Move-only semantics
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    Model(Model&& other) noexcept : handle_(other.handle_) {
        other.handle_ = nullptr;
    }

    Model& operator=(Model&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = other.handle_;
            other.handle_ = nullptr;
        }
        return *this;
    }

    void reset() noexcept {
        if (handle_ != nullptr) {
            axman_model_destroy(handle_);
            handle_ = nullptr;
        }
    }

    [[nodiscard]] bool is_valid() const noexcept {
        return handle_ != nullptr;
    }

    /**
     * @brief Factory: load model from weights file and/or JSON configuration.
     * @param model_path Path to weights file (or std::nullopt).
     * @param config_json JSON configuration string (or std::nullopt).
     */
    static Result<Model> load(
        std::optional<std::string_view> model_path = std::nullopt,
        std::optional<std::string_view> config_json = std::nullopt
    ) {
        std::string path_str;
        const char* path_ptr = nullptr;
        if (model_path.has_value()) {
            path_str = std::string(*model_path);
            path_ptr = path_str.c_str();
        }

        std::string json_str;
        const char* json_ptr = nullptr;
        if (config_json.has_value()) {
            json_str = std::string(*config_json);
            json_ptr = json_str.c_str();
        }

        axman_model_handle_t handle = nullptr;
        axman_status_t status = axman_model_create(path_ptr, json_ptr, &handle);
        if (status != AXMAN_SUCCESS) {
            return make_error(status);
        }

        Model m;
        m.handle_ = handle;
        return m;
    }

    /**
     * @brief Factory: load model directly from weights file.
     */
    static Result<Model> from_file(
        std::string_view model_path,
        std::optional<std::string_view> config_json = std::nullopt
    ) {
        return load(model_path, config_json);
    }

    /**
     * @brief Factory: load model with JSON configuration and initialized weights.
     */
    static Result<Model> from_config(std::string_view config_json) {
        return load(std::nullopt, config_json);
    }

    /**
     * @brief Factory: load model with default configuration and initialized weights.
     */
    static Result<Model> default_model() {
        return load(std::nullopt, std::nullopt);
    }

    /**
     * @brief Execute forward inference on an input port-hypergraph.
     */
    [[nodiscard]] Result<Output> forward(const HypergraphView& graph) const {
        if (handle_ == nullptr) {
            return Error{AXMAN_ERR_NULL_POINTER, "Attempted forward inference on invalid model handle"};
        }

        int64_t shape_buf[2];
        axman_hypergraph_t raw_graph = graph.to_c_abi(shape_buf);

        Output out;
        axman_status_t status = axman_model_forward(handle_, &raw_graph, out.raw_ptr());
        if (status != AXMAN_SUCCESS) {
            return make_error(status);
        }

        return out;
    }

    [[nodiscard]] axman_model_handle_t raw_handle() const noexcept {
        return handle_;
    }

private:
    axman_model_handle_t handle_{nullptr};
};

/**
 * @brief Returns library version string.
 */
inline std::string_view version() noexcept {
    const char* v = axman_get_version();
    return v ? std::string_view(v) : std::string_view(AXMAN_VERSION_STRING);
}

} // namespace axman

#endif // AXMAN_CPP_AXMAN_HPP
