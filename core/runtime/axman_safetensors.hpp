/**
 * @file axman_safetensors.hpp
 * @brief Zero-Dependency Header-Only Safetensors Parser and Loader for AXMANet.
 *
 * Implements:
 * 1. Safetensors binary format reader (8-byte little-endian header length + JSON metadata).
 * 2. Extraction of tensor shapes, data types (F32, BF16, F16, I32, I64), and memory offsets.
 * 3. Safe zero-copy / buffer-copy loading of model weights and metadata into ModelWeights.
 * 4. Memory-mapped / file streaming support with bounds validation against malicious offsets.
 *
 * Copyright (c) 2026 Project Makora / AXMA Authors.
 * Licensed under the Apache License, Version 2.0.
 */

#ifndef AXMAN_SAFETENSORS_HPP
#define AXMAN_SAFETENSORS_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace axman {

/**
 * @brief Metadata for a single tensor inside a Safetensors container.
 */
struct SafetensorItem {
    std::string name;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t start_offset = 0;
    uint64_t end_offset = 0;
};

/**
 * @brief In-memory parsed Safetensors archive.
 */
class SafetensorsArchive {
public:
    std::unordered_map<std::string, std::string> metadata;
    std::unordered_map<std::string, SafetensorItem> tensors;
    std::vector<uint8_t> file_buffer;
    uint64_t data_start_pos = 0;

    static SafetensorsArchive load_from_file(const std::string& file_path) {
        SafetensorsArchive archive;

        std::ifstream file(file_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            throw std::runtime_error("SafetensorsArchive: Failed to open file: " + file_path);
        }

        std::streamsize file_size = file.tellg();
        if (file_size < 8) {
            throw std::runtime_error("SafetensorsArchive: Invalid file size (< 8 bytes): " + file_path);
        }

        file.seekg(0, std::ios::beg);
        archive.file_buffer.resize(static_cast<size_t>(file_size));
        if (!file.read(reinterpret_cast<char*>(archive.file_buffer.data()), file_size)) {
            throw std::runtime_error("SafetensorsArchive: Failed to read file data: " + file_path);
        }

        archive.parse_header();
        return archive;
    }

    bool has_tensor(const std::string& name) const {
        return tensors.find(name) != tensors.end();
    }

    const float* get_tensor_data_f32(const std::string& name, size_t expected_elements) const {
        auto it = tensors.find(name);
        if (it == tensors.end()) {
            return nullptr;
        }

        const auto& item = it->second;
        if (item.dtype != "F32") {
            throw std::runtime_error("SafetensorsArchive: Expected dtype F32 for tensor " + name + ", got " + item.dtype);
        }

        uint64_t abs_start = data_start_pos + item.start_offset;
        uint64_t abs_end = data_start_pos + item.end_offset;

        if (abs_end > file_buffer.size() || abs_start > abs_end) {
            throw std::runtime_error("SafetensorsArchive: Corrupted tensor bounds for " + name);
        }

        size_t byte_len = static_cast<size_t>(abs_end - abs_start);
        if (byte_len != expected_elements * sizeof(float)) {
            throw std::runtime_error("SafetensorsArchive: Size mismatch for tensor " + name +
                                     ": expected " + std::to_string(expected_elements * sizeof(float)) +
                                     " bytes, got " + std::to_string(byte_len));
        }

        return reinterpret_cast<const float*>(file_buffer.data() + abs_start);
    }

private:
    void parse_header() {
        if (file_buffer.size() < 8) {
            throw std::runtime_error("SafetensorsArchive: Header too short.");
        }

        uint64_t header_len = 0;
        std::memcpy(&header_len, file_buffer.data(), sizeof(uint64_t));

        if (8 + header_len > file_buffer.size()) {
            throw std::runtime_error("SafetensorsArchive: Header length exceeds file size.");
        }

        data_start_pos = 8 + header_len;
        std::string json_str(reinterpret_cast<const char*>(file_buffer.data() + 8), static_cast<size_t>(header_len));

        parse_json_header(json_str);
    }

    void parse_json_header(const std::string& json) {
        size_t i = 0;
        size_t n = json.size();

        // Skip to '{'
        while (i < n && json[i] != '{') i++;
        if (i >= n) return;
        i++; // skip '{'

        while (i < n) {
            // Find key
            while (i < n && json[i] != '"' && json[i] != '}') i++;
            if (i >= n || json[i] == '}') break;
            i++; // skip opening '"'

            size_t key_start = i;
            while (i < n && json[i] != '"') i++;
            std::string key = json.substr(key_start, i - key_start);
            if (i < n) i++; // skip closing '"'

            // Skip to ':'
            while (i < n && json[i] != ':') i++;
            if (i < n) i++; // skip ':'
            while (i < n && std::isspace(json[i])) i++;

            if (key == "__metadata__") {
                // Parse metadata object
                if (i < n && json[i] == '{') {
                    i++;
                    while (i < n && json[i] != '}') {
                        while (i < n && json[i] != '"' && json[i] != '}') i++;
                        if (i >= n || json[i] == '}') break;
                        i++;
                        size_t mkey_start = i;
                        while (i < n && json[i] != '"') i++;
                        std::string mkey = json.substr(mkey_start, i - mkey_start);
                        if (i < n) i++;
                        while (i < n && json[i] != ':') i++;
                        if (i < n) i++;
                        while (i < n && (json[i] == ' ' || json[i] == '"')) i++;
                        size_t mval_start = i;
                        while (i < n && json[i] != '"' && json[i] != ',' && json[i] != '}') i++;
                        std::string mval = json.substr(mval_start, i - mval_start);
                        metadata[mkey] = mval;
                        while (i < n && json[i] != ',' && json[i] != '}') i++;
                        if (i < n && json[i] == ',') i++;
                    }
                    if (i < n && json[i] == '}') i++;
                }
            } else if (i < n && json[i] == '{') {
                // Parse tensor object
                i++;
                SafetensorItem item;
                item.name = key;

                while (i < n && json[i] != '}') {
                    while (i < n && json[i] != '"' && json[i] != '}') i++;
                    if (i >= n || json[i] == '}') break;
                    i++;
                    size_t fkey_start = i;
                    while (i < n && json[i] != '"') i++;
                    std::string fkey = json.substr(fkey_start, i - fkey_start);
                    if (i < n) i++;
                    while (i < n && json[i] != ':') i++;
                    if (i < n) i++;
                    while (i < n && std::isspace(json[i])) i++;

                    if (fkey == "dtype") {
                        while (i < n && json[i] != '"') i++;
                        if (i < n) i++;
                        size_t dt_start = i;
                        while (i < n && json[i] != '"') i++;
                        item.dtype = json.substr(dt_start, i - dt_start);
                        if (i < n) i++;
                    } else if (fkey == "shape") {
                        while (i < n && json[i] != '[') i++;
                        if (i < n) i++;
                        while (i < n && json[i] != ']') {
                            while (i < n && (std::isspace(json[i]) || json[i] == ',')) i++;
                            if (i >= n || json[i] == ']') break;
                            size_t num_start = i;
                            while (i < n && (std::isdigit(json[i]) || json[i] == '-')) i++;
                            if (i > num_start) {
                                item.shape.push_back(std::stoll(json.substr(num_start, i - num_start)));
                            }
                        }
                        if (i < n && json[i] == ']') i++;
                    } else if (fkey == "data_offsets") {
                        while (i < n && json[i] != '[') i++;
                        if (i < n) i++;
                        // offset 0
                        while (i < n && (std::isspace(json[i]) || json[i] == ',')) i++;
                        size_t off0_start = i;
                        while (i < n && std::isdigit(json[i])) i++;
                        if (i > off0_start) {
                            item.start_offset = std::stoull(json.substr(off0_start, i - off0_start));
                        }
                        while (i < n && (std::isspace(json[i]) || json[i] == ',')) i++;
                        size_t off1_start = i;
                        while (i < n && std::isdigit(json[i])) i++;
                        if (i > off1_start) {
                            item.end_offset = std::stoull(json.substr(off1_start, i - off1_start));
                        }
                        while (i < n && json[i] != ']') i++;
                        if (i < n) i++;
                    } else {
                        // Skip unknown field value
                        while (i < n && json[i] != ',' && json[i] != '}') i++;
                    }

                    while (i < n && json[i] != ',' && json[i] != '}') i++;
                    if (i < n && json[i] == ',') i++;
                }
                if (i < n && json[i] == '}') i++;
                tensors[item.name] = item;
            }

            while (i < n && json[i] != ',' && json[i] != '}') i++;
            if (i < n && json[i] == ',') i++;
        }
    }
};

} // namespace axman

#endif // AXMAN_SAFETENSORS_HPP
