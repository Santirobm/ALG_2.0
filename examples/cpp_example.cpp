/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>
#include <vector>
#include <memory>
#include "alg_engine.h"

// Modern C++ RAII wrapper for AlgIndex
class VectorIndex {
public:
    explicit VectorIndex(uint32_t dimension, uint32_t capacity = 0) {
        AlgIndexConfig config;
        config.dimension = dimension;
        config.initial_capacity = capacity;
        config.metric = ALG_METRIC_L2_SQR;

        AlgStatus status = alg_create_index(&config, &index_);
        if (status != ALG_OK) {
            throw std::runtime_error(std::string("alg_create_index failed: ") + alg_status_to_string(status));
        }
    }

    ~VectorIndex() {
        if (index_) {
            alg_free_index(index_);
        }
    }

    // Disable copy, enable move
    VectorIndex(const VectorIndex&) = delete;
    VectorIndex& operator=(const VectorIndex&) = delete;
    VectorIndex(VectorIndex&& other) noexcept : index_(other.index_) { other.index_ = nullptr; }
    VectorIndex& operator=(VectorIndex&& other) noexcept {
        if (this != &other) {
            if (index_) alg_free_index(index_);
            index_ = other.index_;
            other.index_ = nullptr;
        }
        return *this;
    }

    void set_vectors(const std::vector<float>& vectors, uint32_t num_vectors) {
        AlgStatus status = alg_set_vectors(index_, num_vectors, vectors.data());
        if (status != ALG_OK) {
            throw std::runtime_error(std::string("alg_set_vectors failed: ") + alg_status_to_string(status));
        }
    }

    std::vector<AlgSearchResult> search(const std::vector<float>& query, uint32_t top_k) const {
        std::vector<AlgSearchResult> results(top_k);
        uint32_t count = 0;
        AlgStatus status = alg_search(index_, query.data(), top_k, ALG_SEARCH_FAST_FLOAT32,
                                      results.data(), &count);
        if (status != ALG_OK) {
            throw std::runtime_error(std::string("alg_search failed: ") + alg_status_to_string(status));
        }
        results.resize(count);
        return results;
    }

    std::vector<AlgSearchResult> search_filtered(
        const std::vector<float>& query,
        const std::vector<uint32_t>& eligible_ids,
        uint32_t top_k
    ) const {
        std::vector<AlgSearchResult> results(top_k);
        uint32_t count = 0;
        AlgStatus status = alg_search_compact(index_, query.data(), eligible_ids.data(),
                                              static_cast<uint32_t>(eligible_ids.size()),
                                              top_k, ALG_SEARCH_FAST_FLOAT32,
                                              results.data(), &count);
        if (status != ALG_OK) {
            throw std::runtime_error(std::string("alg_search_compact failed: ") + alg_status_to_string(status));
        }
        results.resize(count);
        return results;
    }

private:
    AlgIndex* index_ = nullptr;
};

int main() {
    std::cout << "ALG Vector Engine - C++ RAII Consumer Example\n";
    std::cout << "Instruction Set: " << alg_get_active_isa_name() << "\n\n";

    const uint32_t dim = 128;
    const uint32_t n = 2000;

    VectorIndex index(dim, n);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            corpus[i * dim + d] = static_cast<float>((i * 13 + d) % 101) * 0.05f;
        }
    }
    index.set_vectors(corpus, n);

    std::vector<float> query(dim);
    for (size_t d = 0; d < dim; ++d) {
        query[d] = corpus[100 * dim + d]; // Target vector 100
    }

    auto results = index.search(query, 5);
    std::cout << "Top-5 Unfiltered Search Results:\n";
    for (size_t i = 0; i < results.size(); ++i) {
        std::cout << "  Rank " << (i + 1) << ": ID " << results[i].id
                  << ", Dist = " << results[i].distance << "\n";
    }

    // Filtered search: only vectors with ID > 500
    std::vector<uint32_t> eligible;
    for (uint32_t id = 501; id < n; ++id) {
        eligible.push_back(id);
    }

    auto filtered_results = index.search_filtered(query, eligible, 5);
    std::cout << "\nTop-5 Filtered Search Results (IDs > 500):\n";
    for (size_t i = 0; i < filtered_results.size(); ++i) {
        std::cout << "  Rank " << (i + 1) << ": ID " << filtered_results[i].id
                  << ", Dist = " << filtered_results[i].distance << "\n";
    }

    return 0;
}
