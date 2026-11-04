/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "alg_engine.h"

#ifdef _WIN32
#include <windows.h>
static inline uint64_t get_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}
#else
static inline uint64_t get_time_ns() {
    auto now = std::chrono::high_resolution_clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
}
#endif

struct BenchStats {
    double mean_us;
    double stddev_us;
    double p50_us;
    double p95_us;
    double p99_us;
    double qps;
};

static BenchStats compute_stats(std::vector<double>& times_us) {
    std::sort(times_us.begin(), times_us.end());
    size_t n = times_us.size();
    double sum = std::accumulate(times_us.begin(), times_us.end(), 0.0);
    double mean = sum / n;

    double var = 0.0;
    for (double t : times_us) {
        double diff = t - mean;
        var += diff * diff;
    }
    double stddev = std::sqrt(var / n);

    BenchStats s;
    s.mean_us = mean;
    s.stddev_us = stddev;
    s.p50_us = times_us[n * 50 / 100];
    s.p95_us = times_us[n * 95 / 100];
    s.p99_us = times_us[n * 99 / 100];
    s.qps = 1000000.0 / mean;
    return s;
}

int main(int argc, char* argv[]) {
    std::cout << "========================================================================\n";
    std::cout << "ALG Vector Engine - Standalone Production Benchmark Harness\n";
    std::cout << "Engine Version: " << alg_get_version() << "\n";
    std::cout << "Active Instruction Set: " << alg_get_active_isa_name() << "\n";
    std::cout << "========================================================================\n\n";

    uint32_t num_vectors = 100000;
    uint32_t num_queries = 200;
    uint32_t top_k = 10;
    std::vector<uint32_t> dims = {64, 128, 256};
    std::vector<double> selectivities = {0.01, 0.10, 1.00};

    std::cout << std::left
              << std::setw(6)  << "Dim"
              << std::setw(12) << "Selectivity"
              << std::setw(12) << "Candidates"
              << std::setw(12) << "Mean (us)"
              << std::setw(12) << "p50 (us)"
              << std::setw(12) << "p95 (us)"
              << std::setw(12) << "p99 (us)"
              << std::setw(14) << "QPS"
              << "\n";
    std::cout << std::string(82, '-') << "\n";

    for (uint32_t dim : dims) {
        AlgIndexConfig cfg;
        cfg.dimension = dim;
        cfg.initial_capacity = num_vectors;
        cfg.metric = ALG_METRIC_L2_SQR;

        AlgIndex* index = nullptr;
        AlgStatus st = alg_create_index(&cfg, &index);
        if (st != ALG_OK) {
            std::cerr << "Failed to create index for dim " << dim << "\n";
            continue;
        }

        // Initialize synthetic vectors
        std::vector<float> corpus((size_t)num_vectors * dim);
        for (size_t i = 0; i < corpus.size(); ++i) {
            corpus[i] = static_cast<float>((i * 7 + 13) % 1009) * 0.001f;
        }
        alg_set_vectors(index, num_vectors, corpus.data());

        // Prepare queries
        std::vector<float> queries((size_t)num_queries * dim);
        for (size_t i = 0; i < queries.size(); ++i) {
            queries[i] = static_cast<float>((i * 11 + 29) % 1013) * 0.001f;
        }

        for (double sel : selectivities) {
            uint32_t eligible_count = static_cast<uint32_t>(num_vectors * sel);
            if (eligible_count == 0) eligible_count = 1;

            std::vector<uint32_t> eligible_ids(eligible_count);
            for (uint32_t i = 0; i < eligible_count; ++i) {
                eligible_ids[i] = static_cast<uint32_t>(i * (num_vectors / (double)eligible_count));
                if (eligible_ids[i] >= num_vectors) eligible_ids[i] = num_vectors - 1;
            }

            // Warmup
            std::vector<AlgSearchResult> results(top_k);
            uint32_t out_count = 0;
            for (uint32_t w = 0; w < 10; ++w) {
                if (sel >= 0.999) {
                    alg_search(index, &queries[w * dim], top_k, ALG_SEARCH_FAST_FLOAT32, results.data(), &out_count);
                } else {
                    alg_search_compact(index, &queries[w * dim], eligible_ids.data(), eligible_count,
                                       top_k, ALG_SEARCH_FAST_FLOAT32, results.data(), &out_count);
                }
            }

            // Benchmark runs
            std::vector<double> latencies_us;
            latencies_us.reserve(num_queries);

            for (uint32_t q = 0; q < num_queries; ++q) {
                const float* q_ptr = &queries[q * dim];
                uint64_t t0 = get_time_ns();
                if (sel >= 0.999) {
                    alg_search(index, q_ptr, top_k, ALG_SEARCH_FAST_FLOAT32, results.data(), &out_count);
                } else {
                    alg_search_compact(index, q_ptr, eligible_ids.data(), eligible_count,
                                       top_k, ALG_SEARCH_FAST_FLOAT32, results.data(), &out_count);
                }
                uint64_t t1 = get_time_ns();
                latencies_us.push_back(static_cast<double>(t1 - t0) / 1000.0);
            }

            BenchStats stats = compute_stats(latencies_us);

            std::cout << std::left
                      << std::setw(6)  << dim
                      << std::setw(12) << (std::to_string(static_cast<int>(sel * 100)) + "%")
                      << std::setw(12) << eligible_count
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.mean_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.p50_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.p95_us
                      << std::setw(12) << std::fixed << std::setprecision(2) << stats.p99_us
                      << std::setw(14) << std::fixed << std::setprecision(1) << stats.qps
                      << "\n";
        }

        alg_free_index(index);
    }

    std::cout << std::string(82, '-') << "\n";
    std::cout << "[DONE] Benchmark completed successfully.\n";
    return 0;
}
