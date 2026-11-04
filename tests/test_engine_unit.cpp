#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <cassert>
#include <cstring>
#include <limits>
#include <random>
#include "alg_engine.h"
#include "alg_test_hooks.h"
#include "../src/platform/allocator.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "\n[-] Assertion FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return 1; \
        } \
    } while (0)

#define RUN_TEST(fn) \
    do { \
        std::cout << "[RUNNING] " << #fn << "..." << std::flush; \
        if (fn() == 0) { \
            std::cout << " [PASSED]\n"; \
        } else { \
            std::cout << " [FAILED]\n"; \
            failed_tests++; \
        } \
    } while (0)

// 1. Invalid Arguments & Out-Count Guarantee
static int test_invalid_arguments_and_out_count() {
    // Verify out_index is set to NULL on any failure, even with stale pointer
    AlgIndex* stale_index = (AlgIndex*)(uintptr_t)0xDEADBEEF;
    AlgStatus status = alg_create_index(nullptr, &stale_index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "NULL config must fail");
    TEST_ASSERT(stale_index == nullptr, "out_index must be set to NULL on failure");

    stale_index = (AlgIndex*)(uintptr_t)0xCAFEBABE;
    AlgIndexConfig cfg = {0, 100, ALG_METRIC_L2_SQR};
    status = alg_create_index(&cfg, &stale_index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "Dimension 0 must fail");
    TEST_ASSERT(stale_index == nullptr, "out_index must be set to NULL on dimension 0");

    cfg.dimension = ALG_MAX_DIMENSION + 1;
    stale_index = (AlgIndex*)(uintptr_t)0x12345678;
    status = alg_create_index(&cfg, &stale_index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "Dimension > ALG_MAX_DIMENSION must fail");
    TEST_ASSERT(stale_index == nullptr, "out_index must be set to NULL on dimension overflow");

    AlgIndex* index = nullptr;
    cfg.dimension = 64;
    cfg.metric = (AlgMetric)999;
    status = alg_create_index(&cfg, &index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "Invalid metric must fail");
    TEST_ASSERT(index == nullptr, "out_index must be NULL on invalid metric");

    // NULL index operations
    float q[64] = {0};
    AlgSearchResult res[5];
    uint32_t count = 999;
    status = alg_search(nullptr, q, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(status == ALG_NOT_INITIALIZED, "NULL index search must return ALG_NOT_INITIALIZED");
    TEST_ASSERT(count == 0, "Error must guarantee count == 0");

    // top_k = 0
    cfg.metric = ALG_METRIC_L2_SQR;
    alg_create_index(&cfg, &index);
    float corpus[64 * 10] = {0};
    alg_set_vectors(index, 10, corpus);

    count = 999;
    status = alg_search(index, q, 0, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "top_k = 0 must fail");
    TEST_ASSERT(count == 0, "Error must guarantee count == 0");

    alg_free_index(index);
    return 0;
}

// 2. Search Mode Contract
static int test_search_mode_contract() {
    AlgIndexConfig cfg = {64, 10, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    float corpus[64 * 10] = {0};
    alg_set_vectors(index, 10, corpus);

    float q[64] = {0};
    AlgSearchResult res[5];
    uint32_t count = 999;

    // Mode ALG_SEARCH_EXACT_TOPK is reserved for future refinement pass
    AlgStatus status = alg_search(index, q, 5, ALG_SEARCH_EXACT_TOPK, res, &count);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "ALG_SEARCH_EXACT_TOPK must return ALG_INVALID_ARGUMENT");
    TEST_ASSERT(count == 0, "count must be 0 on error");

    // Fast float32 mode must succeed
    status = alg_search(index, q, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(status == ALG_OK, "ALG_SEARCH_FAST_FLOAT32 must succeed");
    TEST_ASSERT(count == 5, "count must be 5 on success");

    alg_free_index(index);
    return 0;
}

// 3. Finiteness Validation (NaN and Infinities)
static int test_finiteness_validation_nan_inf() {
    const uint32_t dim = 64;
    AlgIndexConfig cfg = {dim, 10, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(10 * dim, 1.0f);
    // Inject NaN into corpus
    corpus[5 * dim + 3] = std::numeric_limits<float>::quiet_NaN();
    AlgStatus st = alg_set_vectors(index, 10, corpus.data());
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Corpus with NaN must be rejected");

    // Inject +Infinity
    corpus[5 * dim + 3] = std::numeric_limits<float>::infinity();
    st = alg_set_vectors(index, 10, corpus.data());
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Corpus with +Inf must be rejected");

    // Inject -Infinity
    corpus[5 * dim + 3] = -std::numeric_limits<float>::infinity();
    st = alg_set_vectors(index, 10, corpus.data());
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Corpus with -Inf must be rejected");

    // Set valid corpus
    corpus[5 * dim + 3] = 1.0f;
    st = alg_set_vectors(index, 10, corpus.data());
    TEST_ASSERT(st == ALG_OK, "Valid corpus must be accepted");

    // Test Query with NaN
    std::vector<float> query(dim, 1.0f);
    query[12] = std::numeric_limits<float>::quiet_NaN();
    AlgSearchResult res[5];
    uint32_t count = 999;

    st = alg_search(index, query.data(), 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Query with NaN must be rejected");
    TEST_ASSERT(count == 0, "count must be reset to 0 on NaN error");

    // Test Query with +Inf
    query[12] = std::numeric_limits<float>::infinity();
    count = 999;
    st = alg_search(index, query.data(), 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Query with +Inf must be rejected");
    TEST_ASSERT(count == 0, "count must be reset to 0 on Inf error");

    // Test raw distance with non-finite inputs
    float d = alg_distance_l2_sqr(corpus.data(), query.data(), dim);
    TEST_ASSERT(std::isnan(d), "alg_distance_l2_sqr must return NaN for non-finite inputs");

    alg_free_index(index);
    return 0;
}

// 4. Set Vectors Atomicity on Failure
static int test_set_vectors_atomicity_on_failure() {
    const uint32_t dim = 32;
    const uint32_t n = 50;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus1(n * dim, 2.5f);
    AlgStatus st = alg_set_vectors(index, n, corpus1.data());
    TEST_ASSERT(st == ALG_OK, "Initial corpus set failed");

    // Attempt to set invalid corpus containing NaN
    std::vector<float> corpus_bad(n * dim, 1.0f);
    corpus_bad[10] = std::numeric_limits<float>::quiet_NaN();
    st = alg_set_vectors(index, n, corpus_bad.data());
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Bad corpus should fail");

    // Existing index must still be 100% valid and searchable
    std::vector<float> q(dim, 2.5f);
    AlgSearchResult res[5];
    uint32_t count = 0;
    st = alg_search(index, q.data(), 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_OK, "Index must remain operational after rejected update");
    TEST_ASSERT(count == 5, "Should return 5 neighbors");
    TEST_ASSERT(res[0].distance == 0.0f, "Should match 2.5f corpus with distance 0");

    alg_free_index(index);
    return 0;
}

// 5. Arithmetic Overflow Unit Tests
static int test_arithmetic_overflow_protection() {
    size_t result = 0;

    // 2-way multiplication overflow
    TEST_ASSERT(alg_mul_overflow_2(SIZE_MAX / 2 + 1, 2, &result) == true, "Must detect 2-way overflow");
    TEST_ASSERT(alg_mul_overflow_2(SIZE_MAX, 2, &result) == true, "Must detect overflow with SIZE_MAX");
    TEST_ASSERT(alg_mul_overflow_2(1000, 2000, &result) == false && result == 2000000, "Valid 2-way multiplication must succeed");

    // 3-way multiplication overflow
    TEST_ASSERT(alg_mul_overflow_3(SIZE_MAX / 4 + 1, 2, 2, &result) == true, "Must detect 3-way overflow");
    TEST_ASSERT(alg_mul_overflow_3(100, 200, 4, &result) == false && result == 80000, "Valid 3-way multiplication must succeed");

    // Allocator overflow guard: bytes > SIZE_MAX - 63
    void* ptr = alg_aligned_alloc_64(SIZE_MAX - 10);
    TEST_ASSERT(ptr == nullptr, "Allocator must reject allocation exceeding SIZE_MAX - 63");

    // alg_search_buffer bounds and overflow validation
    float dummy_q[64] = {0};
    float dummy_buf[64] = {0};
    AlgSearchResult res[5];
    uint32_t out_count = 999;

    // Dimension > ALG_MAX_DIMENSION
    AlgStatus st = alg_search_buffer(dummy_q, dummy_buf, nullptr, 1, ALG_MAX_DIMENSION + 1, 5,
                                     ALG_SEARCH_FAST_FLOAT32, res, &out_count);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "alg_search_buffer with dimension > ALG_MAX_DIMENSION must fail");
    TEST_ASSERT(out_count == 0, "out_count must be reset to 0 on failure");

    // Candidate count causing multiplication overflow
    out_count = 999;
    st = alg_search_buffer(dummy_q, dummy_buf, nullptr, 0xFFFFFFFFu, 0xFFFFFFFFu, 5,
                           ALG_SEARCH_FAST_FLOAT32, res, &out_count);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "alg_search_buffer with overflowing count*dim must fail");
    TEST_ASSERT(out_count == 0, "out_count must be reset to 0 on overflow");

    return 0;
}

// 6. Simulated OOM Failure Propagation
static int test_simulated_oom_propagation() {
    AlgIndexConfig cfg = {64, 100, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;

    // 1. Simulate OOM during index creation
    alg_test_simulate_oom(1);
    AlgStatus st = alg_create_index(&cfg, &index);
    TEST_ASSERT(st == ALG_OUT_OF_MEMORY, "Index creation under OOM must return ALG_OUT_OF_MEMORY");
    TEST_ASSERT(index == nullptr, "Failed index must remain null");
    alg_test_simulate_oom(0);

    // 2. Create valid index
    st = alg_create_index(&cfg, &index);
    TEST_ASSERT(st == ALG_OK, "Index creation must succeed");

    std::vector<float> corpus(100 * 64, 1.0f);
    st = alg_set_vectors(index, 100, corpus.data());
    TEST_ASSERT(st == ALG_OK, "Vector set must succeed");

    // 3. Simulate OOM during corpus reallocation
    std::vector<float> big_corpus(500 * 64, 2.0f);
    alg_test_simulate_oom(1);
    st = alg_set_vectors(index, 500, big_corpus.data());
    TEST_ASSERT(st == ALG_OUT_OF_MEMORY, "Corpus reallocation under OOM must return ALG_OUT_OF_MEMORY");
    alg_test_simulate_oom(0);

    // Prior corpus of 100 vectors must still be valid
    std::vector<float> q(64, 1.0f);
    AlgSearchResult res[5];
    uint32_t count = 0;
    st = alg_search(index, q.data(), 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_OK && count == 5 && res[0].distance == 0.0f, "Prior index must remain operational after OOM");

    // 4. Simulate OOM in dynamic Top-K heap allocation (K = 750 > 512)
    std::vector<AlgSearchResult> big_res(750);
    count = 999;
    alg_test_simulate_oom(1);
    st = alg_search(index, q.data(), 750, ALG_SEARCH_FAST_FLOAT32, big_res.data(), &count);
    TEST_ASSERT(st == ALG_OUT_OF_MEMORY, "Search with K > 512 under OOM must return ALG_OUT_OF_MEMORY");
    TEST_ASSERT(count == 0, "out_count must be reset to 0 on heap OOM");
    alg_test_simulate_oom(0);

    alg_free_index(index);
    return 0;
}

// 7. Strict Compact Filter Contract & Candidate Validation
static int test_compact_search_strict_ids_and_out_of_bounds() {
    const uint32_t dim = 64;
    const uint32_t n = 100;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim, 0.5f);
    alg_set_vectors(index, n, corpus.data());

    std::vector<float> q(dim, 0.5f);
    AlgSearchResult res[10];
    uint32_t count = 999;

    // NULL eligible_ids must return ALG_INVALID_ARGUMENT
    AlgStatus st = alg_search_compact(index, q.data(), nullptr, 10, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "NULL eligible_ids must return ALG_INVALID_ARGUMENT");
    TEST_ASSERT(count == 0, "count must be reset to 0");

    // Candidate ID out of bounds (>= n)
    uint32_t bad_ids[3] = {10, 50, 105}; // 105 >= n (100)
    count = 999;
    st = alg_search_compact(index, q.data(), bad_ids, 3, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Out of bounds candidate ID must fail");
    TEST_ASSERT(count == 0, "count must be reset to 0");

    // Empty candidate list (count = 0)
    uint32_t valid_ids[1] = {5};
    count = 999;
    st = alg_search_compact(index, q.data(), valid_ids, 0, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_OK, "Empty candidate list must succeed");
    TEST_ASSERT(count == 0, "count must be 0 for empty list");

    alg_free_index(index);
    return 0;
}

// 8. Large Top-K (> 512 Dynamic Heap Allocation)
static int test_topk_greater_than_512() {
    const uint32_t dim = 64;
    const uint32_t n = 1000;
    const uint32_t target_k = 750; // Exercises dynamic heap path (> 512)

    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            corpus[i * dim + d] = static_cast<float>(i + d) * 0.01f;
        }
    }
    alg_set_vectors(index, n, corpus.data());

    std::vector<AlgSearchResult> results(target_k);
    uint32_t count = 0;
    AlgStatus st = alg_search(index, corpus.data(), target_k, ALG_SEARCH_FAST_FLOAT32, results.data(), &count);
    TEST_ASSERT(st == ALG_OK, "Dynamic Top-K search failed");
    TEST_ASSERT(count == target_k, "Should retrieve exactly target_k results");

    // Verify sorted ascending
    for (size_t i = 1; i < count; ++i) {
        TEST_ASSERT(results[i].distance >= results[i - 1].distance, "Top-K results must be sorted ascending");
    }

    alg_free_index(index);
    return 0;
}

// 9. Top-K Larger Than Candidate Count
static int test_topk_larger_than_candidate_count() {
    const uint32_t dim = 32;
    const uint32_t n = 10;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim, 1.0f);
    alg_set_vectors(index, n, corpus.data());

    std::vector<float> q(dim, 1.0f);
    AlgSearchResult res[50];
    uint32_t count = 0;

    // Search with top_k = 50 on corpus of 10
    AlgStatus st = alg_search(index, q.data(), 50, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_OK, "Search failed");
    TEST_ASSERT(count == 10, "Must return min(top_k, n) = 10 results");

    alg_free_index(index);
    return 0;
}

// 10. Duplicate Candidate IDs Handling
static int test_duplicate_candidate_ids() {
    const uint32_t dim = 64;
    const uint32_t n = 20;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < corpus.size(); ++i) corpus[i] = (float)i * 0.01f;
    alg_set_vectors(index, n, corpus.data());

    // Duplicate candidate IDs
    uint32_t dup_ids[6] = {5, 5, 12, 12, 5, 12};
    std::vector<float> q = std::vector<float>(corpus.begin(), corpus.begin() + dim);

    AlgSearchResult res[10];
    uint32_t count = 0;
    AlgStatus st = alg_search_compact(index, q.data(), dup_ids, 6, 10, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(st == ALG_OK, "Duplicate candidate search must succeed without error");
    TEST_ASSERT(count == 6, "Must process each candidate instance");

    alg_free_index(index);
    return 0;
}

// 11. Exhaustive Scalar vs AVX2 Ranking Equivalence (Position by Position)
static int test_exhaustive_scalar_avx2_ranking_equivalence() {
    std::vector<uint32_t> test_dims = {1, 7, 8, 9, 63, 64, 65, 127, 128, 256};
    const uint32_t n = 200;
    const uint32_t top_k = 10;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(0.1f, 10.0f);

    for (uint32_t dim : test_dims) {
        AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
        AlgIndex* index = nullptr;
        alg_create_index(&cfg, &index);

        std::vector<float> corpus((size_t)n * dim);
        for (size_t i = 0; i < corpus.size(); ++i) {
            corpus[i] = dist(rng);
        }
        alg_set_vectors(index, n, corpus.data());

        // Test 15 queries per dimension (including exact self-match target 0 and 42)
        std::vector<std::vector<float>> test_queries;
        test_queries.push_back(std::vector<float>(corpus.begin(), corpus.begin() + dim));
        test_queries.push_back(std::vector<float>(corpus.begin() + 42 * dim, corpus.begin() + 43 * dim));

        for (int q_rand = 0; q_rand < 13; ++q_rand) {
            std::vector<float> q(dim);
            for (size_t d = 0; d < dim; ++d) q[d] = dist(rng);
            test_queries.push_back(q);
        }

        for (const auto& q : test_queries) {
            // Run AVX2
            alg_test_force_scalar_isa(0);
            AlgSearchResult res_avx2[10];
            uint32_t count_avx2 = 0;
            AlgStatus st1 = alg_search(index, q.data(), top_k, ALG_SEARCH_FAST_FLOAT32, res_avx2, &count_avx2);
            TEST_ASSERT(st1 == ALG_OK, "AVX2 search failed");

            // Run Scalar
            alg_test_force_scalar_isa(1);
            AlgSearchResult res_scalar[10];
            uint32_t count_scalar = 0;
            AlgStatus st2 = alg_search(index, q.data(), top_k, ALG_SEARCH_FAST_FLOAT32, res_scalar, &count_scalar);
            TEST_ASSERT(st2 == ALG_OK, "Scalar search failed");

            TEST_ASSERT(count_avx2 == count_scalar, "Count mismatch between AVX2 and Scalar");

            // Verify position-by-position ranking and ID equality
            for (uint32_t r = 0; r < count_avx2; ++r) {
                TEST_ASSERT(res_avx2[r].id == res_scalar[r].id,
                            "Scalar and AVX2 Top-K ranking ID mismatch at position");
                float dist_diff = std::fabs(res_avx2[r].distance - res_scalar[r].distance);
                float rel_diff = dist_diff / ((res_avx2[r].distance > 1.0f) ? res_avx2[r].distance : 1.0f);
                TEST_ASSERT(rel_diff < 1e-4f,
                            "Scalar and AVX2 relative distance mismatch at position");
            }
        }

        alg_test_force_scalar_isa(0);
        alg_free_index(index);
    }

    return 0;
}

// 12. Multithreaded Read Concurrency with Serial Reference Comparison
static int test_multithreaded_read_concurrency_serial_reference() {
    const uint32_t dim = 64;
    const uint32_t n = 500;
    const uint32_t num_queries = 30;
    const uint32_t top_k = 5;

    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            corpus[i * dim + d] = static_cast<float>(i + 1) * 0.1f + static_cast<float>(d) * 0.001f;
        }
    }
    alg_set_vectors(index, n, corpus.data());

    // Generate fixed queries
    std::vector<std::vector<float>> queries(num_queries, std::vector<float>(dim));
    for (size_t q = 0; q < num_queries; ++q) {
        for (size_t d = 0; d < dim; ++d) {
            queries[q][d] = static_cast<float>(q * 13 + d) * 0.05f;
        }
    }

    // 1. Compute serial reference results for all queries
    std::vector<std::vector<AlgSearchResult>> serial_ref(num_queries, std::vector<AlgSearchResult>(top_k));
    for (size_t q = 0; q < num_queries; ++q) {
        uint32_t out_count = 0;
        AlgStatus st = alg_search(index, queries[q].data(), top_k, ALG_SEARCH_FAST_FLOAT32,
                                  serial_ref[q].data(), &out_count);
        TEST_ASSERT(st == ALG_OK && out_count == top_k, "Serial reference computation failed");
    }

    // 2. Launch 16 concurrent threads querying the same AlgIndex simultaneously
    const int NUM_THREADS = 16;
    const int ITERATIONS = 100;
    std::vector<std::thread> threads;
    std::vector<int> thread_mismatches(NUM_THREADS, 0);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int it = 0; it < ITERATIONS; ++it) {
                uint32_t q_idx = (t * 7 + it) % num_queries;
                AlgSearchResult thread_res[top_k];
                uint32_t count = 0;

                AlgStatus st = alg_search(index, queries[q_idx].data(), top_k, ALG_SEARCH_FAST_FLOAT32,
                                          thread_res, &count);
                if (st != ALG_OK || count != top_k) {
                    thread_mismatches[t]++;
                    continue;
                }

                // Verify every result matches the serial reference position-by-position
                for (uint32_t r = 0; r < top_k; ++r) {
                    if (thread_res[r].id != serial_ref[q_idx][r].id ||
                        std::fabs(thread_res[r].distance - serial_ref[q_idx][r].distance) > 1e-6f) {
                        thread_mismatches[t]++;
                        break;
                    }
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    int total_err = 0;
    for (int err : thread_mismatches) total_err += err;
    TEST_ASSERT(total_err == 0, "Concurrent queries suffered data race or mismatch with serial reference");

    alg_free_index(index);
    return 0;
}

// 13. Concurrent Index Initialization from Fresh State
static int test_concurrent_index_initialization() {
    const int NUM_THREADS = 16;
    std::vector<std::thread> threads;
    std::vector<int> errors(NUM_THREADS, 0);

    // Reset CPU detection cache to uninitialized state
    alg_test_reset_cpu_cache();

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            AlgIndexConfig cfg = {32, 50, ALG_METRIC_L2_SQR};
            AlgIndex* local_idx = nullptr;
            AlgStatus st = alg_create_index(&cfg, &local_idx);
            if (st != ALG_OK || !local_idx) {
                errors[t]++;
                return;
            }

            std::vector<float> corpus(50 * 32, (float)(t + 1));
            st = alg_set_vectors(local_idx, 50, corpus.data());
            if (st != ALG_OK) {
                errors[t]++;
                alg_free_index(local_idx);
                return;
            }

            std::vector<float> q(32, (float)(t + 1));
            AlgSearchResult res[5];
            uint32_t count = 0;
            st = alg_search(local_idx, q.data(), 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
            if (st != ALG_OK || count != 5 || res[0].distance != 0.0f) {
                errors[t]++;
            }

            alg_free_index(local_idx);
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    int total_err = 0;
    for (int err : errors) total_err += err;
    TEST_ASSERT(total_err == 0, "Concurrent index initialization suffered data race or failure");

    return 0;
}

// 14. Deterministic Tie-Breaking
static int test_deterministic_tie_breaking() {
    const uint32_t dim = 64;
    const uint32_t n = 20;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim, 1.0f);
    alg_set_vectors(index, n, corpus.data());

    std::vector<float> query(dim, 1.0f);
    AlgSearchResult results[10];
    uint32_t out_count = 0;

    AlgStatus st = alg_search(index, query.data(), 10, ALG_SEARCH_FAST_FLOAT32, results, &out_count);
    TEST_ASSERT(st == ALG_OK, "Search failed");
    TEST_ASSERT(out_count == 10, "Expected 10 results");

    for (uint32_t r = 0; r < out_count; ++r) {
        TEST_ASSERT(results[r].id == r, "Tie break must order by ID ascending");
        TEST_ASSERT(results[r].distance == 0.0f, "Distance must be exactly 0.0f");
    }

    alg_free_index(index);
    return 0;
}

int main() {
    std::cout << "============================================================\n";
    std::cout << "ALG Vector Engine - Certified Unit & Regression Test Suite\n";
    std::cout << "Version: " << alg_get_version() << "\n";
    std::cout << "Active ISA: " << alg_get_active_isa_name() << "\n";
    std::cout << "Supported CPU Features bitmask: 0x" << std::hex << alg_get_supported_isa() << std::dec << "\n";
    std::cout << "============================================================\n\n";

    int failed_tests = 0;

    RUN_TEST(test_invalid_arguments_and_out_count);
    RUN_TEST(test_search_mode_contract);
    RUN_TEST(test_finiteness_validation_nan_inf);
    RUN_TEST(test_set_vectors_atomicity_on_failure);
    RUN_TEST(test_arithmetic_overflow_protection);
    RUN_TEST(test_simulated_oom_propagation);
    RUN_TEST(test_compact_search_strict_ids_and_out_of_bounds);
    RUN_TEST(test_topk_greater_than_512);
    RUN_TEST(test_topk_larger_than_candidate_count);
    RUN_TEST(test_duplicate_candidate_ids);
    RUN_TEST(test_exhaustive_scalar_avx2_ranking_equivalence);
    RUN_TEST(test_multithreaded_read_concurrency_serial_reference);
    RUN_TEST(test_concurrent_index_initialization);
    RUN_TEST(test_deterministic_tie_breaking);

    std::cout << "\n============================================================\n";
    if (failed_tests == 0) {
        std::cout << "ALL 14 TESTS PASSED CLEANLY (100% SUCCESS)\n";
        std::cout << "============================================================\n";
        return 0;
    } else {
        std::cerr << failed_tests << " TESTS FAILED\n";
        std::cout << "============================================================\n";
        return 1;
    }
}
