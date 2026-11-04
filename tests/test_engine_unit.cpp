#include <iostream>
#include <vector>
#include <thread>
#include <cmath>
#include <cassert>
#include <cstring>
#include <limits>
#include "alg_engine.h"

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
    AlgIndex* index = nullptr;
    AlgStatus status = alg_create_index(nullptr, &index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "NULL config must fail");

    AlgIndexConfig cfg = {0, 100, ALG_METRIC_L2_SQR};
    status = alg_create_index(&cfg, &index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "Dimension 0 must fail");

    cfg.dimension = 64;
    cfg.metric = (AlgMetric)999;
    status = alg_create_index(&cfg, &index);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "Invalid metric must fail");

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

// 5. Arithmetic and Size Overflow Protection
static int test_size_and_allocation_overflow() {
    AlgIndex* index = nullptr;
    // Dimension and capacity causing size allocation failure / overflow
    AlgIndexConfig cfg = {0xFFFFFFFFu, 100, ALG_METRIC_L2_SQR};
    AlgStatus st = alg_create_index(&cfg, &index);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT || st == ALG_OUT_OF_MEMORY, "Overflow or OOM in initial capacity * dim must fail");

    // Capacity overflow
    cfg.dimension = 1024;
    cfg.initial_capacity = 0xFFFFFFF0u;
    st = alg_create_index(&cfg, &index);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT || st == ALG_OUT_OF_MEMORY, "Overflow in capacity must fail");

    // Valid index, overflow in alg_set_vectors
    cfg.dimension = 10000;
    cfg.initial_capacity = 0;
    alg_create_index(&cfg, &index);

    float dummy[10];
    st = alg_set_vectors(index, 0xFFFFFFFFu, dummy);
    TEST_ASSERT(st == ALG_INVALID_ARGUMENT, "Overflow in num_vectors * dim must fail");

    alg_free_index(index);
    return 0;
}

// 6. Strict Compact Filter Contract & Candidate Validation
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

// 7. Large Top-K (> 512 Dynamic Heap Allocation)
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

// 8. Top-K Larger Than Candidate Count
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

// 9. Duplicate Candidate IDs Handling
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

// 10. Forced Scalar Fallback & AVX2 Equivalence
static int test_forced_scalar_fallback_and_equivalence() {
    const uint32_t dim = 128;
    const uint32_t n = 500;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < corpus.size(); ++i) {
        corpus[i] = static_cast<float>((i * 17 + 3) % 1009) * 0.005f;
    }
    alg_set_vectors(index, n, corpus.data());

    std::vector<float> q(dim);
    for (size_t d = 0; d < dim; ++d) q[d] = corpus[77 * dim + d];

    // 1. Run with standard detected ISA (AVX2 on supported machines)
    alg_debug_force_scalar_isa(0);
    AlgSearchResult res_avx2[10];
    uint32_t count_avx2 = 0;
    AlgStatus st1 = alg_search(index, q.data(), 10, ALG_SEARCH_FAST_FLOAT32, res_avx2, &count_avx2);
    TEST_ASSERT(st1 == ALG_OK, "AVX2 search failed");

    // 2. Force scalar fallback
    alg_debug_force_scalar_isa(1);
    TEST_ASSERT(std::string(alg_get_active_isa_name()).find("Scalar") != std::string::npos,
                "Forced scalar must report Scalar Fallback ISA");

    AlgSearchResult res_scalar[10];
    uint32_t count_scalar = 0;
    AlgStatus st2 = alg_search(index, q.data(), 10, ALG_SEARCH_FAST_FLOAT32, res_scalar, &count_scalar);
    TEST_ASSERT(st2 == ALG_OK, "Scalar search failed");
    TEST_ASSERT(count_avx2 == count_scalar, "Result counts must match");

    // 3. Compare top-1 self result and overall neighbor set
    TEST_ASSERT(res_avx2[0].id == 77 && res_scalar[0].id == 77, "Rank 1 must be target 77 in both");
    for (size_t i = 0; i < count_avx2; ++i) {
        float diff = std::fabs(res_avx2[i].distance - res_scalar[i].distance);
        TEST_ASSERT(diff < 1e-4f, "Distances between AVX2 and Scalar must agree within float32 tolerance");
    }

    // Restore auto detection
    alg_debug_force_scalar_isa(0);

    alg_free_index(index);
    return 0;
}

// 11. Multithreaded Concurrent Searches
static int test_multithreaded_concurrent_searches() {
    const uint32_t dim = 64;
    const uint32_t n = 1000;
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

    const int NUM_THREADS = 8;
    const int QUERIES_PER_THREAD = 200;
    std::vector<std::thread> threads;
    std::vector<int> thread_errors(NUM_THREADS, 0);

    for (int t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            for (int q = 0; q < QUERIES_PER_THREAD; ++q) {
                uint32_t target_id = (t * 31 + q * 7) % n;
                const float* q_ptr = &corpus[target_id * dim];
                AlgSearchResult res[5];
                uint32_t out_count = 0;

                AlgStatus st = alg_search(index, q_ptr, 5, ALG_SEARCH_FAST_FLOAT32, res, &out_count);
                if (st != ALG_OK || out_count != 5 || res[0].id != target_id || std::fabs(res[0].distance) > 1e-6f) {
                    thread_errors[t]++;
                }
            }
        });
    }

    for (auto& th : threads) {
        th.join();
    }

    int total_err = 0;
    for (int err : thread_errors) total_err += err;
    TEST_ASSERT(total_err == 0, "Concurrent searches suffered data race or error");

    alg_free_index(index);
    return 0;
}

// 12. Non-Multiple-of-8 Dimensions
static int test_non_multiple_of_8_dimension() {
    const uint32_t dim = 13;
    const uint32_t n = 50;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            corpus[i * dim + d] = static_cast<float>(i * 3 + d) * 0.1f;
        }
    }
    alg_set_vectors(index, n, corpus.data());

    const float* query = &corpus[25 * dim];
    AlgSearchResult results[5];
    uint32_t count = 0;
    AlgStatus st = alg_search(index, query, 5, ALG_SEARCH_FAST_FLOAT32, results, &count);
    TEST_ASSERT(st == ALG_OK, "Search failed for dim 13");
    TEST_ASSERT(count == 5, "Expected 5 results");
    TEST_ASSERT(results[0].id == 25, "Rank 1 must be self target 25");
    TEST_ASSERT(std::fabs(results[0].distance) < 1e-6f, "Distance must be 0");

    alg_free_index(index);
    return 0;
}

// 13. Deterministic Tie-Breaking
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
    std::cout << "ALG Vector Engine - Hardened Unit & Regression Test Suite\n";
    std::cout << "Version: " << alg_get_version() << "\n";
    std::cout << "Active ISA: " << alg_get_active_isa_name() << "\n";
    std::cout << "Supported CPU Features bitmask: 0x" << std::hex << alg_get_supported_isa() << std::dec << "\n";
    std::cout << "============================================================\n\n";

    int failed_tests = 0;

    RUN_TEST(test_invalid_arguments_and_out_count);
    RUN_TEST(test_search_mode_contract);
    RUN_TEST(test_finiteness_validation_nan_inf);
    RUN_TEST(test_set_vectors_atomicity_on_failure);
    RUN_TEST(test_size_and_allocation_overflow);
    RUN_TEST(test_compact_search_strict_ids_and_out_of_bounds);
    RUN_TEST(test_topk_greater_than_512);
    RUN_TEST(test_topk_larger_than_candidate_count);
    RUN_TEST(test_duplicate_candidate_ids);
    RUN_TEST(test_forced_scalar_fallback_and_equivalence);
    RUN_TEST(test_multithreaded_concurrent_searches);
    RUN_TEST(test_non_multiple_of_8_dimension);
    RUN_TEST(test_deterministic_tie_breaking);

    std::cout << "\n============================================================\n";
    if (failed_tests == 0) {
        std::cout << "ALL 13 TESTS PASSED CLEANLY (100% SUCCESS)\n";
        std::cout << "============================================================\n";
        return 0;
    } else {
        std::cerr << failed_tests << " TESTS FAILED\n";
        std::cout << "============================================================\n";
        return 1;
    }
}
