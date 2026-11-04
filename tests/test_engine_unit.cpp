#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <cstring>
#include "alg_engine.h"

#define TEST_ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "[-] Assertion FAILED: " << msg << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
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

static int test_invalid_arguments() {
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
    uint32_t count = 0;
    status = alg_search(nullptr, q, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(status == ALG_NOT_INITIALIZED, "NULL index search must return ALG_NOT_INITIALIZED");

    return 0;
}

static int test_search_mode_contract() {
    AlgIndexConfig cfg = {64, 10, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    AlgStatus status = alg_create_index(&cfg, &index);
    TEST_ASSERT(status == ALG_OK, "Create index failed");

    float corpus[64 * 10] = {0};
    alg_set_vectors(index, 10, corpus);

    float q[64] = {0};
    AlgSearchResult res[5];
    uint32_t count = 0;

    // Mode ALG_SEARCH_EXACT_TOPK is reserved for future refinement pass
    status = alg_search(index, q, 5, ALG_SEARCH_EXACT_TOPK, res, &count);
    TEST_ASSERT(status == ALG_INVALID_ARGUMENT, "ALG_SEARCH_EXACT_TOPK must return ALG_INVALID_ARGUMENT in current version");

    // Fast float32 mode must succeed
    status = alg_search(index, q, 5, ALG_SEARCH_FAST_FLOAT32, res, &count);
    TEST_ASSERT(status == ALG_OK, "ALG_SEARCH_FAST_FLOAT32 must succeed");

    alg_free_index(index);
    return 0;
}

static int test_exact_self_search() {
    const uint32_t dim = 64;
    const uint32_t n = 200;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            corpus[i * dim + d] = static_cast<float>((i + 1) * (d + 1)) * 0.005f;
        }
    }
    alg_set_vectors(index, n, corpus.data());

    // Query for each vector i in the corpus; rank 1 must be i with distance 0.0
    for (uint32_t target_id : {0u, 17u, 42u, 99u, 199u}) {
        const float* query = &corpus[target_id * dim];
        AlgSearchResult results[10];
        uint32_t out_count = 0;
        AlgStatus st = alg_search(index, query, 5, ALG_SEARCH_FAST_FLOAT32, results, &out_count);
        TEST_ASSERT(st == ALG_OK, "Search failed");
        TEST_ASSERT(out_count == 5, "Expected 5 results");
        TEST_ASSERT(results[0].id == target_id, "Rank 1 ID must match target_id");
        TEST_ASSERT(std::fabs(results[0].distance) < 1e-6f, "Self distance must be 0");
    }

    alg_free_index(index);
    return 0;
}

static int test_deterministic_tie_breaking() {
    // Construct corpus with identical vectors but distinct IDs
    const uint32_t dim = 64;
    const uint32_t n = 20;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    // All vectors have exact same values (1.0f)
    std::vector<float> corpus(n * dim, 1.0f);
    alg_set_vectors(index, n, corpus.data());

    // Query with identical vector -> all distances are exactly 0.0f
    std::vector<float> query(dim, 1.0f);
    AlgSearchResult results[10];
    uint32_t out_count = 0;

    AlgStatus st = alg_search(index, query.data(), 10, ALG_SEARCH_FAST_FLOAT32, results, &out_count);
    TEST_ASSERT(st == ALG_OK, "Search failed");
    TEST_ASSERT(out_count == 10, "Expected 10 results");

    // Must be sorted strictly by ID ascending: 0, 1, 2, ..., 9
    for (uint32_t r = 0; r < out_count; ++r) {
        TEST_ASSERT(results[r].id == r, "Tie break must order by ID ascending");
        TEST_ASSERT(results[r].distance == 0.0f, "Distance must be exactly 0.0f");
    }

    alg_free_index(index);
    return 0;
}

static int test_filtered_candidate_search() {
    const uint32_t dim = 64;
    const uint32_t n = 500;
    AlgIndexConfig cfg = {dim, n, ALG_METRIC_L2_SQR};
    AlgIndex* index = nullptr;
    alg_create_index(&cfg, &index);

    std::vector<float> corpus(n * dim);
    for (size_t i = 0; i < n; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            corpus[i * dim + d] = static_cast<float>(i * dim + d) * 0.001f;
        }
    }
    alg_set_vectors(index, n, corpus.data());

    // Filter candidate list: only IDs in [200, 250]
    std::vector<uint32_t> eligible;
    for (uint32_t id = 200; id <= 250; ++id) {
        eligible.push_back(id);
    }

    // Query is target ID 100 (which is NOT in eligible list)
    const float* query = &corpus[100 * dim];
    AlgSearchResult results[10];
    uint32_t count = 0;

    AlgStatus st = alg_search_compact(index, query, eligible.data(), (uint32_t)eligible.size(),
                                      10, ALG_SEARCH_FAST_FLOAT32, results, &count);
    TEST_ASSERT(st == ALG_OK, "Search compact failed");
    TEST_ASSERT(count == 10, "Expected 10 results");

    // Verify all returned IDs belong to eligible set
    for (uint32_t j = 0; j < count; ++j) {
        TEST_ASSERT(results[j].id >= 200 && results[j].id <= 250, "Returned ID outside candidate filter");
    }

    // Nearest eligible vector to ID 100 in linear distribution is ID 200
    TEST_ASSERT(results[0].id == 200, "Closest eligible vector must be 200");

    alg_free_index(index);
    return 0;
}

static int test_non_multiple_of_8_dimension() {
    // Dimension 13 (tests scalar tail handling)
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

int main() {
    std::cout << "============================================================\n";
    std::cout << "ALG Vector Engine - Automated Unit & Regression Test Suite\n";
    std::cout << "Version: " << alg_get_version() << "\n";
    std::cout << "Active ISA: " << alg_get_active_isa_name() << "\n";
    std::cout << "Supported CPU Features bitmask: 0x" << std::hex << alg_get_supported_isa() << std::dec << "\n";
    std::cout << "============================================================\n\n";

    int failed_tests = 0;

    RUN_TEST(test_invalid_arguments);
    RUN_TEST(test_search_mode_contract);
    RUN_TEST(test_exact_self_search);
    RUN_TEST(test_deterministic_tie_breaking);
    RUN_TEST(test_filtered_candidate_search);
    RUN_TEST(test_non_multiple_of_8_dimension);

    std::cout << "\n============================================================\n";
    if (failed_tests == 0) {
        std::cout << "ALL 6 TESTS PASSED CLEANLY (100% SUCCESS)\n";
        std::cout << "============================================================\n";
        return 0;
    } else {
        std::cerr << failed_tests << " TESTS FAILED\n";
        std::cout << "============================================================\n";
        return 1;
    }
}
