#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "alg_engine.h"

int main(void) {
    printf("============================================================\n");
    printf("ALG Vector Engine - C ABI Example Consumer\n");
    printf("Library Version: %s\n", alg_get_version());
    printf("Active Instruction Set: %s\n", alg_get_active_isa_name());
    printf("============================================================\n\n");

    const uint32_t DIM = 64;
    const uint32_t NUM_VECTORS = 1000;
    const uint32_t TOP_K = 5;

    // 1. Configure and Create Index
    AlgIndexConfig config;
    config.dimension = DIM;
    config.initial_capacity = NUM_VECTORS;
    config.metric = ALG_METRIC_L2_SQR;

    AlgIndex* index = NULL;
    AlgStatus status = alg_create_index(&config, &index);
    if (status != ALG_OK) {
        fprintf(stderr, "Error creating index: %s\n", alg_status_to_string(status));
        return 1;
    }
    printf("[1] Index created successfully for dimension %u.\n", DIM);

    // 2. Generate synthetic corpus vectors
    float* corpus = (float*)malloc((size_t)NUM_VECTORS * DIM * sizeof(float));
    if (!corpus) {
        fprintf(stderr, "Out of memory allocating corpus.\n");
        alg_free_index(index);
        return 1;
    }

    for (uint32_t i = 0; i < NUM_VECTORS; ++i) {
        for (uint32_t d = 0; d < DIM; ++d) {
            corpus[i * DIM + d] = (float)((i + d) % 97) * 0.01f;
        }
    }

    status = alg_set_vectors(index, NUM_VECTORS, corpus);
    if (status != ALG_OK) {
        fprintf(stderr, "Error setting vectors: %s\n", alg_status_to_string(status));
        free(corpus);
        alg_free_index(index);
        return 1;
    }
    printf("[2] Ingested %u vectors into contiguous aligned engine memory.\n", NUM_VECTORS);

    // 3. Define Query Vector (identical to vector 42 for verification)
    float query[64];
    for (uint32_t d = 0; d < DIM; ++d) {
        query[d] = corpus[42 * DIM + d];
    }

    // 4. Perform Full-Corpus Unfiltered Search
    AlgSearchResult results[10];
    uint32_t result_count = 0;
    status = alg_search(index, query, TOP_K, ALG_SEARCH_FAST_FLOAT32, results, &result_count);
    if (status != ALG_OK) {
        fprintf(stderr, "Error searching: %s\n", alg_status_to_string(status));
        free(corpus);
        alg_free_index(index);
        return 1;
    }

    printf("\n[3] Full Search (Top-%u):\n", result_count);
    for (uint32_t j = 0; j < result_count; ++j) {
        printf("    Rank %u: Vector ID = %4u, L2 Squared Distance = %.6f\n",
               j + 1, results[j].id, results[j].distance);
    }

    // 5. Perform Filtered Search over specific candidate subset
    // E.g., only even IDs: 0, 2, 4, ..., 998
    uint32_t eligible_count = NUM_VECTORS / 2;
    uint32_t* eligible_ids = (uint32_t*)malloc(eligible_count * sizeof(uint32_t));
    for (uint32_t i = 0; i < eligible_count; ++i) {
        eligible_ids[i] = i * 2;
    }

    status = alg_search_compact(index, query, eligible_ids, eligible_count, TOP_K,
                                ALG_SEARCH_FAST_FLOAT32, results, &result_count);
    if (status != ALG_OK) {
        fprintf(stderr, "Error searching compact: %s\n", alg_status_to_string(status));
    } else {
        printf("\n[4] Filtered Search (Selectivity 50%%, Even IDs only, Top-%u):\n", result_count);
        for (uint32_t j = 0; j < result_count; ++j) {
            printf("    Rank %u: Vector ID = %4u, L2 Squared Distance = %.6f\n",
                   j + 1, results[j].id, results[j].distance);
        }
    }

    // 6. Clean up resources
    free(eligible_ids);
    free(corpus);
    alg_free_index(index);
    printf("\n[5] Cleaned up resources successfully.\n");
    return 0;
}
