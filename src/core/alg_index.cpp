#include "alg_index.h"
#include "../platform/cpu_features.h"
#include <stdlib.h>
#include <string.h>

#define ALG_VERSION_STR "1.0.0-rc1"

AlgStatus alg_create_index(const AlgIndexConfig* config, AlgIndex** out_index) {
    if (!config || !out_index) {
        return ALG_INVALID_ARGUMENT;
    }
    if (config->dimension == 0) {
        return ALG_INVALID_ARGUMENT;
    }
    if (config->metric != ALG_METRIC_L2_SQR) {
        return ALG_INVALID_ARGUMENT;
    }

    AlgIndex* idx = (AlgIndex*)malloc(sizeof(AlgIndex));
    if (!idx) {
        return ALG_OUT_OF_MEMORY;
    }

    idx->config = *config;
    idx->num_vectors = 0;
    idx->capacity = config->initial_capacity;
    idx->vector_data = NULL;
    idx->dispatch = alg_get_kernel_dispatch();

    if (idx->capacity > 0) {
        size_t total_bytes = (size_t)idx->capacity * idx->config.dimension * sizeof(float);
        idx->vector_data = (float*)alg_aligned_alloc_64(total_bytes);
        if (!idx->vector_data) {
            free(idx);
            return ALG_OUT_OF_MEMORY;
        }
    }

    *out_index = idx;
    return ALG_OK;
}

void alg_free_index(AlgIndex* index) {
    if (!index) return;
    if (index->vector_data) {
        alg_aligned_free_64(index->vector_data);
        index->vector_data = NULL;
    }
    free(index);
}

AlgStatus alg_set_vectors(AlgIndex* index, uint32_t num_vectors, const float* vectors) {
    if (!index) return ALG_NOT_INITIALIZED;
    if (num_vectors == 0 || !vectors) return ALG_INVALID_ARGUMENT;

    size_t dim = index->config.dimension;
    size_t total_bytes = (size_t)num_vectors * dim * sizeof(float);

    if (num_vectors > index->capacity || !index->vector_data) {
        if (index->vector_data) {
            alg_aligned_free_64(index->vector_data);
            index->vector_data = NULL;
        }
        index->vector_data = (float*)alg_aligned_alloc_64(total_bytes);
        if (!index->vector_data) {
            index->capacity = 0;
            index->num_vectors = 0;
            return ALG_OUT_OF_MEMORY;
        }
        index->capacity = num_vectors;
    }

    memcpy(index->vector_data, vectors, total_bytes);
    index->num_vectors = num_vectors;
    return ALG_OK;
}

float alg_distance_l2_sqr(const float* a, const float* b, uint32_t dimension) {
    if (!a || !b || dimension == 0) return 0.0f;
    AlgKernelDispatchTable dispatch = alg_get_kernel_dispatch();
    return dispatch.single(a, b, dimension);
}

const char* alg_status_to_string(AlgStatus status) {
    switch (status) {
        case ALG_OK:                return "ALG_OK: Operation completed successfully";
        case ALG_INVALID_ARGUMENT:  return "ALG_INVALID_ARGUMENT: Invalid argument supplied";
        case ALG_UNSUPPORTED_ISA:   return "ALG_UNSUPPORTED_ISA: Required CPU instruction set is unsupported";
        case ALG_OUT_OF_MEMORY:     return "ALG_OUT_OF_MEMORY: Failed to allocate required memory";
        case ALG_NOT_INITIALIZED:   return "ALG_NOT_INITIALIZED: Index or engine component is not initialized";
        case ALG_CAPACITY_EXCEEDED: return "ALG_CAPACITY_EXCEEDED: Requested vector count exceeds index capacity";
        default:                    return "ALG_UNKNOWN_ERROR";
    }
}

uint32_t alg_get_supported_isa(void) {
    return alg_platform_detect_cpu_features();
}

const char* alg_get_active_isa_name(void) {
    return alg_platform_get_isa_string();
}

const char* alg_get_version(void) {
    return ALG_VERSION_STR;
}
