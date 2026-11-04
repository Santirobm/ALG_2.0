#include "alg_index.h"
#include "../platform/cpu_features.h"
#include "../platform/allocator.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define ALG_VERSION_STR "1.0.0-rc1"

static inline bool alg_is_finite_buffer(const float* data, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        uint32_t u;
        memcpy(&u, &data[i], sizeof(float));
        // In IEEE 754 float32, exponent mask is 0x7F800000.
        // If exponent bits are all 1s, the number is either NaN or +/-Infinity.
        if ((u & 0x7F800000u) == 0x7F800000u) {
            return false;
        }
    }
    return true;
}

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

    size_t total_bytes = 0;
    if (config->initial_capacity > 0) {
        if (alg_mul_overflow_3((size_t)config->initial_capacity, (size_t)config->dimension, sizeof(float), &total_bytes)) {
            return ALG_INVALID_ARGUMENT;
        }
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

    if (total_bytes > 0) {
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
    size_t total_floats = 0;
    if (alg_mul_overflow_2((size_t)num_vectors, dim, &total_floats)) {
        return ALG_INVALID_ARGUMENT;
    }

    // Numerical Finiteness Validation (Reject NaN, +Inf, -Inf)
    // Checked before any memory allocation or index modification to preserve atomicity
    if (!alg_is_finite_buffer(vectors, total_floats)) {
        return ALG_INVALID_ARGUMENT;
    }

    size_t total_bytes = 0;
    if (alg_mul_overflow_2(total_floats, sizeof(float), &total_bytes)) {
        return ALG_INVALID_ARGUMENT;
    }

    // Atomicity: If reallocation is needed, allocate new buffer first.
    // If allocation fails, existing corpus and capacity remain untouched.
    if (num_vectors > index->capacity || !index->vector_data) {
        float* new_buffer = (float*)alg_aligned_alloc_64(total_bytes);
        if (!new_buffer) {
            return ALG_OUT_OF_MEMORY;
        }
        if (index->vector_data) {
            alg_aligned_free_64(index->vector_data);
        }
        index->vector_data = new_buffer;
        index->capacity = num_vectors;
    }

    memcpy(index->vector_data, vectors, total_bytes);
    index->num_vectors = num_vectors;
    // Always refresh kernel dispatch table
    index->dispatch = alg_get_kernel_dispatch();

    return ALG_OK;
}

float alg_distance_l2_sqr(const float* a, const float* b, uint32_t dimension) {
    if (!a || !b || dimension == 0) return NAN;
    if (!alg_is_finite_buffer(a, dimension) || !alg_is_finite_buffer(b, dimension)) {
        return NAN;
    }
    AlgKernelDispatchTable dispatch = alg_get_kernel_dispatch();
    return dispatch.single(a, b, dimension);
}

const char* alg_status_to_string(AlgStatus status) {
    switch (status) {
        case ALG_OK:                return "ALG_OK: Operation completed successfully";
        case ALG_INVALID_ARGUMENT:  return "ALG_INVALID_ARGUMENT: Invalid argument supplied";
        case ALG_UNSUPPORTED_ISA:   return "ALG_UNSUPPORTED_ISA: Required CPU instruction set is unsupported";
        case ALG_OUT_OF_MEMORY:     return "ALG_OUT_OF_MEMORY: Failed to allocate required memory or size overflow";
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
