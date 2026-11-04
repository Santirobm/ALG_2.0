/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef ALG_CORE_ALG_INDEX_H
#define ALG_CORE_ALG_INDEX_H

#include "../../include/alg_engine.h"
#include "../kernels/kernels.h"
#include "../platform/allocator.h"

#ifdef __cplusplus
extern "C" {
#endif

struct AlgIndex {
    AlgIndexConfig config;
    uint32_t num_vectors;
    uint32_t capacity;
    float* vector_data;              // 64-byte aligned contiguous AoS buffer
    AlgKernelDispatchTable dispatch; // Resolved SIMD/Scalar function pointers
};

#ifdef __cplusplus
}
#endif

#endif /* ALG_CORE_ALG_INDEX_H */
