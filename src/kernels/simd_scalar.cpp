/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "kernels.h"
#include "../platform/cpu_features.h"

void alg_kernel_l2_sqr_scalar_4x(
    const float* q,
    const float* v0,
    const float* v1,
    const float* v2,
    const float* v3,
    uint32_t dim,
    float* out_dists_4
) {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        float qv = q[d];
        float d0 = qv - v0[d];
        float d1 = qv - v1[d];
        float d2 = qv - v2[d];
        float d3 = qv - v3[d];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
    }
    out_dists_4[0] = s0;
    out_dists_4[1] = s1;
    out_dists_4[2] = s2;
    out_dists_4[3] = s3;
}

float alg_kernel_l2_sqr_scalar_1x(
    const float* q,
    const float* v,
    uint32_t dim
) {
    float sum = 0.0f;
    for (uint32_t d = 0; d < dim; ++d) {
        float diff = q[d] - v[d];
        sum += diff * diff;
    }
    return sum;
}

AlgKernelDispatchTable alg_get_kernel_dispatch(void) {
    AlgKernelDispatchTable table;
    if (alg_platform_has_avx2()) {
        table.batch4 = alg_kernel_l2_sqr_avx2_4x;
        table.single = alg_kernel_l2_sqr_avx2_1x;
        table.name = "AVX2+FMA (Vectorized 4x-unrolled)";
    } else {
        table.batch4 = alg_kernel_l2_sqr_scalar_4x;
        table.single = alg_kernel_l2_sqr_scalar_1x;
        table.name = "Scalar Fallback";
    }
    return table;
}
