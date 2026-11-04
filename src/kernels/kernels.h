#ifndef ALG_KERNELS_H
#define ALG_KERNELS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*AlgBatch4KernelFn)(
    const float* q,
    const float* v0,
    const float* v1,
    const float* v2,
    const float* v3,
    uint32_t dim,
    float* out_dists_4
);

typedef float (*AlgSingleKernelFn)(
    const float* q,
    const float* v,
    uint32_t dim
);

typedef struct AlgKernelDispatchTable {
    AlgBatch4KernelFn batch4;
    AlgSingleKernelFn single;
    const char* name;
} AlgKernelDispatchTable;

// AVX2+FMA Kernels (implemented in simd_avx2.cpp)
void alg_kernel_l2_sqr_avx2_4x(
    const float* q,
    const float* v0,
    const float* v1,
    const float* v2,
    const float* v3,
    uint32_t dim,
    float* out_dists_4
);

float alg_kernel_l2_sqr_avx2_1x(
    const float* q,
    const float* v,
    uint32_t dim
);

// Scalar Fallback Kernels (implemented in simd_scalar.cpp)
void alg_kernel_l2_sqr_scalar_4x(
    const float* q,
    const float* v0,
    const float* v1,
    const float* v2,
    const float* v3,
    uint32_t dim,
    float* out_dists_4
);

float alg_kernel_l2_sqr_scalar_1x(
    const float* q,
    const float* v,
    uint32_t dim
);

// Dispatch table resolver
AlgKernelDispatchTable alg_get_kernel_dispatch(void);

#ifdef __cplusplus
}
#endif

#endif /* ALG_KERNELS_H */
