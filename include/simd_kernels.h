#ifndef ALG_FLAT_SIMD_KERNELS_H
#define ALG_FLAT_SIMD_KERNELS_H

#include "alg_flat_types.h"
#include <immintrin.h>

namespace alg_flat {

/*
 * 1. Kernel Escalar Estricto (con vectorizacion automatica deshabilitada
 *    para medir el coste escalar real sin que LLVM inyecte AVX2 en secreto).
 */
#if defined(__clang__)
__attribute__((noinline))
#endif
inline float l2_sq_scalar_strict(const float* a, const float* b, uint32_t dim) {
    float sum0 = 0.0f, sum1 = 0.0f;
#if defined(__clang__)
    #pragma clang loop vectorize(disable) interleave(disable)
#endif
    for (uint32_t d = 0; d < dim; d += 2) {
        float diff0 = a[d] - b[d];
        float diff1 = a[d + 1] - b[d + 1];
        sum0 += diff0 * diff0;
        sum1 += diff1 * diff1;
        asm volatile("" : "+x"(sum0), "+x"(sum1));
    }
    return sum0 + sum1;
}

/*
 * Reduccion horizontal rapida de __m256 sin usar _mm_hadd_ps (evita presion en puerto 5)
 */
static inline float hsum256_ps_fast(__m256 v) {
    __m128 vlow  = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    vlow = _mm_add_ps(vlow, vhigh);
    __m128 shuf = _mm_movehdup_ps(vlow);
    __m128 sums = _mm_add_ps(vlow, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

/*
 * 2. Kernel AVX2 AoS (1x1) con doble acumulador para romper dependencia de latencia FMA (4 ciclos)
 */
static inline float l2_sq_avx2_1x_aos(const float* q, const float* v, uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    for (uint32_t d = 0; d < dim; d += 16) {
        __m256 q0 = _mm256_loadu_ps(q + d);
        __m256 v0 = _mm256_loadu_ps(v + d);
        __m256 diff0 = _mm256_sub_ps(q0, v0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 q1 = _mm256_loadu_ps(q + d + 8);
        __m256 v1 = _mm256_loadu_ps(v + d + 8);
        __m256 diff1 = _mm256_sub_ps(q1, v1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }
    return hsum256_ps_fast(_mm256_add_ps(acc0, acc1));
}

/*
 * 3. Kernel AVX2 AoS (4x1 Multi-Vector):
 *    Evalua 1 consulta contra 4 vectores AoS simultaneamente, reutilizando los registros de `q`
 *    y saturando el throughput de 2 FMA/ciclo con 4 acumuladores independientes.
 *    Realiza la reduccion horizontal 4x8 -> 1x4 de forma vectorizada mediante transposicion.
 */
static inline __m128 l2_sq_avx2_4x_aos(const float* q,
                                       const float* v0,
                                       const float* v1,
                                       const float* v2,
                                       const float* v3,
                                       uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (uint32_t d = 0; d < dim; d += 8) {
        __m256 qv = _mm256_loadu_ps(q + d);

        __m256 d0 = _mm256_sub_ps(qv, _mm256_loadu_ps(v0 + d));
        acc0 = _mm256_fmadd_ps(d0, d0, acc0);

        __m256 d1 = _mm256_sub_ps(qv, _mm256_loadu_ps(v1 + d));
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);

        __m256 d2 = _mm256_sub_ps(qv, _mm256_loadu_ps(v2 + d));
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);

        __m256 d3 = _mm256_sub_ps(qv, _mm256_loadu_ps(v3 + d));
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }

    /* Reduccion simultanea de 4 registros __m256 a un unico __m128 con [dist0, dist1, dist2, dist3] */
    __m256 t0 = _mm256_hadd_ps(acc0, acc1);
    __m256 t1 = _mm256_hadd_ps(acc2, acc3);
    __m256 t2 = _mm256_hadd_ps(t0, t1);
    __m128 lo = _mm256_castps256_ps128(t2);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    return _mm_add_ps(lo, hi);
}

/*
 * 4. Kernel AVX2 AoSoA (Bloque transpuesto 8 vectores x 64 dimensiones = 2048 bytes contiguos):
 *    Calcula 8 distancias en paralelo sin NINGUNA reduccion horizontal.
 *    Layout de `block_ptr`: para cada dimension d in [0..63], 8 floats [v0_d .. v7_d].
 */
static inline __m256 l2_sq_avx2_8x_aosoa(const float* q, const float* block_ptr, uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (uint32_t d = 0; d < dim; d += 4) {
        __m256 q0 = _mm256_broadcast_ss(q + d);
        __m256 v0 = _mm256_load_ps(block_ptr + (d + 0) * 8);
        __m256 diff0 = _mm256_sub_ps(q0, v0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 q1 = _mm256_broadcast_ss(q + d + 1);
        __m256 v1 = _mm256_load_ps(block_ptr + (d + 1) * 8);
        __m256 diff1 = _mm256_sub_ps(q1, v1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        __m256 q2 = _mm256_broadcast_ss(q + d + 2);
        __m256 v2 = _mm256_load_ps(block_ptr + (d + 2) * 8);
        __m256 diff2 = _mm256_sub_ps(q2, v2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        __m256 q3 = _mm256_broadcast_ss(q + d + 3);
        __m256 v3 = _mm256_load_ps(block_ptr + (d + 3) * 8);
        __m256 diff3 = _mm256_sub_ps(q3, v3);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
    }

    return _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
}

/*
 * 5. Kernel AVX2 SoA (Structure of Arrays global: 64 flujos de `stride` floats):
 *    Evalua 8 vectores contiguos en la posicion `vec_idx` a traves de las 64 bandas dimensionales.
 */
static inline __m256 l2_sq_avx2_8x_soa(const float* q,
                                       const float* soa_data,
                                       size_t stride,
                                       size_t vec_idx,
                                       uint32_t dim) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const float* base = soa_data + vec_idx;
    for (uint32_t d = 0; d < dim; d += 4) {
        __m256 q0 = _mm256_broadcast_ss(q + d);
        __m256 v0 = _mm256_loadu_ps(base + (d + 0) * stride);
        __m256 diff0 = _mm256_sub_ps(q0, v0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 q1 = _mm256_broadcast_ss(q + d + 1);
        __m256 v1 = _mm256_loadu_ps(base + (d + 1) * stride);
        __m256 diff1 = _mm256_sub_ps(q1, v1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);

        __m256 q2 = _mm256_broadcast_ss(q + d + 2);
        __m256 v2 = _mm256_loadu_ps(base + (d + 2) * stride);
        __m256 diff2 = _mm256_sub_ps(q2, v2);
        acc2 = _mm256_fmadd_ps(diff2, diff2, acc2);

        __m256 q3 = _mm256_broadcast_ss(q + d + 3);
        __m256 v3 = _mm256_loadu_ps(base + (d + 3) * stride);
        __m256 diff3 = _mm256_sub_ps(q3, v3);
        acc3 = _mm256_fmadd_ps(diff3, diff3, acc3);
    }

    return _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
}

} // namespace alg_flat

#endif // ALG_FLAT_SIMD_KERNELS_H
