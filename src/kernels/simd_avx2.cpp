#include "kernels.h"
#include <immintrin.h>

// Vectorized horizontal reduction of 4 independent __m256 accumulators
// into a single __m128 register containing [sum0, sum1, sum2, sum3].
static inline __m128 reduce_4x256_to_128(__m256 acc0, __m256 acc1, __m256 acc2, __m256 acc3) {
    __m256 t0 = _mm256_hadd_ps(acc0, acc1);
    __m256 t1 = _mm256_hadd_ps(acc2, acc3);
    __m256 t2 = _mm256_hadd_ps(t0, t1);
    __m128 lo = _mm256_castps256_ps128(t2);
    __m128 hi = _mm256_extractf128_ps(t2, 1);
    return _mm_add_ps(lo, hi);
}

// Fast horizontal sum of a single __m256 without _mm_hadd_ps
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

void alg_kernel_l2_sqr_avx2_4x(
    const float* q,
    const float* v0,
    const float* v1,
    const float* v2,
    const float* v3,
    uint32_t dim,
    float* out_dists_4
) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    uint32_t d = 0;
    for (; d + 8 <= dim; d += 8) {
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

    __m128 dists = reduce_4x256_to_128(acc0, acc1, acc2, acc3);
    _mm_storeu_ps(out_dists_4, dists);

    // Handle scalar tail if dim is not a multiple of 8
    for (; d < dim; ++d) {
        float qv = q[d];
        float df0 = qv - v0[d];
        float df1 = qv - v1[d];
        float df2 = qv - v2[d];
        float df3 = qv - v3[d];
        out_dists_4[0] += df0 * df0;
        out_dists_4[1] += df1 * df1;
        out_dists_4[2] += df2 * df2;
        out_dists_4[3] += df3 * df3;
    }
}

float alg_kernel_l2_sqr_avx2_1x(
    const float* q,
    const float* v,
    uint32_t dim
) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();

    uint32_t d = 0;
    for (; d + 16 <= dim; d += 16) {
        __m256 q0 = _mm256_loadu_ps(q + d);
        __m256 v0 = _mm256_loadu_ps(v + d);
        __m256 diff0 = _mm256_sub_ps(q0, v0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);

        __m256 q1 = _mm256_loadu_ps(q + d + 8);
        __m256 v1 = _mm256_loadu_ps(v + d + 8);
        __m256 diff1 = _mm256_sub_ps(q1, v1);
        acc1 = _mm256_fmadd_ps(diff1, diff1, acc1);
    }

    for (; d + 8 <= dim; d += 8) {
        __m256 q0 = _mm256_loadu_ps(q + d);
        __m256 v0 = _mm256_loadu_ps(v + d);
        __m256 diff0 = _mm256_sub_ps(q0, v0);
        acc0 = _mm256_fmadd_ps(diff0, diff0, acc0);
    }

    float total = hsum256_ps_fast(_mm256_add_ps(acc0, acc1));

    // Handle tail
    for (; d < dim; ++d) {
        float diff = q[d] - v[d];
        total += diff * diff;
    }

    return total;
}
