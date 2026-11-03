#include "../include/execution_plans.h"
#include "../include/simd_kernels.h"
#include "../include/ieee754_bounds.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>

namespace alg_flat {

struct TimerScope {
    LARGE_INTEGER freq;
    LARGE_INTEGER t0, t1, t2;
    uint64_t c0, c1;

    inline void start() {
        QueryPerformanceFrequency(&freq);
        QueryPerformanceCounter(&t0);
        c0 = __rdtsc();
    }

    inline void mark_prep_done() {
        QueryPerformanceCounter(&t1);
    }

    inline void stop(QueryExecutionMetrics& m) {
        c1 = __rdtsc();
        QueryPerformanceCounter(&t2);
        double inv_f = 1e6 / static_cast<double>(freq.QuadPart);
        m.prep_us = static_cast<double>(t1.QuadPart - t0.QuadPart) * inv_f;
        m.calc_us = static_cast<double>(t2.QuadPart - t1.QuadPart) * inv_f;
        m.total_us = static_cast<double>(t2.QuadPart - t0.QuadPart) * inv_f;
        m.cpu_cycles = (c1 > c0) ? (c1 - c0) : 0;
    }
};

/*
 * Distancia exacta canonica en espacio original float32 (identica a Hito 8/9).
 * Todos los planes refinan los candidatos que ingresan al TopKHeap con esta funcion
 * en el espacio original para garantizar igualdad exacta de distancias y desempates.
 */
static inline float l2_hito9_ref(const float *a, const float *b, uint32_t dim) {
    __m256 acc = _mm256_setzero_ps();
    for (uint32_t i = 0; i < dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 d  = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(d, d, acc);
    }
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum4 = _mm_add_ps(lo, hi);
    sum4 = _mm_hadd_ps(sum4, sum4);
    sum4 = _mm_hadd_ps(sum4, sum4);
    return _mm_cvtss_f32(sum4);
}

struct RefDistPair {
    float dist;
    uint32_t id;
    bool operator<(const RefDistPair &other) const {
        if (dist != other.dist) return dist < other.dist;
        return id < other.id;
    }
};

void execute_plan0_ref_hito9(const FlatAoSIndex& idx,
                             const float* q,
                             const QueryPredicate& pred,
                             uint32_t k,
                             Neighbor* out_results,
                             uint32_t* out_count,
                             QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();
    ts.mark_prep_done();

    std::priority_queue<RefDistPair> pq;
    const uint32_t dim = idx.dim;
    const uint32_t n_el = pred.eligible_count;
    const uint32_t* el_ids = pred.eligible_ids;

    for (uint32_t i = 0; i < n_el; ++i) {
        uint32_t id = el_ids[i];
        float d = l2_hito9_ref(q, idx.vectors + static_cast<size_t>(id) * dim, dim);
        if (pq.size() < k) {
            pq.push({d, id});
        } else if (d < pq.top().dist || (d == pq.top().dist && id < pq.top().id)) {
            pq.pop();
            pq.push({d, id});
        }
    }

    uint32_t count = static_cast<uint32_t>(pq.size());
    *out_count = count;
    for (int i = static_cast<int>(count) - 1; i >= 0; --i) {
        out_results[i] = {pq.top().id, pq.top().dist};
        pq.pop();
    }

    ts.stop(metrics);
    metrics.dists_evaluated = n_el;
    metrics.eligible_total = n_el;
    metrics.partitions_visited = 1;
    metrics.partitions_total = 1;
    metrics.bytes_read = static_cast<uint64_t>(n_el) * (dim * sizeof(float) + sizeof(uint32_t));
}

void execute_plan1_flat_scalar_aos(const FlatAoSIndex& idx,
                                   const float* q,
                                   const QueryPredicate& pred,
                                   uint32_t k,
                                   Neighbor* out_results,
                                   uint32_t* out_count,
                                   QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();
    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t dim = idx.dim;
    const uint32_t n_el = pred.eligible_count;
    const uint32_t* el_ids = pred.eligible_ids;

    for (uint32_t i = 0; i < n_el; ++i) {
        uint32_t id = el_ids[i];
        const float* vp = idx.vectors + static_cast<size_t>(id) * dim;
        float d = l2_sq_scalar_strict(q, vp, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d <= gate) {
            float d_exact = l2_hito9_ref(q, vp, dim);
            heap.push(id, d_exact);
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = n_el;
    metrics.eligible_total = n_el;
    metrics.partitions_visited = 1;
    metrics.partitions_total = 1;
    metrics.bytes_read = static_cast<uint64_t>(n_el) * (dim * sizeof(float) + sizeof(uint32_t));
}

void execute_plan2a_flat_avx2_aos_1x(const FlatAoSIndex& idx,
                                     const float* q,
                                     const QueryPredicate& pred,
                                     uint32_t k,
                                     Neighbor* out_results,
                                     uint32_t* out_count,
                                     QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();
    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t dim = idx.dim;
    const uint32_t n_el = pred.eligible_count;
    const uint32_t* el_ids = pred.eligible_ids;

    for (uint32_t i = 0; i < n_el; ++i) {
        uint32_t id = el_ids[i];
        const float* vp = idx.vectors + static_cast<size_t>(id) * dim;
        float d = l2_sq_avx2_1x_aos(q, vp, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d <= gate) {
            float d_exact = l2_hito9_ref(q, vp, dim);
            heap.push(id, d_exact);
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = n_el;
    metrics.eligible_total = n_el;
    metrics.partitions_visited = 1;
    metrics.partitions_total = 1;
    metrics.bytes_read = static_cast<uint64_t>(n_el) * (dim * sizeof(float) + sizeof(uint32_t));
}

void execute_plan2b_flat_avx2_aos_4x(const FlatAoSIndex& idx,
                                     const float* q,
                                     const QueryPredicate& pred,
                                     uint32_t k,
                                     Neighbor* out_results,
                                     uint32_t* out_count,
                                     QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();
    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t dim = idx.dim;
    const uint32_t n_el = pred.eligible_count;
    const uint32_t* el_ids = pred.eligible_ids;
    const float* base_v = idx.vectors;

    uint32_t i = 0;
    for (; i + 4 <= n_el; i += 4) {
        uint32_t id0 = el_ids[i];
        uint32_t id1 = el_ids[i + 1];
        uint32_t id2 = el_ids[i + 2];
        uint32_t id3 = el_ids[i + 3];

        const float* v0 = base_v + static_cast<size_t>(id0) * dim;
        const float* v1 = base_v + static_cast<size_t>(id1) * dim;
        const float* v2 = base_v + static_cast<size_t>(id2) * dim;
        const float* v3 = base_v + static_cast<size_t>(id3) * dim;

        __m128 d4 = l2_sq_avx2_4x_aos(q, v0, v1, v2, v3, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        __m128 v_thresh = _mm_set1_ps(gate);
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, v_thresh));
        if (mask != 0) {
            if (mask & 1) heap.push(id0, l2_hito9_ref(q, v0, dim));
            if (mask & 2) heap.push(id1, l2_hito9_ref(q, v1, dim));
            if (mask & 4) heap.push(id2, l2_hito9_ref(q, v2, dim));
            if (mask & 8) heap.push(id3, l2_hito9_ref(q, v3, dim));
        }
    }

    for (; i < n_el; ++i) {
        uint32_t id = el_ids[i];
        const float* vp = base_v + static_cast<size_t>(id) * dim;
        float d = l2_sq_avx2_1x_aos(q, vp, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d <= gate) {
            heap.push(id, l2_hito9_ref(q, vp, dim));
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = n_el;
    metrics.eligible_total = n_el;
    metrics.partitions_visited = 1;
    metrics.partitions_total = 1;
    metrics.bytes_read = static_cast<uint64_t>(n_el) * (dim * sizeof(float) + sizeof(uint32_t));
}

void execute_plan3a_flat_avx2_soa(const FlatSoAIndex& idx,
                                  const float* q,
                                  const QueryPredicate& pred,
                                  uint32_t k,
                                  Neighbor* out_results,
                                  uint32_t* out_count,
                                  QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();
    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t dim = idx.dim;
    const uint32_t num_blocks = idx.num_blocks;
    const size_t stride = idx.stride;
    const uint32_t target_cat = pred.target_category;
    __m256i v_target = _mm256_set1_epi32(static_cast<int>(target_cat));

    uint32_t evaluated_blocks = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        size_t vec_idx = static_cast<size_t>(b) * 8U;
        __m256i v_cats = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx.categories + vec_idx));
        __m256i v_elig = _mm256_and_si256(_mm256_cmpgt_epi32(v_target, v_cats),
                                          _mm256_cmpgt_epi32(v_cats, _mm256_set1_epi32(-1)));

        int elig_mask = _mm256_movemask_ps(_mm256_castsi256_ps(v_elig));
        if (elig_mask == 0) continue;

        evaluated_blocks++;
        __m256 d8 = l2_sq_avx2_8x_soa(q, idx.soa_data, stride, vec_idx, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        __m256 v_thresh = _mm256_set1_ps(gate);
        int hit_mask = _mm256_movemask_ps(_mm256_cmp_ps(d8, v_thresh, _CMP_LE_OS)) & elig_mask;

        if (hit_mask != 0) {
            for (uint32_t lane = 0; lane < 8U; ++lane) {
                if (hit_mask & (1 << lane)) {
                    uint32_t orig_id = idx.ids[vec_idx + lane];
                    float d_exact = l2_hito9_ref(q, idx.orig_vectors + static_cast<size_t>(orig_id) * dim, dim);
                    heap.push(orig_id, d_exact);
                }
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = evaluated_blocks * 8U;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = 1;
    metrics.partitions_total = 1;
    metrics.bytes_read = static_cast<uint64_t>(evaluated_blocks) * 8U * dim * sizeof(float)
                       + static_cast<uint64_t>(num_blocks) * 8U * sizeof(uint32_t);
}

void execute_plan3b_flat_avx2_aosoa(const FlatAoSoAIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    Neighbor* out_results,
                                    uint32_t* out_count,
                                    QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();
    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t dim = idx.dim;
    const uint32_t num_blocks = idx.num_blocks;
    const uint32_t target_cat = pred.target_category;
    __m256i v_target = _mm256_set1_epi32(static_cast<int>(target_cat));

    uint32_t evaluated_blocks = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        size_t vec_idx = static_cast<size_t>(b) * 8U;
        __m256i v_cats = _mm256_load_si256(reinterpret_cast<const __m256i*>(idx.categories + vec_idx));
        __m256i v_elig = _mm256_and_si256(_mm256_cmpgt_epi32(v_target, v_cats),
                                          _mm256_cmpgt_epi32(v_cats, _mm256_set1_epi32(-1)));

        int elig_mask = _mm256_movemask_ps(_mm256_castsi256_ps(v_elig));
        if (elig_mask == 0) continue;

        evaluated_blocks++;
        const float* blk_ptr = idx.block_data + static_cast<size_t>(b) * dim * 8U;
        __m256 d8 = l2_sq_avx2_8x_aosoa(q, blk_ptr, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        __m256 v_thresh = _mm256_set1_ps(gate);
        int hit_mask = _mm256_movemask_ps(_mm256_cmp_ps(d8, v_thresh, _CMP_LE_OS)) & elig_mask;

        if (hit_mask != 0) {
            for (uint32_t lane = 0; lane < 8U; ++lane) {
                if (hit_mask & (1 << lane)) {
                    uint32_t orig_id = idx.ids[vec_idx + lane];
                    float d_exact = l2_hito9_ref(q, idx.orig_vectors + static_cast<size_t>(orig_id) * dim, dim);
                    heap.push(orig_id, d_exact);
                }
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = evaluated_blocks * 8U;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = 1;
    metrics.partitions_total = 1;
    metrics.bytes_read = static_cast<uint64_t>(evaluated_blocks) * 8U * dim * sizeof(float)
                       + static_cast<uint64_t>(num_blocks) * 8U * sizeof(uint32_t);
}

/*
 * Gestor formal de contrato IEEE-754 ante condiciones no nominales en tiempo de consulta:
 * Separa estrictamente las 3 situaciones identificadas en la auditoria de Hito 2A.3:
 *   1. Corpus / Indice invalido (`!idx.certified_Contract_valid || idx.num_vectors == 0`):
 *      Rechaza explicitamente la consulta (`query_rejected = 1`, `status = REJECTED_INVALID_CORPUS_INDEX`).
 *   2. Consulta no procesable con garantias (`NaN`, `Inf` o `|q_d| > CANONICAL_FALLBACK_MAX_COORD_ABS = 1e18`):
 *      Rechaza explicitamente la consulta (`query_rejected = 1`, `status = REJECTED_UNSAFE_QUERY_DOMAIN`)
 *      porque incluso `ExactScan` podria desbordar a `Inf`/`NaN` y romper el orden total del heap.
 *   3. MXCSR alterado (`FTZ=1`, `DAZ=1` o redondeo != Nearest-Even) y/o consulta finita fuera del
 *      dominio certificado de poda pero dentro del dominio seguro de fallback (`1e15 < |q_d| <= 1e18`):
 *      Desactiva toda poda geometrica, restaura temporalmente un entorno MXCSR canonico mediante
 *      `CanonicalMxcsrScope` (devolviendo el MXCSR original del hilo al salir), ejecuta `ExactScan`
 *      verificando que toda distancia sea finita (< FLT_MAX), y registra `fallback_triggered = 1`.
 */
static void handle_ivf_non_nominal_contract_or_fallback(const IVFPartitionedIndex& idx,
                                                        const float* q,
                                                        const QueryPredicate& pred,
                                                        uint32_t k,
                                                        Neighbor* out_results,
                                                        uint32_t* out_count,
                                                        QueryExecutionMetrics& metrics,
                                                        TimerScope& ts,
                                                        bool mxcsr_was_valid) {
    /* Situacion 1: Corpus / indice invalido rechazado en construccion */
    if (!idx.certified_Contract_valid || idx.num_vectors == 0U || idx.dim != 64U || !idx.part_orig_vectors) {
        *out_count = 0U;
        metrics.fallback_triggered = 0U;
        metrics.query_rejected = 1U;
        metrics.status = QueryExecutionStatus::REJECTED_INVALID_CORPUS_INDEX;
        ts.mark_prep_done();
        ts.stop(metrics);
        return;
    }

    /* Establecer entorno MXCSR canonico temporal y restaurar automaticamente al salir */
    CanonicalMxcsrScope mxcsr_guard;

    /* Situacion 2: Verificar si la consulta puede evaluarse en ExactScan sin overflow ni NaN/Inf */
    if (!check_query_canonical_fallback_safe(q, idx.dim)) {
        *out_count = 0U;
        metrics.fallback_triggered = 0U;
        metrics.query_rejected = 1U;
        metrics.status = QueryExecutionStatus::REJECTED_UNSAFE_QUERY_DOMAIN;
        ts.mark_prep_done();
        ts.stop(metrics);
        return;
    }

    /* Situacion 3: Fallback ExactScan bajo entorno MXCSR canonico y dominio finito seguro */
    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t dim = idx.dim;
    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t n_el = static_cast<uint32_t>(idx.eligible_ivf_ids_per_sel[pred.sel_index].size());
    const float* base_orig = idx.part_orig_vectors;

    for (uint32_t e = 0; e < n_el; ++e) {
        uint32_t iv = el_ivf_ids[e];
        float d_orig = l2_hito9_ref(q, base_orig + static_cast<size_t>(iv) * dim, dim);
        if (!std::isfinite(d_orig) || d_orig >= FLT_MAX) {
            *out_count = 0U;
            metrics.fallback_triggered = 0U;
            metrics.query_rejected = 1U;
            metrics.status = QueryExecutionStatus::REJECTED_UNSAFE_QUERY_DOMAIN;
            ts.stop(metrics);
            return;
        }
        heap.push(idx.part_aos_ids[iv], d_orig);
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = n_el;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = idx.num_partitions;
    metrics.partitions_total = idx.num_partitions;
    metrics.bytes_read = static_cast<uint64_t>(n_el) * (dim * sizeof(float) + sizeof(uint32_t));
    metrics.fallback_triggered = 1U;
    metrics.query_rejected = 0U;
    metrics.status = (!mxcsr_was_valid)
                   ? QueryExecutionStatus::FALLBACK_MXCSR_RESTORED
                   : QueryExecutionStatus::FALLBACK_QUERY_OUT_OF_CERT_DOMAIN;
}

/*
 * Rota el vector centrado (q - mu) por Q^T y calcula cota superior de ||q - mu||_2
 * utilizando redondeo hacia afuera en binary64 + round_up_f32.
 */
static inline void rotate_centered_query_avx2(const float* qt,
                                              const float* q,
                                              const float* mean,
                                              float* q_rot,
                                              float& out_q_mean_norm,
                                              uint32_t dim) {
    alignas(64) float q_diff[64];
    __m256 v_norm_sq = _mm256_setzero_ps();
    for (uint32_t d = 0; d < dim; d += 8) {
        __m256 diff = _mm256_sub_ps(_mm256_loadu_ps(q + d), _mm256_load_ps(mean + d));
        _mm256_store_ps(q_diff + d, diff);
        v_norm_sq = _mm256_fmadd_ps(diff, diff, v_norm_sq);
    }
    double raw_norm_sq = static_cast<double>(hsum256_ps_fast(v_norm_sq));
    double norm_up = div_up_f64(sqrt_up_f64(raw_norm_sq), sqrt_down_f64(one_minus_gamma_down_f64(66)));
    out_q_mean_norm = round_up_f32(norm_up);

    for (uint32_t r = 0; r < dim; ++r) {
        const float* row = qt + static_cast<size_t>(r) * dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (uint32_t d = 0; d < dim; d += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(row + d), _mm256_load_ps(q_diff + d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(row + d + 8), _mm256_load_ps(q_diff + d + 8), acc1);
        }
        q_rot[r] = hsum256_ps_fast(_mm256_add_ps(acc0, acc1));
    }
}

struct PartitionOrder {
    uint32_t part_id;
    float dc;           /* ||q - c_p||_2 en espacio original */
    float dc_low;       /* Redondeado hacia 0 */
    float dc_high;      /* Redondeado hacia +inf */
    float lb_part_sq;   /* Cota inferior certificada de particion al cuadrado (redondeada hacia 0) */
};

/*
 * Plan 4: IVF Flat AVX2 en espacio original (SIN rotacion de consulta ni poda)
 */
void execute_plan4_ivf_flat_avx2(const IVFPartitionedIndex& idx,
                                 const float* q,
                                 const QueryPredicate& pred,
                                 uint32_t k,
                                 Neighbor* out_results,
                                 uint32_t* out_count,
                                 QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();

    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        p_order[p] = {p, dc, dc, dc, 0.0f};
    }
    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_orig = idx.part_orig_vectors;

    uint32_t dists_eval = 0;
    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];

        uint32_t e = e_start;
        for (; e + 4 <= e_end; e += 4) {
            uint32_t iv0 = el_ivf_ids[e];
            uint32_t iv1 = el_ivf_ids[e + 1];
            uint32_t iv2 = el_ivf_ids[e + 2];
            uint32_t iv3 = el_ivf_ids[e + 3];

            const float* v0 = base_orig + static_cast<size_t>(iv0) * dim;
            const float* v1 = base_orig + static_cast<size_t>(iv1) * dim;
            const float* v2 = base_orig + static_cast<size_t>(iv2) * dim;
            const float* v3 = base_orig + static_cast<size_t>(iv3) * dim;

            dists_eval += 4;
            __m128 d4 = l2_sq_avx2_4x_aos(q, v0, v1, v2, v3, dim);
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
            int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
            if (mask != 0) {
                if (mask & 1) heap.push(idx.part_aos_ids[iv0], l2_hito9_ref(q, v0, dim));
                if (mask & 2) heap.push(idx.part_aos_ids[iv1], l2_hito9_ref(q, v1, dim));
                if (mask & 4) heap.push(idx.part_aos_ids[iv2], l2_hito9_ref(q, v2, dim));
                if (mask & 8) heap.push(idx.part_aos_ids[iv3], l2_hito9_ref(q, v3, dim));
            }
        }
        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            const float* vp = base_orig + static_cast<size_t>(iv) * dim;
            dists_eval++;
            float d = l2_sq_avx2_1x_aos(q, vp, dim);
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
            if (d <= gate) {
                heap.push(idx.part_aos_ids[iv], l2_hito9_ref(q, vp, dim));
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = dists_eval;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = n_parts;
    metrics.partitions_total = n_parts;
    metrics.bytes_read = static_cast<uint64_t>(dists_eval) * (dim * sizeof(float) + sizeof(uint32_t));
}

/*
 * Plan 5A: IVF + Poda Exacta Conservadora por Particion en espacio original
 */
void execute_plan5a_ivf_part_prune_avx2(const IVFPartitionedIndex& idx,
                                        const float* q,
                                        const QueryPredicate& pred,
                                        uint32_t k,
                                        Neighbor* out_results,
                                        uint32_t* out_count,
                                        QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();

    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;

    /* Validacion del contrato de ejecucion IEEE-754: indice valido + MXCSR + dominio de la consulta */
    const bool mxcsr_ok = check_ieee754_mxcsr_state();
    if (!idx.certified_Contract_valid ||
        !mxcsr_ok ||
        !check_vector_domain_certified(q, dim)) {
        handle_ivf_non_nominal_contract_or_fallback(idx, q, pred, k, out_results, out_count, metrics, ts, mxcsr_ok);
        return;
    }
    metrics.status = QueryExecutionStatus::OK_CERTIFIED;

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        const GeometricPartition& gp = idx.partitions[p];
        PartitionDistanceBounds pdb = compute_partition_distance_bounds(d2, gp.min_radius, gp.max_radius);
        p_order[p] = {p, pdb.dc, pdb.dc_low, pdb.dc_high, pdb.lb_part_sq};
    }
    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_orig = idx.part_orig_vectors;

    uint32_t dists_eval = 0;
    uint32_t parts_visited = 0;

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        if (heap.size == k) {
            if (p_order[pi].lb_part_sq > heap.threshold_sq) {
                continue;
            }
        }

        parts_visited++;
        uint32_t e = e_start;
        for (; e + 4 <= e_end; e += 4) {
            uint32_t iv0 = el_ivf_ids[e];
            uint32_t iv1 = el_ivf_ids[e + 1];
            uint32_t iv2 = el_ivf_ids[e + 2];
            uint32_t iv3 = el_ivf_ids[e + 3];

            const float* v0 = base_orig + static_cast<size_t>(iv0) * dim;
            const float* v1 = base_orig + static_cast<size_t>(iv1) * dim;
            const float* v2 = base_orig + static_cast<size_t>(iv2) * dim;
            const float* v3 = base_orig + static_cast<size_t>(iv3) * dim;

            dists_eval += 4;
            __m128 d4 = l2_sq_avx2_4x_aos(q, v0, v1, v2, v3, dim);
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
            int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
            if (mask != 0) {
                if (mask & 1) heap.push(idx.part_aos_ids[iv0], l2_hito9_ref(q, v0, dim));
                if (mask & 2) heap.push(idx.part_aos_ids[iv1], l2_hito9_ref(q, v1, dim));
                if (mask & 4) heap.push(idx.part_aos_ids[iv2], l2_hito9_ref(q, v2, dim));
                if (mask & 8) heap.push(idx.part_aos_ids[iv3], l2_hito9_ref(q, v3, dim));
            }
        }
        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            const float* vp = base_orig + static_cast<size_t>(iv) * dim;
            dists_eval++;
            float d = l2_sq_avx2_1x_aos(q, vp, dim);
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
            if (d <= gate) {
                heap.push(idx.part_aos_ids[iv], l2_hito9_ref(q, vp, dim));
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = dists_eval;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = parts_visited;
    metrics.partitions_total = n_parts;
    metrics.bytes_read = static_cast<uint64_t>(dists_eval) * (dim * sizeof(float) + sizeof(uint32_t));
}

/*
 * ============================================================================
 * KERNEL UNIFICADO DE CASCADA PROGRESIVA DE 5 ETAPAS (HITO 2A.3)
 * ============================================================================
 * Para garantizar por construccion que el codigo de produccion (`Plan 5B`) y el
 * auditor (`audit_plan5b_pruning_soundness`) jamas puedan divergir, ambos ejecutan
 * exactamente las mismas plantillas `AuditMode = false` (produccion, 0 overhead) y
 * `AuditMode = true` (auditoria exhaustiva de las 5 etapas).
 */
enum class PruneStage : uint8_t {
    PART   = 0,
    RADIAL = 1,
    PCA16  = 2,
    PCA32  = 3,
    PCA64  = 4
};

struct NoOpPruneObserver {
    inline void on_checked(uint32_t) const {}
    inline void on_pruned(PruneStage, uint32_t, double) const {}
    inline void on_evaluated_orig(uint32_t) const {}
};

template <bool AuditMode, typename Observer>
static inline void eval_4x_pca_progressive_unified(const float* q_orig,
                                                   const float* q_rot_p,
                                                   float q_tail48_p,
                                                   float q_tail32_p,
                                                   const PartitionCertifiedBounds& bounds,
                                                   const IVFPartitionedIndex& idx,
                                                   uint32_t iv0, uint32_t iv1, uint32_t iv2, uint32_t iv3,
                                                   TopKHeap& heap,
                                                   uint32_t& quarter_evals,
                                                   uint64_t& bytes_read,
                                                   Observer& obs) {
    if constexpr (AuditMode) {
        obs.on_checked(4);
    }

    const float* v0 = idx.part_aos_vectors + static_cast<size_t>(iv0) * 64U;
    const float* v1 = idx.part_aos_vectors + static_cast<size_t>(iv1) * 64U;
    const float* v2 = idx.part_aos_vectors + static_cast<size_t>(iv2) * 64U;
    const float* v3 = idx.part_aos_vectors + static_cast<size_t>(iv3) * 64U;

    /* Etapa 3 (PCA 16D): Primeras 16 dimensiones principales de (q - c_p) vs (x - c_p) */
    __m256 q0 = _mm256_load_ps(q_rot_p + 0);
    __m256 q1 = _mm256_load_ps(q_rot_p + 8);

    __m256 d0_a = _mm256_sub_ps(q0, _mm256_load_ps(v0 + 0));
    __m256 d0_b = _mm256_sub_ps(q1, _mm256_load_ps(v0 + 8));
    __m256 acc0 = _mm256_fmadd_ps(d0_a, d0_a, _mm256_mul_ps(d0_b, d0_b));

    __m256 d1_a = _mm256_sub_ps(q0, _mm256_load_ps(v1 + 0));
    __m256 d1_b = _mm256_sub_ps(q1, _mm256_load_ps(v1 + 8));
    __m256 acc1 = _mm256_fmadd_ps(d1_a, d1_a, _mm256_mul_ps(d1_b, d1_b));

    __m256 d2_a = _mm256_sub_ps(q0, _mm256_load_ps(v2 + 0));
    __m256 d2_b = _mm256_sub_ps(q1, _mm256_load_ps(v2 + 8));
    __m256 acc2 = _mm256_fmadd_ps(d2_a, d2_a, _mm256_mul_ps(d2_b, d2_b));

    __m256 d3_a = _mm256_sub_ps(q0, _mm256_load_ps(v3 + 0));
    __m256 d3_b = _mm256_sub_ps(q1, _mm256_load_ps(v3 + 8));
    __m256 acc3 = _mm256_fmadd_ps(d3_a, d3_a, _mm256_mul_ps(d3_b, d3_b));

    __m256 t0 = _mm256_hadd_ps(acc0, acc1);
    __m256 t1 = _mm256_hadd_ps(acc2, acc3);
    __m256 t2 = _mm256_hadd_ps(t0, t1);
    __m128 d16_4 = _mm_add_ps(_mm256_castps256_ps128(t2), _mm256_extractf128_ps(t2, 1));

    quarter_evals += 4;
    bytes_read += 4 * 64 + 4 * sizeof(float);

    __m128 b4 = _mm_set_ps(idx.part_aos_tail48_norm[iv3],
                           idx.part_aos_tail48_norm[iv2],
                           idx.part_aos_tail48_norm[iv1],
                           idx.part_aos_tail48_norm[iv0]);
    __m128 raw_diff = _mm_sub_ps(_mm_set1_ps(q_tail48_p), b4);
    __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), raw_diff);
    __m128 lb_tail = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(abs_diff, _mm_set1_ps(bounds.fp_slack_p)));
    __m128 d16_safe = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(d16_4, _mm_set1_ps(bounds.sq_slack_16_p)));
    __m128 lb64_4 = _mm_mul_ps(_mm_fmadd_ps(lb_tail, lb_tail, d16_safe), _mm_set1_ps(bounds.scale_safe_16));

    int mask16 = _mm_movemask_ps(_mm_cmple_ps(lb64_4, _mm_set1_ps(heap.threshold_sq)));
    if constexpr (!AuditMode) {
        if (mask16 == 0) {
            return;
        }
    }

    alignas(16) float p16[4];
    alignas(16) float lb16_vals[4];
    _mm_store_ps(p16, d16_4);
    if constexpr (AuditMode) {
        _mm_store_ps(lb16_vals, lb64_4);
    }
    const uint32_t ivs[4] = {iv0, iv1, iv2, iv3};
    const float* v_ptrs[4] = {v0, v1, v2, v3};

    for (int lane = 0; lane < 4; ++lane) {
        uint32_t iv = ivs[lane];
        if (!(mask16 & (1 << lane))) {
            if constexpr (AuditMode) {
                obs.on_pruned(PruneStage::PCA16, iv, static_cast<double>(lb16_vals[lane]));
            }
            continue;
        }
        const float* vp = v_ptrs[lane];

        /* Etapa 4 (PCA 32D): Dimensiones 16..31 (2da cache line) */
        __m256 qa = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 16), _mm256_load_ps(vp + 16));
        __m256 qb = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 24), _mm256_load_ps(vp + 24));
        float d32 = p16[lane] + hsum256_ps_fast(_mm256_fmadd_ps(qa, qa, _mm256_mul_ps(qb, qb)));
        quarter_evals += 1;
        bytes_read += 64;

        float abs_t32 = std::fabs(q_tail32_p - idx.part_aos_tail32_norm[iv]);
        float lb_t32 = (abs_t32 > bounds.fp_slack_p) ? (abs_t32 - bounds.fp_slack_p) : 0.0f;
        float d32_safe = (d32 > bounds.sq_slack_32_p) ? (d32 - bounds.sq_slack_32_p) : 0.0f;
        float lb64_stage2 = bounds.scale_safe_32 * (d32_safe + lb_t32 * lb_t32);
        if (lb64_stage2 > heap.threshold_sq) {
            if constexpr (AuditMode) {
                obs.on_pruned(PruneStage::PCA32, iv, static_cast<double>(lb64_stage2));
            }
            continue;
        }

        /* Etapa 5 (PCA 64D): Completar dimensiones 32..63 en espacio rotado */
        __m256 qc = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 32), _mm256_load_ps(vp + 32));
        __m256 qd = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 40), _mm256_load_ps(vp + 40));
        __m256 acc_tail0 = _mm256_fmadd_ps(qc, qc, _mm256_mul_ps(qd, qd));

        __m256 qe = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 48), _mm256_load_ps(vp + 48));
        __m256 qf = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 56), _mm256_load_ps(vp + 56));
        __m256 acc_tail1 = _mm256_fmadd_ps(qe, qe, _mm256_mul_ps(qf, qf));

        float d64_rot = d32 + hsum256_ps_fast(_mm256_add_ps(acc_tail0, acc_tail1));
        quarter_evals += 2;
        bytes_read += 128;

        float d64_safe = (d64_rot > bounds.sq_slack_64_p) ? (d64_rot - bounds.sq_slack_64_p) : 0.0f;
        float lb64_stage3 = bounds.scale_safe_64 * d64_safe;
        if (lb64_stage3 > heap.threshold_sq) {
            if constexpr (AuditMode) {
                obs.on_pruned(PruneStage::PCA64, iv, static_cast<double>(lb64_stage3));
            }
            continue;
        }

        /* Nivel 2 de Certificacion: Recalcular distancia final exacta en el espacio original float32 */
        if constexpr (AuditMode) {
            obs.on_evaluated_orig(1);
        }
        const float* orig_vp = idx.part_orig_vectors + static_cast<size_t>(iv) * 64U;
        float d_orig = l2_hito9_ref(q_orig, orig_vp, 64U);
        bytes_read += 256;
        heap.push(idx.part_aos_ids[iv], d_orig);
    }
}

template <bool AuditMode, typename Observer>
static inline void eval_1x_pca_progressive_unified(const float* q_orig,
                                                   const float* q_rot_p,
                                                   float q_tail48_p,
                                                   float q_tail32_p,
                                                   const PartitionCertifiedBounds& bounds,
                                                   const IVFPartitionedIndex& idx,
                                                   uint32_t iv,
                                                   TopKHeap& heap,
                                                   uint32_t& quarter_evals,
                                                   uint64_t& bytes_read,
                                                   Observer& obs) {
    if constexpr (AuditMode) {
        obs.on_checked(1);
    }
    const float* vp = idx.part_aos_vectors + static_cast<size_t>(iv) * 64U;

    /* Etapa 3 (PCA 16D): 16D + cola 48D */
    __m256 q0 = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 0), _mm256_load_ps(vp + 0));
    __m256 q1 = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 8), _mm256_load_ps(vp + 8));
    float d16 = hsum256_ps_fast(_mm256_fmadd_ps(q0, q0, _mm256_mul_ps(q1, q1)));
    quarter_evals += 1;
    bytes_read += 68;

    float abs_t48 = std::fabs(q_tail48_p - idx.part_aos_tail48_norm[iv]);
    float lb_t48 = (abs_t48 > bounds.fp_slack_p) ? (abs_t48 - bounds.fp_slack_p) : 0.0f;
    float d16_safe = (d16 > bounds.sq_slack_16_p) ? (d16 - bounds.sq_slack_16_p) : 0.0f;
    float lb64_s1 = bounds.scale_safe_16 * (d16_safe + lb_t48 * lb_t48);
    if (lb64_s1 > heap.threshold_sq) {
        if constexpr (AuditMode) {
            obs.on_pruned(PruneStage::PCA16, iv, static_cast<double>(lb64_s1));
        }
        return;
    }

    /* Etapa 4 (PCA 32D): 32D + cola 32D */
    __m256 qa = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 16), _mm256_load_ps(vp + 16));
    __m256 qb = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 24), _mm256_load_ps(vp + 24));
    float d32 = d16 + hsum256_ps_fast(_mm256_fmadd_ps(qa, qa, _mm256_mul_ps(qb, qb)));
    quarter_evals += 1;
    bytes_read += 64;

    float abs_t32 = std::fabs(q_tail32_p - idx.part_aos_tail32_norm[iv]);
    float lb_t32 = (abs_t32 > bounds.fp_slack_p) ? (abs_t32 - bounds.fp_slack_p) : 0.0f;
    float d32_safe = (d32 > bounds.sq_slack_32_p) ? (d32 - bounds.sq_slack_32_p) : 0.0f;
    float lb64_s2 = bounds.scale_safe_32 * (d32_safe + lb_t32 * lb_t32);
    if (lb64_s2 > heap.threshold_sq) {
        if constexpr (AuditMode) {
            obs.on_pruned(PruneStage::PCA32, iv, static_cast<double>(lb64_s2));
        }
        return;
    }

    /* Etapa 5 (PCA 64D): 64D completo en espacio rotado */
    __m256 qc = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 32), _mm256_load_ps(vp + 32));
    __m256 qd = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 40), _mm256_load_ps(vp + 40));
    __m256 acc_tail0 = _mm256_fmadd_ps(qc, qc, _mm256_mul_ps(qd, qd));

    __m256 qe = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 48), _mm256_load_ps(vp + 48));
    __m256 qf = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 56), _mm256_load_ps(vp + 56));
    __m256 acc_tail1 = _mm256_fmadd_ps(qe, qe, _mm256_mul_ps(qf, qf));

    float d64_rot = d32 + hsum256_ps_fast(_mm256_add_ps(acc_tail0, acc_tail1));
    quarter_evals += 2;
    bytes_read += 128;

    float d64_safe = (d64_rot > bounds.sq_slack_64_p) ? (d64_rot - bounds.sq_slack_64_p) : 0.0f;
    float lb64_s3 = bounds.scale_safe_64 * d64_safe;
    if (lb64_s3 > heap.threshold_sq) {
        if constexpr (AuditMode) {
            obs.on_pruned(PruneStage::PCA64, iv, static_cast<double>(lb64_s3));
        }
        return;
    }

    if constexpr (AuditMode) {
        obs.on_evaluated_orig(1);
    }
    const float* orig_vp = idx.part_orig_vectors + static_cast<size_t>(iv) * 64U;
    float d_orig = l2_hito9_ref(q_orig, orig_vp, 64U);
    bytes_read += 256;
    heap.push(idx.part_aos_ids[iv], d_orig);
}

template <bool AuditMode, typename Observer>
static void run_plan5b_unified_core(const IVFPartitionedIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    TopKHeap& heap,
                                    QueryExecutionMetrics& metrics,
                                    TimerScope* opt_ts,
                                    Observer& obs) {
    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;
    alignas(64) float q_rot[64];
    float q_mean_norm = 0.0f;
    rotate_centered_query_avx2(idx.ortho_basis, q, idx.global_mean, q_rot, q_mean_norm, dim);

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        const GeometricPartition& gp = idx.partitions[p];
        PartitionDistanceBounds pdb = compute_partition_distance_bounds(d2, gp.min_radius, gp.max_radius);
        p_order[p] = {p, pdb.dc, pdb.dc_low, pdb.dc_high, pdb.lb_part_sq};
    }
    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    if (opt_ts) {
        opt_ts->mark_prep_done();
    }

    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_orig = idx.part_orig_vectors;

    uint32_t quarter_evals = 0;
    uint32_t parts_visited = 0;
    uint64_t total_bytes = 0;

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        float dc_low = p_order[pi].dc_low;
        float dc_high = p_order[pi].dc_high;
        const GeometricPartition& gp = idx.partitions[pid];

        if (heap.size == k) {
            if (p_order[pi].lb_part_sq > heap.threshold_sq) {
                /* Etapa 1 (Particion): Particion entera descartada con garantia conservadora */
                if constexpr (AuditMode) {
                    for (uint32_t e = e_start; e < e_end; ++e) {
                        obs.on_checked(1);
                        obs.on_pruned(PruneStage::PART, el_ivf_ids[e], static_cast<double>(p_order[pi].lb_part_sq));
                    }
                }
                continue;
            }

            /* Etapa 2 (Radial): Poda Exacta Conservadora de Cascara Radial [r_low, r_high] */
            float r_low = 0.0f, r_high = 0.0f, r_pad = 0.0f;
            compute_radial_shell_interval(dc_low, dc_high, gp.max_radius, heap.threshold_sq, r_low, r_high, r_pad);

            uint32_t orig_e_start = e_start;
            uint32_t orig_e_end = e_end;

            if (r_low > gp.min_radius) {
                uint32_t lo = e_start, hi = e_end;
                while (lo < hi) {
                    uint32_t mid = lo + ((hi - lo) >> 1);
                    if (idx.part_aos_radii[el_ivf_ids[mid]] < r_low) lo = mid + 1;
                    else hi = mid;
                }
                e_start = lo;
            }
            if (r_high < gp.max_radius && e_start < e_end) {
                uint32_t lo = e_start, hi = e_end;
                while (lo < hi) {
                    uint32_t mid = lo + ((hi - lo) >> 1);
                    if (idx.part_aos_radii[el_ivf_ids[mid]] <= r_high) lo = mid + 1;
                    else hi = mid;
                }
                e_end = lo;
            }

            if constexpr (AuditMode) {
                const double one_m_g66_down = one_minus_gamma_down_f64(66);
                for (uint32_t e = orig_e_start; e < e_start; ++e) {
                    uint32_t iv = el_ivf_ids[e];
                    obs.on_checked(1);
                    double r_x = static_cast<double>(idx.part_aos_radii[iv]);
                    double raw_rad = sub_down_f64(sub_down_f64(static_cast<double>(dc_low), r_x),
                                                  static_cast<double>(r_pad));
                    float lb_rad_sq = round_down_f32(mul_down_f64(mul_down_f64(raw_rad, raw_rad), one_m_g66_down));
                    obs.on_pruned(PruneStage::RADIAL, iv, static_cast<double>(lb_rad_sq));
                }
                for (uint32_t e = e_end; e < orig_e_end; ++e) {
                    uint32_t iv = el_ivf_ids[e];
                    obs.on_checked(1);
                    double r_x = static_cast<double>(idx.part_aos_radii[iv]);
                    double raw_rad = sub_down_f64(sub_down_f64(r_x, static_cast<double>(dc_high)),
                                                  static_cast<double>(r_pad));
                    float lb_rad_sq = round_down_f32(mul_down_f64(mul_down_f64(raw_rad, raw_rad), one_m_g66_down));
                    obs.on_pruned(PruneStage::RADIAL, iv, static_cast<double>(lb_rad_sq));
                }
            }

            if (e_start >= e_end) continue;
        }

        parts_visited++;
        uint32_t e = e_start;

        while (e + 4 <= e_end && heap.size < k) {
            uint32_t iv0 = el_ivf_ids[e];
            uint32_t iv1 = el_ivf_ids[e + 1];
            uint32_t iv2 = el_ivf_ids[e + 2];
            uint32_t iv3 = el_ivf_ids[e + 3];
            quarter_evals += 16;
            total_bytes += 4 * 256;
            if constexpr (AuditMode) {
                obs.on_checked(4);
                obs.on_evaluated_orig(4);
            }

            const float* v0 = base_orig + static_cast<size_t>(iv0) * dim;
            const float* v1 = base_orig + static_cast<size_t>(iv1) * dim;
            const float* v2 = base_orig + static_cast<size_t>(iv2) * dim;
            const float* v3 = base_orig + static_cast<size_t>(iv3) * dim;

            heap.push(idx.part_aos_ids[iv0], l2_hito9_ref(q, v0, dim));
            heap.push(idx.part_aos_ids[iv1], l2_hito9_ref(q, v1, dim));
            heap.push(idx.part_aos_ids[iv2], l2_hito9_ref(q, v2, dim));
            heap.push(idx.part_aos_ids[iv3], l2_hito9_ref(q, v3, dim));
            e += 4;
        }
        while (e < e_end && heap.size < k) {
            uint32_t iv = el_ivf_ids[e];
            quarter_evals += 4;
            total_bytes += 256;
            if constexpr (AuditMode) {
                obs.on_checked(1);
                obs.on_evaluated_orig(1);
            }
            heap.push(idx.part_aos_ids[iv], l2_hito9_ref(q, base_orig + static_cast<size_t>(iv) * dim, dim));
            e++;
        }
        if (e >= e_end) continue;

        alignas(64) float q_rot_p[64];
        const float* rot_c = idx.rotated_centroids + static_cast<size_t>(pid) * 64U;
        __m256 v_tail16_31 = _mm256_setzero_ps();
        __m256 v_tail32_63 = _mm256_setzero_ps();

        for (uint32_t d = 0; d < 16; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
            _mm256_store_ps(q_rot_p + d, diff);
        }
        for (uint32_t d = 16; d < 32; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
            _mm256_store_ps(q_rot_p + d, diff);
            v_tail16_31 = _mm256_fmadd_ps(diff, diff, v_tail16_31);
        }
        for (uint32_t d = 32; d < 64; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
            _mm256_store_ps(q_rot_p + d, diff);
            v_tail32_63 = _mm256_fmadd_ps(diff, diff, v_tail32_63);
        }

        float qt32_sq = hsum256_ps_fast(v_tail32_63);
        float qt16_31_sq = hsum256_ps_fast(v_tail16_31);
        float q_tail32_p = sqrtf(qt32_sq);
        float q_tail48_p = sqrtf(qt32_sq + qt16_31_sq);

        PartitionCertifiedBounds bounds = compute_partition_certified_bounds(
            q_mean_norm, idx.centroid_mean_norms[pid], dc_high, gp.max_radius, idx.ortho_defect
        );

        for (; e + 4 <= e_end; e += 4) {
            eval_4x_pca_progressive_unified<AuditMode>(
                q, q_rot_p, q_tail48_p, q_tail32_p,
                bounds, idx,
                el_ivf_ids[e], el_ivf_ids[e + 1], el_ivf_ids[e + 2], el_ivf_ids[e + 3],
                heap, quarter_evals, total_bytes, obs
            );
        }

        for (; e < e_end; ++e) {
            eval_1x_pca_progressive_unified<AuditMode>(
                q, q_rot_p, q_tail48_p, q_tail32_p,
                bounds, idx,
                el_ivf_ids[e],
                heap, quarter_evals, total_bytes, obs
            );
        }
    }

    metrics.dists_evaluated = (quarter_evals + 2) / 4;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = parts_visited;
    metrics.partitions_total = n_parts;
    metrics.bytes_read = total_bytes;
}

void execute_plan5b_ivf_full_exact_prune_avx2(const IVFPartitionedIndex& idx,
                                              const float* q,
                                              const QueryPredicate& pred,
                                              uint32_t k,
                                              Neighbor* out_results,
                                              uint32_t* out_count,
                                              QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();

    /* Validacion del contrato de ejecucion IEEE-754: indice valido + MXCSR + dominio de la consulta */
    const bool mxcsr_ok = check_ieee754_mxcsr_state();
    if (!idx.certified_Contract_valid ||
        !mxcsr_ok ||
        !check_vector_domain_certified(q, idx.dim)) {
        handle_ivf_non_nominal_contract_or_fallback(idx, q, pred, k, out_results, out_count, metrics, ts, mxcsr_ok);
        return;
    }
    metrics.status = QueryExecutionStatus::OK_CERTIFIED;

    TopKHeap heap;
    NoOpPruneObserver noop_obs;
    run_plan5b_unified_core<false>(idx, q, pred, k, heap, metrics, &ts, noop_obs);

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
}

/*
 * Auditoria Directa de Poda sin Tolerancia Oculta en las 5 Etapas (Hito 2A.3 - Nivel 3):
 * Ejecuta exactamente el mismo nucleo templado `run_plan5b_unified_core<true>` que el
 * kernel de produccion `Plan 5B`, eliminando toda duplicacion de codigo SIMD/escalar.
 */
void audit_plan5b_pruning_soundness(const IVFPartitionedIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    const Neighbor* true_topk,
                                    uint32_t true_topk_count,
                                    PruningAuditReport& report) {
    if (!idx.certified_Contract_valid ||
        !check_ieee754_mxcsr_state() ||
        !check_vector_domain_certified(q, idx.dim)) {
        report.contract_fallbacks++;
        uint32_t n_el = static_cast<uint32_t>(idx.eligible_ivf_ids_per_sel[pred.sel_index].size());
        report.total_eligible_checked += n_el;
        report.evaluated_full_64d += n_el;
        return;
    }

    struct AuditObserver {
        const IVFPartitionedIndex& idx;
        const float* q;
        const Neighbor* true_topk;
        uint32_t true_topk_count;
        PruningAuditReport& report;

        inline bool is_in_true_topk(uint32_t orig_id) const {
            for (uint32_t t = 0; t < true_topk_count; ++t) {
                if (true_topk[t].id == orig_id) return true;
            }
            return false;
        }

        inline void on_checked(uint32_t cnt) {
            report.total_eligible_checked += cnt;
        }

        inline void on_evaluated_orig(uint32_t cnt) {
            report.evaluated_full_64d += cnt;
        }

        inline void on_pruned(PruneStage stage, uint32_t iv, double lb_sq) {
            uint64_t* p_stage_cnt = nullptr;
            uint64_t* p_strict    = nullptr;
            uint64_t* p_margin    = nullptr;
            uint64_t* p_exceed    = nullptr;

            switch (stage) {
                case PruneStage::PART:
                    p_stage_cnt = &report.pruned_by_partition;
                    p_strict    = &report.strict_violations_partition;
                    p_margin    = &report.diag_margin_events_partition;
                    p_exceed    = &report.diag_exceed_violations_partition;
                    break;
                case PruneStage::RADIAL:
                    p_stage_cnt = &report.pruned_by_radial_shell;
                    p_strict    = &report.strict_violations_radial;
                    p_margin    = &report.diag_margin_events_radial;
                    p_exceed    = &report.diag_exceed_violations_radial;
                    break;
                case PruneStage::PCA16:
                    p_stage_cnt = &report.pruned_by_pca_16d;
                    p_strict    = &report.strict_violations_pca16;
                    p_margin    = &report.diag_margin_events_pca16;
                    p_exceed    = &report.diag_exceed_violations_pca16;
                    break;
                case PruneStage::PCA32:
                    p_stage_cnt = &report.pruned_by_pca_32d;
                    p_strict    = &report.strict_violations_pca32;
                    p_margin    = &report.diag_margin_events_pca32;
                    p_exceed    = &report.diag_exceed_violations_pca32;
                    break;
                case PruneStage::PCA64:
                    p_stage_cnt = &report.pruned_by_pca_64d;
                    p_strict    = &report.strict_violations_pca64;
                    p_margin    = &report.diag_margin_events_pca64;
                    p_exceed    = &report.diag_exceed_violations_pca64;
                    break;
            }

            (*p_stage_cnt)++;

            uint32_t orig_id = idx.part_aos_ids[iv];
            const float* base_orig = idx.part_orig_vectors;
            double orig_d2 = static_cast<double>(l2_hito9_ref(q, base_orig + static_cast<size_t>(iv) * idx.dim, idx.dim));

            /* Exceso firmado exacto: Delta = LB_safe^2 - d_orig^2 (sin ninguna tolerancia oculta) */
            double signed_excess = lb_sq - orig_d2;
            if (signed_excess > report.max_signed_excess) {
                report.max_signed_excess = signed_excess;
            }

            if (signed_excess > 0.0) {
                (*p_strict)++;
                if (signed_excess > report.max_positive_excess) {
                    report.max_positive_excess = signed_excess;
                }
                double diag_thresh = orig_d2 * ieee754_gamma(66);
                if (signed_excess <= diag_thresh) {
                    (*p_margin)++;
                } else {
                    (*p_exceed)++;
                }
            }

            if (lb_sq > 0.0) {
                double ratio = orig_d2 / lb_sq;
                if (ratio < report.min_ratio_orig_over_lb) {
                    report.min_ratio_orig_over_lb = ratio;
                }
            }
            if (is_in_true_topk(orig_id)) {
                report.false_prunings_topk++;
            }
        }
    };

    TopKHeap heap;
    QueryExecutionMetrics dummy_metrics{};
    AuditObserver audit_obs{idx, q, true_topk, true_topk_count, report};
    run_plan5b_unified_core<true>(idx, q, pred, k, heap, dummy_metrics, nullptr, audit_obs);
}

/*
 * ============================================================================
 * HITO 2B: SELECTOR ADAPTATIVO LIGERO GUIADO POR MODELO DE COSTO (PLAN 6)
 * ============================================================================
 * Restricciones metodologicas de Hito 2B:
 *   1. Mantiene congelados los kernels certificados 5A y 5B y el contrato IEEE-754.
 *   2. Usa Plan 5A como baseline de ejecucion (Prep identico a 5A, sin rotar q por defecto).
 *   3. Estructura la ejecucion en 3 Fases Secuenciales sin sobrecarga de registros en 5A:
 *      - Fase 1 (Sembrado Local en 5A): Evalua las primeras 3 particiones no vacias
 *        (el cluster local de la consulta) con el kernel 4x AVX2 de 5A hasta poblar
 *        `heap.size == k && seed_nonempty_parts >= 3`, convergiendo `heap.threshold_sq`
 *        en ~2.5 us y evitando la doble evaluacion (PCA 64D + orig 64D = 512B/vec) que
 *        sufre 5B en las particiones mas cercanas donde casi ningun vector es podable.
 *      - Modelo de Costo de 3 Compuertas Deterministas:
 *        * Compuerta 1 (Suficiencia de Poda de Particion 5A): Si `p_surv_5a <= 32` o
 *          `n_surv_5a <= 4500` (ej. `R_k64_s003_Tight`), 5A ya poda >75% del corpus en
 *          O(1) y termina en Fase 3 con 5A puro sin rotar `q`.
 *        * Compuerta 2 (Concentracion Geometrica gamma_0, Compacidad eta_pot y Densidad m_bar):
 *          Calcula `gamma_0 = d_c(0)^2 / d_med^2`, `eta_pot = theta_K^2 / d_med^2` y la
 *          ocupacion media por particion sobreviviente `m_bar = n_surv_5a / p_surv_5a`.
 *          Las consultas difusas o uniformes en la esfera (`gamma_0 > 0.46`) y las bolas
 *          Top-K demasiado anchas para amortizar el setup por particion de 5B se descartan
 *          de inmediato sin rotar `q`, pasando directo a Fase 3 (5A puro).
 *        * Compuerta 3 (Piloto Empirico en Particiones No-Locales `pi >= 3`):
 *          Si se superan las Compuertas 1 y 2, rota `q` de forma perezosa y ejecuta la
 *          cascada 5B en Fase 2 monitoreando el trabajo efectivo en cuartos de vector
 *          por candidato (`pilot_quarters / pilot_vecs`) tras >= 2 particiones no-locales.
 *          Si supera el umbral de amortizacion frente a 5A, aborta 5B y completa en Fase 3.
 */
template <bool AuditMode, typename Observer>
static inline void eval_partition_5a_step(const IVFPartitionedIndex& idx,
                                          const float* q,
                                          const uint32_t* el_ivf_ids,
                                          uint32_t e_start,
                                          uint32_t e_end,
                                          float lb_part_sq,
                                          uint32_t k,
                                          TopKHeap& heap,
                                          uint32_t& quarter_evals,
                                          uint32_t& parts_visited,
                                          uint64_t& total_bytes,
                                          Observer& obs) {
    if (heap.size == k) {
        if (lb_part_sq > heap.threshold_sq) {
            if constexpr (AuditMode) {
                for (uint32_t e = e_start; e < e_end; ++e) {
                    obs.on_checked(1);
                    obs.on_pruned(PruneStage::PART, el_ivf_ids[e], static_cast<double>(lb_part_sq));
                }
            }
            return;
        }
    }

    parts_visited++;
    const uint32_t dim = idx.dim;
    const float* base_orig = idx.part_orig_vectors;
    const uint32_t cand_cnt = e_end - e_start;
    quarter_evals += cand_cnt * 4U;
    total_bytes += static_cast<uint64_t>(cand_cnt) * (dim * sizeof(float) + sizeof(uint32_t));
    if constexpr (AuditMode) {
        obs.on_checked(cand_cnt);
        obs.on_evaluated_orig(cand_cnt);
    }

    uint32_t e = e_start;
    for (; e + 4 <= e_end; e += 4) {
        uint32_t iv0 = el_ivf_ids[e];
        uint32_t iv1 = el_ivf_ids[e + 1];
        uint32_t iv2 = el_ivf_ids[e + 2];
        uint32_t iv3 = el_ivf_ids[e + 3];

        const float* v0 = base_orig + static_cast<size_t>(iv0) * dim;
        const float* v1 = base_orig + static_cast<size_t>(iv1) * dim;
        const float* v2 = base_orig + static_cast<size_t>(iv2) * dim;
        const float* v3 = base_orig + static_cast<size_t>(iv3) * dim;

        __m128 d4 = l2_sq_avx2_4x_aos(q, v0, v1, v2, v3, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
        if (mask != 0) {
            if (mask & 1) heap.push(idx.part_aos_ids[iv0], l2_hito9_ref(q, v0, dim));
            if (mask & 2) heap.push(idx.part_aos_ids[iv1], l2_hito9_ref(q, v1, dim));
            if (mask & 4) heap.push(idx.part_aos_ids[iv2], l2_hito9_ref(q, v2, dim));
            if (mask & 8) heap.push(idx.part_aos_ids[iv3], l2_hito9_ref(q, v3, dim));
        }
    }
    for (; e < e_end; ++e) {
        uint32_t iv = el_ivf_ids[e];
        const float* vp = base_orig + static_cast<size_t>(iv) * dim;
        float d = l2_sq_avx2_1x_aos(q, vp, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d <= gate) {
            heap.push(idx.part_aos_ids[iv], l2_hito9_ref(q, vp, dim));
        }
    }
}

template <bool AuditMode, typename Observer>
static __attribute__((noinline)) void run_plan6_phase2_5b_noinline(
    const IVFPartitionedIndex& idx,
    const float* q,
    const PartitionOrder* p_order,
    const uint32_t* el_ivf_ids,
    const uint32_t* part_offs,
    double max_q_ratio,
    uint32_t& pi,
    TopKHeap& heap,
    uint32_t& quarter_evals,
    uint32_t& parts_visited,
    uint64_t& total_bytes,
    QueryExecutionMetrics& metrics,
    Observer& obs) {
    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;

    alignas(64) float q_rot[64];
    float q_mean_norm = 0.0f;
    rotate_centered_query_avx2(idx.ortho_basis, q, idx.global_mean, q_rot, q_mean_norm, dim);

    uint32_t pilot_parts = 0;
    uint32_t pilot_vecs = 0;
    uint32_t pilot_quarters = 0;
    bool pilot_checked = false;

    for (; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        if (p_order[pi].lb_part_sq > heap.threshold_sq) {
            if constexpr (AuditMode) {
                for (uint32_t e = e_start; e < e_end; ++e) {
                    obs.on_checked(1);
                    obs.on_pruned(PruneStage::PART, el_ivf_ids[e], static_cast<double>(p_order[pi].lb_part_sq));
                }
            }
            continue;
        }

        const uint32_t part_cand_count = e_end - e_start;
        const uint32_t quarters_before_part = quarter_evals;

        float dc_low = p_order[pi].dc_low;
        float dc_high = p_order[pi].dc_high;
        const GeometricPartition& gp = idx.partitions[pid];

        /* Etapa 2 (Radial): Poda Exacta Conservadora de Cascara Radial [r_low, r_high] */
        float r_low = 0.0f, r_high = 0.0f, r_pad = 0.0f;
        compute_radial_shell_interval(dc_low, dc_high, gp.max_radius, heap.threshold_sq, r_low, r_high, r_pad);

        uint32_t orig_e_start = e_start;
        uint32_t orig_e_end = e_end;

        if (r_low > gp.min_radius) {
            uint32_t lo = e_start, hi = e_end;
            while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                if (idx.part_aos_radii[el_ivf_ids[mid]] < r_low) lo = mid + 1;
                else hi = mid;
            }
            e_start = lo;
        }
        if (r_high < gp.max_radius && e_start < e_end) {
            uint32_t lo = e_start, hi = e_end;
            while (lo < hi) {
                uint32_t mid = lo + ((hi - lo) >> 1);
                if (idx.part_aos_radii[el_ivf_ids[mid]] <= r_high) lo = mid + 1;
                else hi = mid;
            }
            e_end = lo;
        }

        if constexpr (AuditMode) {
            const double one_m_g66_down = one_minus_gamma_down_f64(66);
            for (uint32_t e = orig_e_start; e < e_start; ++e) {
                uint32_t iv = el_ivf_ids[e];
                obs.on_checked(1);
                double r_x = static_cast<double>(idx.part_aos_radii[iv]);
                double raw_rad = sub_down_f64(sub_down_f64(static_cast<double>(dc_low), r_x),
                                              static_cast<double>(r_pad));
                float lb_rad_sq = round_down_f32(mul_down_f64(mul_down_f64(raw_rad, raw_rad), one_m_g66_down));
                obs.on_pruned(PruneStage::RADIAL, iv, static_cast<double>(lb_rad_sq));
            }
            for (uint32_t e = e_end; e < orig_e_end; ++e) {
                uint32_t iv = el_ivf_ids[e];
                obs.on_checked(1);
                double r_x = static_cast<double>(idx.part_aos_radii[iv]);
                double raw_rad = sub_down_f64(sub_down_f64(r_x, static_cast<double>(dc_high)),
                                              static_cast<double>(r_pad));
                float lb_rad_sq = round_down_f32(mul_down_f64(mul_down_f64(raw_rad, raw_rad), one_m_g66_down));
                obs.on_pruned(PruneStage::RADIAL, iv, static_cast<double>(lb_rad_sq));
            }
        }

        if (e_start < e_end) {
            parts_visited++;
            alignas(64) float q_rot_p[64];
            const float* rot_c = idx.rotated_centroids + static_cast<size_t>(pid) * 64U;
            __m256 v_tail16_31 = _mm256_setzero_ps();
            __m256 v_tail32_63 = _mm256_setzero_ps();

            for (uint32_t d = 0; d < 16; d += 8) {
                __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
                _mm256_store_ps(q_rot_p + d, diff);
            }
            for (uint32_t d = 16; d < 32; d += 8) {
                __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
                _mm256_store_ps(q_rot_p + d, diff);
                v_tail16_31 = _mm256_fmadd_ps(diff, diff, v_tail16_31);
            }
            for (uint32_t d = 32; d < 64; d += 8) {
                __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
                _mm256_store_ps(q_rot_p + d, diff);
                v_tail32_63 = _mm256_fmadd_ps(diff, diff, v_tail32_63);
            }

            float qt32_sq = hsum256_ps_fast(v_tail32_63);
            float qt16_31_sq = hsum256_ps_fast(v_tail16_31);
            float q_tail32_p = sqrtf(qt32_sq);
            float q_tail48_p = sqrtf(qt32_sq + qt16_31_sq);

            PartitionCertifiedBounds bounds = compute_partition_certified_bounds(
                q_mean_norm, idx.centroid_mean_norms[pid], dc_high, gp.max_radius, idx.ortho_defect
            );

            uint32_t e = e_start;
            for (; e + 4 <= e_end; e += 4) {
                eval_4x_pca_progressive_unified<AuditMode>(
                    q, q_rot_p, q_tail48_p, q_tail32_p,
                    bounds, idx,
                    el_ivf_ids[e], el_ivf_ids[e + 1], el_ivf_ids[e + 2], el_ivf_ids[e + 3],
                    heap, quarter_evals, total_bytes, obs
                );
            }
            for (; e < e_end; ++e) {
                eval_1x_pca_progressive_unified<AuditMode>(
                    q, q_rot_p, q_tail48_p, q_tail32_p,
                    bounds, idx,
                    el_ivf_ids[e],
                    heap, quarter_evals, total_bytes, obs
                );
            }
        }

        /* Compuerta 3: Piloto empirico de 2 particiones al ingresar a la Zona Externa */
        if (!pilot_checked) {
            pilot_parts++;
            pilot_vecs += part_cand_count;
            pilot_quarters += (quarter_evals - quarters_before_part);
            if (pilot_parts >= 2U && pilot_vecs >= 120U) {
                pilot_checked = true;
                metrics.plan6_gate3_pilot_work = static_cast<float>(pilot_quarters) / static_cast<float>(pilot_vecs > 0 ? pilot_vecs : 1);
                metrics.plan6_gate3_max_q_ratio = static_cast<float>(max_q_ratio);
                if (static_cast<double>(pilot_quarters) > max_q_ratio * static_cast<double>(pilot_vecs)) {
                    metrics.plan6_gate3_aborted = 1;
                    ++pi;
                    break;
                }
            }
        }
    }
}

template <bool AuditMode, typename Observer>
static void run_plan6_adaptive_unified_core(const IVFPartitionedIndex& idx,
                                            const float* q,
                                            const QueryPredicate& pred,
                                            uint32_t k,
                                            TopKHeap& heap,
                                            QueryExecutionMetrics& metrics,
                                            TimerScope* opt_ts,
                                            Observer& obs) {
    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;

    /* Fase Prep: Identica al baseline Plan 5A (sin pagar rotacion PCA por adelantado) */
    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        const GeometricPartition& gp = idx.partitions[p];
        PartitionDistanceBounds pdb = compute_partition_distance_bounds(d2, gp.min_radius, gp.max_radius);
        p_order[p] = {p, pdb.dc, pdb.dc_low, pdb.dc_high, pdb.lb_part_sq};
    }
    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    if (opt_ts) {
        opt_ts->mark_prep_done();
    }

    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();

    uint32_t quarter_evals = 0;
    uint32_t parts_visited = 0;
    uint64_t total_bytes = 0;

    /*
     * Fase 1 (Sembrado y Zona Interna en 5A):
     * - Si 2*k < m_est (la bola Top-K cabe holgadamente en una sola particion, ej. Top-10),
     *   basta poblar la primera particion semilla (`pi = 0`) con 5A.
     * - Si 2*k >= m_est (la bola Top-K abarca multiples particiones, ej. Top-100),
     *   barre con 5A la Zona Interna (`dc^2 <= 1.55 * theta_K^2`) donde la bola Top-K
     *   se solapa con los nucleos de particion y provocaria caidas escalares a PCA32/PCA64.
     */
    const float m_est = static_cast<float>(pred.eligible_count) / static_cast<float>(n_parts);
    const float inner_beta = (static_cast<float>(2U * k) < m_est) ? 0.0f : 1.55f;
    const uint32_t min_seed_parts = (static_cast<float>(2U * k) < m_est) ? 1U : 3U;

    uint32_t pi = 0;
    uint32_t seed_nonempty_parts = 0;
    for (; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        if (heap.size == k && seed_nonempty_parts >= min_seed_parts) {
            float dc_sq = p_order[pi].dc * p_order[pi].dc;
            if (dc_sq > inner_beta * heap.threshold_sq) {
                break;
            }
        }

        eval_partition_5a_step<AuditMode>(
            idx, q, el_ivf_ids, e_start, e_end, p_order[pi].lb_part_sq,
            k, heap, quarter_evals, parts_visited, total_bytes, obs
        );
        seed_nonempty_parts++;
    }

    /*
     * Evaluacion determinista del Modelo de Costo de 3 Compuertas al ingresar a la Zona Externa
     */
    bool use_5b_cascade = false;
    double max_q_ratio = 1.95;
    if (pi < n_parts && heap.size == k) {
        uint32_t p_surv_5a = 0;
        uint32_t n_surv_5a = 0;
        for (uint32_t j = pi; j < n_parts; ++j) {
            uint32_t pid_j = p_order[j].part_id;
            uint32_t cnt_j = part_offs[pid_j + 1] - part_offs[pid_j];
            if (cnt_j > 0 && p_order[j].lb_part_sq <= heap.threshold_sq) {
                p_surv_5a++;
                n_surv_5a += cnt_j;
            }
        }

        /* Compuerta 1 (Suficiencia de Volumen Sobreviviente tras 5A): p_surv_5a > 32 y n_surv_5a > 4000 */
        metrics.plan6_gate1_p_surv = static_cast<float>(p_surv_5a);
        metrics.plan6_gate1_n_surv = static_cast<float>(n_surv_5a);
        if (p_surv_5a > 32U && n_surv_5a > 4000U) {
            float d_med = p_order[n_parts >> 1].dc;
            float d_med_sq = d_med * d_med + 1.0e-12f;
            float d0_sq = p_order[0].dc * p_order[0].dc;
            float gamma_0 = d0_sq / d_med_sq;
            float eta_pot = heap.threshold_sq / d_med_sq;
            float m_bar = static_cast<float>(n_surv_5a) / static_cast<float>(p_surv_5a);
            metrics.plan6_gate2_gamma0 = gamma_0;
            metrics.plan6_gate2_eta_pot = eta_pot;

            /*
             * Compuerta 2:
             * - gamma_0 <= 0.45f separa consultas concentradas en un cluster de consultas
             *   uniformes/difusas en la esfera 64D (donde 5B nunca amortiza su sobrecarga).
             * - eta_limit depende de la densidad media por particion m_bar = n_surv_5a / p_surv_5a
             *   que amortiza el costo fijo O(D) de preparacion PCA por particion en 5B.
             */
            float eta_limit = (m_bar < 95.0f) ? 0.38f : 0.56f;
            if (gamma_0 <= 0.45f && eta_pot <= eta_limit) {
                use_5b_cascade = true;
                metrics.plan6_selected_cascade = 1;
                max_q_ratio = (m_bar >= 150.0f) ? 2.15 : 1.95;
            }
        }
    }

    /*
     * Fase 2 (Cascada Certificada 5B aislada en funcion noinline para evitar presion
     * de registros XMM6-XMM15 sobre la ruta rapida 5A)
     */
    if (use_5b_cascade) {
        run_plan6_phase2_5b_noinline<AuditMode>(
            idx, q, p_order, el_ivf_ids, part_offs, max_q_ratio,
            pi, heap, quarter_evals, parts_visited, total_bytes, metrics, obs
        );
    }

    /*
     * Fase 3 (Barrido Limpio 5A para las particiones restantes)
     */
    for (; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        eval_partition_5a_step<AuditMode>(
            idx, q, el_ivf_ids, e_start, e_end, p_order[pi].lb_part_sq,
            k, heap, quarter_evals, parts_visited, total_bytes, obs
        );
    }

    metrics.dists_evaluated = (quarter_evals + 2) / 4;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = parts_visited;
    metrics.partitions_total = n_parts;
    metrics.bytes_read = total_bytes;
}

void execute_plan6_ivf_adaptive_cost_model_avx2(const IVFPartitionedIndex& idx,
                                                const float* q,
                                                const QueryPredicate& pred,
                                                uint32_t k,
                                                Neighbor* out_results,
                                                uint32_t* out_count,
                                                QueryExecutionMetrics& metrics) {
    memset(&metrics, 0, sizeof(metrics));
    TimerScope ts;
    ts.start();

    /* Validacion intacta del contrato de ejecucion IEEE-754 */
    const bool mxcsr_ok = check_ieee754_mxcsr_state();
    if (!idx.certified_Contract_valid ||
        !mxcsr_ok ||
        !check_vector_domain_certified(q, idx.dim)) {
        handle_ivf_non_nominal_contract_or_fallback(idx, q, pred, k, out_results, out_count, metrics, ts, mxcsr_ok);
        return;
    }
    metrics.status = QueryExecutionStatus::OK_CERTIFIED;

    TopKHeap heap;
    NoOpPruneObserver noop_obs;
    run_plan6_adaptive_unified_core<false>(idx, q, pred, k, heap, metrics, &ts, noop_obs);

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
}

void audit_plan6_adaptive_soundness(const IVFPartitionedIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    const Neighbor* true_topk,
                                    uint32_t true_topk_count,
                                    PruningAuditReport& report) {
    if (!idx.certified_Contract_valid ||
        !check_ieee754_mxcsr_state() ||
        !check_vector_domain_certified(q, idx.dim)) {
        report.contract_fallbacks++;
        uint32_t n_el = static_cast<uint32_t>(idx.eligible_ivf_ids_per_sel[pred.sel_index].size());
        report.total_eligible_checked += n_el;
        report.evaluated_full_64d += n_el;
        return;
    }

    struct AuditObserver6 {
        const IVFPartitionedIndex& idx;
        const float* q;
        const Neighbor* true_topk;
        uint32_t true_topk_count;
        PruningAuditReport& report;

        inline bool is_in_true_topk(uint32_t orig_id) const {
            for (uint32_t t = 0; t < true_topk_count; ++t) {
                if (true_topk[t].id == orig_id) return true;
            }
            return false;
        }

        inline void on_checked(uint32_t cnt) {
            report.total_eligible_checked += cnt;
        }

        inline void on_evaluated_orig(uint32_t cnt) {
            report.evaluated_full_64d += cnt;
        }

        inline void on_pruned(PruneStage stage, uint32_t iv, double lb_sq) {
            uint64_t* p_stage_cnt = nullptr;
            uint64_t* p_strict    = nullptr;
            uint64_t* p_margin    = nullptr;
            uint64_t* p_exceed    = nullptr;

            switch (stage) {
                case PruneStage::PART:
                    p_stage_cnt = &report.pruned_by_partition;
                    p_strict    = &report.strict_violations_partition;
                    p_margin    = &report.diag_margin_events_partition;
                    p_exceed    = &report.diag_exceed_violations_partition;
                    break;
                case PruneStage::RADIAL:
                    p_stage_cnt = &report.pruned_by_radial_shell;
                    p_strict    = &report.strict_violations_radial;
                    p_margin    = &report.diag_margin_events_radial;
                    p_exceed    = &report.diag_exceed_violations_radial;
                    break;
                case PruneStage::PCA16:
                    p_stage_cnt = &report.pruned_by_pca_16d;
                    p_strict    = &report.strict_violations_pca16;
                    p_margin    = &report.diag_margin_events_pca16;
                    p_exceed    = &report.diag_exceed_violations_pca16;
                    break;
                case PruneStage::PCA32:
                    p_stage_cnt = &report.pruned_by_pca_32d;
                    p_strict    = &report.strict_violations_pca32;
                    p_margin    = &report.diag_margin_events_pca32;
                    p_exceed    = &report.diag_exceed_violations_pca32;
                    break;
                case PruneStage::PCA64:
                    p_stage_cnt = &report.pruned_by_pca_64d;
                    p_strict    = &report.strict_violations_pca64;
                    p_margin    = &report.diag_margin_events_pca64;
                    p_exceed    = &report.diag_exceed_violations_pca64;
                    break;
            }

            (*p_stage_cnt)++;

            uint32_t orig_id = idx.part_aos_ids[iv];
            const float* base_orig = idx.part_orig_vectors;
            double orig_d2 = static_cast<double>(l2_hito9_ref(q, base_orig + static_cast<size_t>(iv) * idx.dim, idx.dim));

            double signed_excess = lb_sq - orig_d2;
            if (signed_excess > report.max_signed_excess) {
                report.max_signed_excess = signed_excess;
            }

            if (signed_excess > 0.0) {
                (*p_strict)++;
                if (signed_excess > report.max_positive_excess) {
                    report.max_positive_excess = signed_excess;
                }
                double diag_thresh = orig_d2 * ieee754_gamma(66);
                if (signed_excess <= diag_thresh) {
                    (*p_margin)++;
                } else {
                    (*p_exceed)++;
                }
            }

            if (lb_sq > 0.0) {
                double ratio = orig_d2 / lb_sq;
                if (ratio < report.min_ratio_orig_over_lb) {
                    report.min_ratio_orig_over_lb = ratio;
                }
            }
            if (is_in_true_topk(orig_id)) {
                report.false_prunings_topk++;
            }
        }
    };

    TopKHeap heap;
    QueryExecutionMetrics dummy_metrics{};
    AuditObserver6 audit_obs{idx, q, true_topk, true_topk_count, report};
    run_plan6_adaptive_unified_core<true>(idx, q, pred, k, heap, dummy_metrics, nullptr, audit_obs);
}

void evaluate_single_partition_5a(const IVFPartitionedIndex& idx,
                                  const float* q,
                                  uint32_t pid,
                                  float lb_part_sq,
                                  const QueryPredicate& pred,
                                  uint32_t k,
                                  TopKHeap& heap,
                                  PartitionStepMetrics& out_met) {
    memset(&out_met, 0, sizeof(out_met));
    out_met.part_id = pid;

    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    uint32_t e_start = part_offs[pid];
    uint32_t e_end = part_offs[pid + 1];
    out_met.cand_count = e_end - e_start;
    if (e_start >= e_end) return;

    if (heap.size == k && lb_part_sq > heap.threshold_sq) {
        out_met.pruned_by_part = true;
        return;
    }

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    uint32_t quarter_evals = 0;
    uint32_t parts_visited = 0;
    uint64_t total_bytes = 0;
    NoOpPruneObserver noop_obs;

    LARGE_INTEGER freq, c0, c1;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&c0);
    _mm_lfence();

    eval_partition_5a_step<false>(
        idx, q, el_ivf_ids, e_start, e_end, lb_part_sq,
        k, heap, quarter_evals, parts_visited, total_bytes, noop_obs
    );

    _mm_lfence();
    QueryPerformanceCounter(&c1);
    _mm_lfence();

    out_met.time_ns = static_cast<double>((c1.QuadPart - c0.QuadPart) * 1000000000ULL) / static_cast<double>(freq.QuadPart);
    out_met.dists_evaluated = (quarter_evals + 2) / 4;
}

void evaluate_single_partition_5b(const IVFPartitionedIndex& idx,
                                  const float* q,
                                  const float* q_rot,
                                  float q_mean_norm,
                                  uint32_t pid,
                                  float dc_low,
                                  float dc_high,
                                  float lb_part_sq,
                                  const QueryPredicate& pred,
                                  uint32_t k,
                                  TopKHeap& heap,
                                  PartitionStepMetrics& out_met) {
    memset(&out_met, 0, sizeof(out_met));
    out_met.part_id = pid;

    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    uint32_t e_start = part_offs[pid];
    uint32_t e_end = part_offs[pid + 1];
    out_met.cand_count = e_end - e_start;
    if (e_start >= e_end) return;

    if (heap.size == k && lb_part_sq > heap.threshold_sq) {
        out_met.pruned_by_part = true;
        return;
    }

    const GeometricPartition& gp = idx.partitions[pid];
    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();

    LARGE_INTEGER freq, c0, c1;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&c0);
    _mm_lfence();

    float r_low = 0.0f, r_high = 0.0f, r_pad = 0.0f;
    compute_radial_shell_interval(dc_low, dc_high, gp.max_radius, heap.threshold_sq, r_low, r_high, r_pad);

    if (r_low > gp.min_radius) {
        uint32_t lo = e_start, hi = e_end;
        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1);
            if (idx.part_aos_radii[el_ivf_ids[mid]] < r_low) lo = mid + 1;
            else hi = mid;
        }
        e_start = lo;
    }
    if (r_high < gp.max_radius && e_start < e_end) {
        uint32_t lo = e_start, hi = e_end;
        while (lo < hi) {
            uint32_t mid = lo + ((hi - lo) >> 1);
            if (idx.part_aos_radii[el_ivf_ids[mid]] <= r_high) lo = mid + 1;
            else hi = mid;
        }
        e_end = lo;
    }

    uint32_t quarter_evals = 0;
    uint64_t total_bytes = 0;
    NoOpPruneObserver noop_obs;

    if (e_start < e_end) {
        alignas(64) float q_rot_p[64];
        const float* rot_c = idx.rotated_centroids + static_cast<size_t>(pid) * 64U;
        __m256 v_tail16_31 = _mm256_setzero_ps();
        __m256 v_tail32_63 = _mm256_setzero_ps();

        for (uint32_t d = 0; d < 16; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
            _mm256_store_ps(q_rot_p + d, diff);
        }
        for (uint32_t d = 16; d < 32; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
            _mm256_store_ps(q_rot_p + d, diff);
            v_tail16_31 = _mm256_fmadd_ps(diff, diff, v_tail16_31);
        }
        for (uint32_t d = 32; d < 64; d += 8) {
            __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
            _mm256_store_ps(q_rot_p + d, diff);
            v_tail32_63 = _mm256_fmadd_ps(diff, diff, v_tail32_63);
        }

        float qt32_sq = hsum256_ps_fast(v_tail32_63);
        float qt16_31_sq = hsum256_ps_fast(v_tail16_31);
        float q_tail32_p = sqrtf(qt32_sq);
        float q_tail48_p = sqrtf(qt32_sq + qt16_31_sq);

        PartitionCertifiedBounds bounds = compute_partition_certified_bounds(
            q_mean_norm, idx.centroid_mean_norms[pid], dc_high, gp.max_radius, idx.ortho_defect
        );

        uint32_t e = e_start;
        for (; e + 4 <= e_end; e += 4) {
            eval_4x_pca_progressive_unified<false>(
                q, q_rot_p, q_tail48_p, q_tail32_p,
                bounds, idx,
                el_ivf_ids[e], el_ivf_ids[e + 1], el_ivf_ids[e + 2], el_ivf_ids[e + 3],
                heap, quarter_evals, total_bytes, noop_obs
            );
        }
        for (; e < e_end; ++e) {
            eval_1x_pca_progressive_unified<false>(
                q, q_rot_p, q_tail48_p, q_tail32_p,
                bounds, idx,
                el_ivf_ids[e],
                heap, quarter_evals, total_bytes, noop_obs
            );
        }
    }

    _mm_lfence();
    QueryPerformanceCounter(&c1);
    _mm_lfence();

    out_met.time_ns = static_cast<double>((c1.QuadPart - c0.QuadPart) * 1000000000ULL) / static_cast<double>(freq.QuadPart);
    out_met.dists_evaluated = (quarter_evals + 2) / 4;
}

void execute_oracle_plan_end_to_end(
    const IVFPartitionedIndex& idx,
    const float* q,
    const uint32_t* sorted_pids,
    const float* sorted_lb_part_sq,
    const float* sorted_dc_low,
    const float* sorted_dc_high,
    const uint8_t* decision_5b_mask,
    bool oracle_needs_pca_rot,
    const QueryPredicate& pred,
    uint32_t k,
    Neighbor* out_results,
    uint32_t* out_count,
    double& out_total_us) {

    LARGE_INTEGER freq, c0, c1;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&c0);
    _mm_lfence();

    const uint32_t n_parts = idx.num_partitions;
    const uint32_t dim = idx.dim;
    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();

    alignas(64) float q_rot[64];
    float q_mean_norm = 0.0f;
    if (oracle_needs_pca_rot) {
        alignas(64) float diff[64];
        double norm_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            diff[d] = q[d] - idx.global_mean[d];
            norm_sq += static_cast<double>(diff[d]) * diff[d];
        }
        q_mean_norm = round_up_f32(sqrt_up_f64(norm_sq));
        for (uint32_t r = 0; r < dim; ++r) {
            const float* row = idx.ortho_basis + static_cast<size_t>(r) * dim;
            float sum = 0.0f;
            for (uint32_t c = 0; c < dim; ++c) sum += row[c] * diff[c];
            q_rot[r] = sum;
        }
    }

    TopKHeap heap;
    heap.init(k);
    uint32_t dummy_q = 0;
    uint32_t dummy_v = 0;
    uint64_t dummy_b = 0;
    NoOpPruneObserver noop_obs;

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = sorted_pids[pi];
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        float lb_p = sorted_lb_part_sq[pi];
        if (heap.size == k && lb_p > heap.threshold_sq) {
            continue;
        }

        if (decision_5b_mask[pi] == 0) {
            eval_partition_5a_step<false>(
                idx, q, el_ivf_ids, e_start, e_end, lb_p,
                k, heap, dummy_q, dummy_v, dummy_b, noop_obs
            );
        } else {
            const GeometricPartition& gp = idx.partitions[pid];
            float dc_low = sorted_dc_low[pi];
            float dc_high = sorted_dc_high[pi];
            float r_low = 0.0f, r_high = 0.0f, r_pad = 0.0f;
            compute_radial_shell_interval(dc_low, dc_high, gp.max_radius, heap.threshold_sq, r_low, r_high, r_pad);

            uint32_t b_start = e_start, b_end = e_end;
            if (r_low > gp.min_radius) {
                uint32_t lo = b_start, hi = b_end;
                while (lo < hi) {
                    uint32_t mid = lo + ((hi - lo) >> 1);
                    if (idx.part_aos_radii[el_ivf_ids[mid]] < r_low) lo = mid + 1;
                    else hi = mid;
                }
                b_start = lo;
            }
            if (r_high < gp.max_radius && b_start < b_end) {
                uint32_t lo = b_start, hi = b_end;
                while (lo < hi) {
                    uint32_t mid = lo + ((hi - lo) >> 1);
                    if (idx.part_aos_radii[el_ivf_ids[mid]] <= r_high) lo = mid + 1;
                    else hi = mid;
                }
                b_end = lo;
            }

            if (b_start < b_end) {
                alignas(64) float q_rot_p[64];
                const float* rot_c = idx.rotated_centroids + static_cast<size_t>(pid) * 64U;
                __m256 v_tail16_31 = _mm256_setzero_ps();
                __m256 v_tail32_63 = _mm256_setzero_ps();
                for (uint32_t d = 0; d < 16; d += 8) {
                    __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
                    _mm256_store_ps(q_rot_p + d, diff);
                }
                for (uint32_t d = 16; d < 32; d += 8) {
                    __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
                    _mm256_store_ps(q_rot_p + d, diff);
                    v_tail16_31 = _mm256_fmadd_ps(diff, diff, v_tail16_31);
                }
                for (uint32_t d = 32; d < 64; d += 8) {
                    __m256 diff = _mm256_sub_ps(_mm256_load_ps(q_rot + d), _mm256_load_ps(rot_c + d));
                    _mm256_store_ps(q_rot_p + d, diff);
                    v_tail32_63 = _mm256_fmadd_ps(diff, diff, v_tail32_63);
                }
                float qt32_sq = hsum256_ps_fast(v_tail32_63);
                float qt16_31_sq = hsum256_ps_fast(v_tail16_31);
                float q_tail32_p = sqrtf(qt32_sq);
                float q_tail48_p = sqrtf(qt32_sq + qt16_31_sq);

                PartitionCertifiedBounds bounds = compute_partition_certified_bounds(
                    q_mean_norm, idx.centroid_mean_norms[pid], dc_high, gp.max_radius, idx.ortho_defect
                );

                uint32_t be = b_start;
                for (; be + 4 <= b_end; be += 4) {
                    eval_4x_pca_progressive_unified<false>(
                        q, q_rot_p, q_tail48_p, q_tail32_p,
                        bounds, idx,
                        el_ivf_ids[be], el_ivf_ids[be+1], el_ivf_ids[be+2], el_ivf_ids[be+3],
                        heap, dummy_q, dummy_b, noop_obs
                    );
                }
                for (; be < b_end; ++be) {
                    eval_1x_pca_progressive_unified<false>(
                        q, q_rot_p, q_tail48_p, q_tail32_p,
                        bounds, idx,
                        el_ivf_ids[be],
                        heap, dummy_q, dummy_b, noop_obs
                    );
                }
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    _mm_lfence();
    QueryPerformanceCounter(&c1);
    _mm_lfence();

    out_total_us = static_cast<double>((c1.QuadPart - c0.QuadPart) * 1000000ULL) / static_cast<double>(freq.QuadPart);
}

} // namespace alg_flat
