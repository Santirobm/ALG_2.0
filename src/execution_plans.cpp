#include "../include/execution_plans.h"
#include "../include/simd_kernels.h"
#include <queue>
#include <vector>
#include <algorithm>
#include <cmath>

namespace alg_flat {

static constexpr float KERNEL_SUM_TREE_REL_EPS = 2.0e-5f;
static constexpr float PART_FP_REL_EPS = 2.0e-5f;

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
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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

/* Rota el vector centrado (q - mu) por Q^T */
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
    out_q_mean_norm = sqrtf(hsum256_ps_fast(v_norm_sq));

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
    float dc;       /* ||q - c_p||_2 en espacio original */
    float dc_low;   /* Cota inferior conservadora de ||q - c_p||_2 */
    float dc_high;  /* Cota superior conservadora de ||q - c_p||_2 */
    float lb;       /* Cota inferior conservadora de distancia a la particion */
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
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        float dc_low = dc * (1.0f - PART_FP_REL_EPS);
        float dc_high = dc * (1.0f + PART_FP_REL_EPS);
        const GeometricPartition& gp = idx.partitions[p];
        float lb = 0.0f;
        if (dc_low > gp.max_radius) lb = dc_low - gp.max_radius;
        else if (dc_high < gp.min_radius) lb = gp.min_radius - dc_high;
        p_order[p] = {p, dc, dc_low, dc_high, lb};
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
            float safety_rho = sqrtf(heap.threshold_sq) * (1.0f + PART_FP_REL_EPS) + 1.0e-6f;
            if (p_order[pi].lb > safety_rho) {
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
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
            float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + KERNEL_SUM_TREE_REL_EPS));
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
 * Kernel 4x1 Progresivo en Base Rotada PCA Centrada por Particion:
 *   - q_rot_p = Q^T * (q - c_p)
 *   - v_rot_p = Q^T * (x - c_p)
 *   - q_tail48_p = ||Q^T(q - c_p)[16..63]||_2  vs  part_aos_tail48_norm = ||Q^T(x - c_p)[16..63]||_2
 *   - q_tail32_p = ||Q^T(q - c_p)[32..63]||_2  vs  part_aos_tail32_norm = ||Q^T(x - c_p)[32..63]||_2
 *   - Distancia final de sobrevivientes evaluada en el espacio original float32 (`q` vs `part_orig_vectors`).
 */
static inline void eval_4x_pca_progressive_certified(const float* q_orig,
                                                     const float* q_rot_p,
                                                     float q_tail48_p,
                                                     float q_tail32_p,
                                                     float fp_slack_p,
                                                     float sq_slack_p,
                                                     float scale_safe,
                                                     const IVFPartitionedIndex& idx,
                                                     uint32_t iv0, uint32_t iv1, uint32_t iv2, uint32_t iv3,
                                                     TopKHeap& heap,
                                                     uint32_t& quarter_evals,
                                                     uint64_t& bytes_read) {
    const float* v0 = idx.part_aos_vectors + static_cast<size_t>(iv0) * 64U;
    const float* v1 = idx.part_aos_vectors + static_cast<size_t>(iv1) * 64U;
    const float* v2 = idx.part_aos_vectors + static_cast<size_t>(iv2) * 64U;
    const float* v3 = idx.part_aos_vectors + static_cast<size_t>(iv3) * 64U;

    /* Etapa 1: Primeras 16 dimensiones principales de (q - c_p) vs (x - c_p) */
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

    /*
     * Cota inferior conservadora en 16D + Cola 48D centrada en el centroide de la particion:
     * LB_tail48 = max(0, | ||q_tail48 - c_tail48|| - ||x_tail48 - c_tail48|| | - fp_slack_p)
     */
    __m128 b4 = _mm_set_ps(idx.part_aos_tail48_norm[iv3],
                           idx.part_aos_tail48_norm[iv2],
                           idx.part_aos_tail48_norm[iv1],
                           idx.part_aos_tail48_norm[iv0]);
    __m128 raw_diff = _mm_sub_ps(_mm_set1_ps(q_tail48_p), b4);
    __m128 abs_diff = _mm_andnot_ps(_mm_set1_ps(-0.0f), raw_diff);
    __m128 lb_tail = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(abs_diff, _mm_set1_ps(fp_slack_p)));
    __m128 d16_safe = _mm_max_ps(_mm_setzero_ps(), _mm_sub_ps(d16_4, _mm_set1_ps(sq_slack_p)));
    __m128 lb64_4 = _mm_mul_ps(_mm_fmadd_ps(lb_tail, lb_tail, d16_safe), _mm_set1_ps(scale_safe));

    int mask16 = _mm_movemask_ps(_mm_cmple_ps(lb64_4, _mm_set1_ps(heap.threshold_sq)));
    if (mask16 == 0) {
        return; /* Los 4 vectores quedan descartados con garantia matematica tras leer solo 1 cache line */
    }

    alignas(16) float p16[4];
    _mm_store_ps(p16, d16_4);
    const uint32_t ivs[4] = {iv0, iv1, iv2, iv3};
    const float* v_ptrs[4] = {v0, v1, v2, v3};

    for (int lane = 0; lane < 4; ++lane) {
        if (!(mask16 & (1 << lane))) continue;
        uint32_t iv = ivs[lane];
        const float* vp = v_ptrs[lane];

        /* Etapa 2: Dimensiones 16..31 (2da cache line) */
        __m256 qa = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 16), _mm256_load_ps(vp + 16));
        __m256 qb = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 24), _mm256_load_ps(vp + 24));
        float d32 = p16[lane] + hsum256_ps_fast(_mm256_fmadd_ps(qa, qa, _mm256_mul_ps(qb, qb)));
        quarter_evals += 1;
        bytes_read += 64;

        float abs_t32 = std::fabs(q_tail32_p - idx.part_aos_tail32_norm[iv]);
        float lb_t32 = (abs_t32 > fp_slack_p) ? (abs_t32 - fp_slack_p) : 0.0f;
        float d32_safe = (d32 > sq_slack_p) ? (d32 - sq_slack_p) : 0.0f;
        float lb64_stage2 = scale_safe * (d32_safe + lb_t32 * lb_t32);
        if (lb64_stage2 > heap.threshold_sq) {
            continue; /* Podado exactamente en 32D con cota conservadora */
        }

        /* Etapa 3: Completar dimensiones 32..63 en espacio rotado */
        __m256 qc = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 32), _mm256_load_ps(vp + 32));
        __m256 qd = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 40), _mm256_load_ps(vp + 40));
        __m256 acc_tail0 = _mm256_fmadd_ps(qc, qc, _mm256_mul_ps(qd, qd));

        __m256 qe = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 48), _mm256_load_ps(vp + 48));
        __m256 qf = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 56), _mm256_load_ps(vp + 56));
        __m256 acc_tail1 = _mm256_fmadd_ps(qe, qe, _mm256_mul_ps(qf, qf));

        float d64_rot = d32 + hsum256_ps_fast(_mm256_add_ps(acc_tail0, acc_tail1));
        quarter_evals += 2;
        bytes_read += 128;

        float d64_safe = (d64_rot > sq_slack_p) ? (d64_rot - sq_slack_p) : 0.0f;
        if (scale_safe * d64_safe <= heap.threshold_sq) {
            /* Nivel 2 de Certificacion: Recalcular distancia final exacta en el espacio original float32 */
            const float* orig_vp = idx.part_orig_vectors + static_cast<size_t>(iv) * 64U;
            float d_orig = l2_hito9_ref(q_orig, orig_vp, 64U);
            bytes_read += 256;
            heap.push(idx.part_aos_ids[iv], d_orig);
        }
    }
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

    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;
    alignas(64) float q_rot[64];
    float q_mean_norm = 0.0f;
    rotate_centered_query_avx2(idx.ortho_basis, q, idx.global_mean, q_rot, q_mean_norm, dim);

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        float dc_low = dc * (1.0f - PART_FP_REL_EPS);
        float dc_high = dc * (1.0f + PART_FP_REL_EPS);
        const GeometricPartition& gp = idx.partitions[p];
        float lb = 0.0f;
        if (dc_low > gp.max_radius) lb = dc_low - gp.max_radius;
        else if (dc_high < gp.min_radius) lb = gp.min_radius - dc_high;
        p_order[p] = {p, dc, dc_low, dc_high, lb};
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

    uint32_t quarter_evals = 0;
    uint32_t parts_visited = 0;
    uint64_t total_bytes = 0;

    const float scale_safe = 1.0f - 2.0f * (idx.ortho_defect + 2.0e-5f);

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        float dc = p_order[pi].dc;
        float dc_low = p_order[pi].dc_low;
        float dc_high = p_order[pi].dc_high;
        const GeometricPartition& gp = idx.partitions[pid];

        if (heap.size == k) {
            float rho_safe = sqrtf(heap.threshold_sq) * (1.0f + PART_FP_REL_EPS) + 1.0e-6f;
            if (p_order[pi].lb > rho_safe) {
                continue; /* Nivel 1: Particion entera descartada con garantia conservadora */
            }

            /* Nivel 2: Poda Exacta Conservadora de Cascara Radial [r_low, r_high] */
            float r_pad = PART_FP_REL_EPS * gp.max_radius + 1.0e-6f;
            float r_low = dc_low - rho_safe - r_pad;
            float r_high = dc_high + rho_safe + r_pad;

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
            if (e_start >= e_end) continue;
        }

        parts_visited++;
        uint32_t e = e_start;

        /* En la primera particion mientras se llena el heap, evaluar 4x1 directamente en espacio original */
        while (e + 4 <= e_end && heap.size < k) {
            uint32_t iv0 = el_ivf_ids[e];
            uint32_t iv1 = el_ivf_ids[e + 1];
            uint32_t iv2 = el_ivf_ids[e + 2];
            uint32_t iv3 = el_ivf_ids[e + 3];
            quarter_evals += 16;
            total_bytes += 4 * 256;

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
            heap.push(idx.part_aos_ids[iv], l2_hito9_ref(q, base_orig + static_cast<size_t>(iv) * dim, dim));
            e++;
        }
        if (e >= e_end) continue;

        /*
         * Calcular residuo de consulta respecto del centroide de esta particion en base rotada:
         *   q_rot_p = Q^T(q - mu) - Q^T(c_p - mu) = Q^T(q - c_p)
         * y sus normas de cola respecto del centroide de la particion:
         *   q_tail48_p = ||Q^T(q - c_p)[16..63]||_2
         *   q_tail32_p = ||Q^T(q - c_p)[32..63]||_2
         */
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

        float max_vec_norm = dc_high + gp.max_radius;
        float fp_slack_p = 8.0e-6f * (q_mean_norm + idx.centroid_mean_norms[pid] + gp.max_radius)
                         + idx.ortho_defect * max_vec_norm
                         + 1.0e-6f;
        float sq_slack_p = 2.0f * fp_slack_p * max_vec_norm + 1.0e-6f;

        /* Con el umbral rho_K activo, usar el kernel 4x1 Progresivo PCA Centrado por Particion */
        for (; e + 4 <= e_end; e += 4) {
            eval_4x_pca_progressive_certified(q, q_rot_p, q_tail48_p, q_tail32_p,
                                              fp_slack_p, sq_slack_p, scale_safe, idx,
                                              el_ivf_ids[e], el_ivf_ids[e + 1], el_ivf_ids[e + 2], el_ivf_ids[e + 3],
                                              heap, quarter_evals, total_bytes);
        }

        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            const float* vp = idx.part_aos_vectors + static_cast<size_t>(iv) * 64U;

            __m256 q0 = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 0), _mm256_load_ps(vp + 0));
            __m256 q1 = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 8), _mm256_load_ps(vp + 8));
            float d16 = hsum256_ps_fast(_mm256_fmadd_ps(q0, q0, _mm256_mul_ps(q1, q1)));
            quarter_evals += 1;
            total_bytes += 68;

            float abs_t48 = std::fabs(q_tail48_p - idx.part_aos_tail48_norm[iv]);
            float lb_t48 = (abs_t48 > fp_slack_p) ? (abs_t48 - fp_slack_p) : 0.0f;
            float d16_safe = (d16 > sq_slack_p) ? (d16 - sq_slack_p) : 0.0f;
            if (scale_safe * (d16_safe + lb_t48 * lb_t48) > heap.threshold_sq) continue;

            __m256 qa = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 16), _mm256_load_ps(vp + 16));
            __m256 qb = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 24), _mm256_load_ps(vp + 24));
            float d32 = d16 + hsum256_ps_fast(_mm256_fmadd_ps(qa, qa, _mm256_mul_ps(qb, qb)));
            quarter_evals += 1;
            total_bytes += 64;

            float abs_t32 = std::fabs(q_tail32_p - idx.part_aos_tail32_norm[iv]);
            float lb_t32 = (abs_t32 > fp_slack_p) ? (abs_t32 - fp_slack_p) : 0.0f;
            float d32_safe = (d32 > sq_slack_p) ? (d32 - sq_slack_p) : 0.0f;
            if (scale_safe * (d32_safe + lb_t32 * lb_t32) > heap.threshold_sq) continue;

            quarter_evals += 2;
            total_bytes += 256;
            const float* orig_vp = base_orig + static_cast<size_t>(iv) * 64U;
            float d_orig = l2_hito9_ref(q, orig_vp, 64U);
            if (d_orig <= heap.threshold_sq) {
                heap.push(idx.part_aos_ids[iv], d_orig);
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = (quarter_evals + 2) / 4;
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = parts_visited;
    metrics.partitions_total = n_parts;
    metrics.bytes_read = total_bytes;
}

/*
 * Auditoria Directa de Poda (Hito 2A - Nivel 3):
 * Verifica que para CADA vector elegible descartado en cualquiera de las 4 etapas
 * (Particion, Cascara Radial, PCA 16D+Tail48, PCA 32D+Tail32):
 *   1. Su cota inferior conservadora LB_safe^2(q, x) <= d_orig^2(q, x).
 *   2. Ningun vector descartado pertenece al conjunto exacto Top-K de referencia.
 */
void audit_plan5b_pruning_soundness(const IVFPartitionedIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    const Neighbor* true_topk,
                                    uint32_t true_topk_count,
                                    PruningAuditReport& report) {
    const uint32_t dim = idx.dim;
    const uint32_t n_parts = idx.num_partitions;
    alignas(64) float q_rot[64];
    float q_mean_norm = 0.0f;
    rotate_centered_query_avx2(idx.ortho_basis, q, idx.global_mean, q_rot, q_mean_norm, dim);

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q, idx.centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        float dc_low = dc * (1.0f - PART_FP_REL_EPS);
        float dc_high = dc * (1.0f + PART_FP_REL_EPS);
        const GeometricPartition& gp = idx.partitions[p];
        float lb = 0.0f;
        if (dc_low > gp.max_radius) lb = dc_low - gp.max_radius;
        else if (dc_high < gp.min_radius) lb = gp.min_radius - dc_high;
        p_order[p] = {p, dc, dc_low, dc_high, lb};
    }

    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    TopKHeap heap;
    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_orig = idx.part_orig_vectors;
    const float scale_safe = 1.0f - 2.0f * (idx.ortho_defect + 2.0e-5f);

    auto is_in_true_topk = [&](uint32_t orig_id) -> bool {
        for (uint32_t t = 0; t < true_topk_count; ++t) {
            if (true_topk[t].id == orig_id) return true;
        }
        return false;
    };

    auto record_pruned_check = [&](uint32_t iv, double lb_sq, uint64_t& stage_violations) {
        uint32_t orig_id = idx.part_aos_ids[iv];
        double orig_d2 = static_cast<double>(l2_hito9_ref(q, base_orig + static_cast<size_t>(iv) * dim, dim));
        if (lb_sq > orig_d2 * (1.0 + 1e-6) + 1e-7) {
            stage_violations++;
        }
        if (lb_sq > 1e-12) {
            double ratio = orig_d2 / lb_sq;
            if (ratio < report.min_ratio_orig_over_lb) {
                report.min_ratio_orig_over_lb = ratio;
            }
        }
        if (is_in_true_topk(orig_id)) {
            report.false_prunings_topk++;
        }
    };

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        float dc_low = p_order[pi].dc_low;
        float dc_high = p_order[pi].dc_high;
        const GeometricPartition& gp = idx.partitions[pid];

        if (heap.size == k) {
            float rho_safe = sqrtf(heap.threshold_sq) * (1.0f + PART_FP_REL_EPS) + 1.0e-6f;
            if (p_order[pi].lb > rho_safe) {
                /* Auditar todos los vectores descartados por Poda de Particion */
                double lb_part_safe = static_cast<double>(p_order[pi].lb) / (1.0 + PART_FP_REL_EPS);
                double lb_part_sq = lb_part_safe * lb_part_safe;
                for (uint32_t e = e_start; e < e_end; ++e) {
                    report.total_eligible_checked++;
                    report.pruned_by_partition++;
                    record_pruned_check(el_ivf_ids[e], lb_part_sq, report.bound_violations_partition);
                }
                continue;
            }

            float r_pad = PART_FP_REL_EPS * gp.max_radius + 1.0e-6f;
            float r_low = dc_low - rho_safe - r_pad;
            float r_high = dc_high + rho_safe + r_pad;

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

            /* Auditar vectores descartados en la cascara radial interior [orig_e_start .. e_start) */
            for (uint32_t e = orig_e_start; e < e_start; ++e) {
                uint32_t iv = el_ivf_ids[e];
                report.total_eligible_checked++;
                report.pruned_by_radial_shell++;
                float r_x = idx.part_aos_radii[iv];
                double lb_rad = std::max(0.0, static_cast<double>(dc_low - r_x - r_pad) / (1.0 + PART_FP_REL_EPS));
                record_pruned_check(iv, lb_rad * lb_rad, report.bound_violations_radial);
            }
            /* Auditar vectores descartados en la cascara radial exterior [e_end .. orig_e_end) */
            for (uint32_t e = e_end; e < orig_e_end; ++e) {
                uint32_t iv = el_ivf_ids[e];
                report.total_eligible_checked++;
                report.pruned_by_radial_shell++;
                float r_x = idx.part_aos_radii[iv];
                double lb_rad = std::max(0.0, static_cast<double>(r_x - dc_high - r_pad) / (1.0 + PART_FP_REL_EPS));
                record_pruned_check(iv, lb_rad * lb_rad, report.bound_violations_radial);
            }

            if (e_start >= e_end) continue;
        }

        uint32_t e = e_start;
        while (e < e_end && heap.size < k) {
            uint32_t iv = el_ivf_ids[e];
            report.total_eligible_checked++;
            report.evaluated_full_64d++;
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

        float max_vec_norm = dc_high + gp.max_radius;
        float fp_slack_p = 8.0e-6f * (q_mean_norm + idx.centroid_mean_norms[pid] + gp.max_radius)
                         + idx.ortho_defect * max_vec_norm
                         + 1.0e-6f;
        float sq_slack_p = 2.0f * fp_slack_p * max_vec_norm + 1.0e-6f;

        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            report.total_eligible_checked++;
            const float* vp = idx.part_aos_vectors + static_cast<size_t>(iv) * 64U;

            __m256 q0 = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 0), _mm256_load_ps(vp + 0));
            __m256 q1 = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 8), _mm256_load_ps(vp + 8));
            float d16 = hsum256_ps_fast(_mm256_fmadd_ps(q0, q0, _mm256_mul_ps(q1, q1)));

            float abs_t48 = std::fabs(q_tail48_p - idx.part_aos_tail48_norm[iv]);
            float lb_t48 = (abs_t48 > fp_slack_p) ? (abs_t48 - fp_slack_p) : 0.0f;
            float d16_safe = (d16 > sq_slack_p) ? (d16 - sq_slack_p) : 0.0f;
            float lb64_s1 = scale_safe * (d16_safe + lb_t48 * lb_t48);
            if (lb64_s1 > heap.threshold_sq) {
                report.pruned_by_pca_16d++;
                record_pruned_check(iv, static_cast<double>(lb64_s1), report.bound_violations_pca16);
                continue;
            }

            __m256 qa = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 16), _mm256_load_ps(vp + 16));
            __m256 qb = _mm256_sub_ps(_mm256_load_ps(q_rot_p + 24), _mm256_load_ps(vp + 24));
            float d32 = d16 + hsum256_ps_fast(_mm256_fmadd_ps(qa, qa, _mm256_mul_ps(qb, qb)));

            float abs_t32 = std::fabs(q_tail32_p - idx.part_aos_tail32_norm[iv]);
            float lb_t32 = (abs_t32 > fp_slack_p) ? (abs_t32 - fp_slack_p) : 0.0f;
            float d32_safe = (d32 > sq_slack_p) ? (d32 - sq_slack_p) : 0.0f;
            float lb64_s2 = scale_safe * (d32_safe + lb_t32 * lb_t32);
            if (lb64_s2 > heap.threshold_sq) {
                report.pruned_by_pca_32d++;
                record_pruned_check(iv, static_cast<double>(lb64_s2), report.bound_violations_pca32);
                continue;
            }

            report.evaluated_full_64d++;
            const float* orig_vp = base_orig + static_cast<size_t>(iv) * 64U;
            float d_orig = l2_hito9_ref(q, orig_vp, 64U);
            if (d_orig <= heap.threshold_sq) {
                heap.push(idx.part_aos_ids[iv], d_orig);
            }
        }
    }
}

} // namespace alg_flat
