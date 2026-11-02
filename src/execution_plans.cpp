#include "../include/execution_plans.h"
#include "../include/simd_kernels.h"
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

/* Kernel original de Hito 8/9 con 1 acumulador y _mm_hadd_ps */
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
        float d = l2_sq_scalar_strict(q, idx.vectors + static_cast<size_t>(id) * dim, dim);
        if (d <= heap.threshold_sq) {
            heap.push(id, d);
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
        float d = l2_sq_avx2_1x_aos(q, idx.vectors + static_cast<size_t>(id) * dim, dim);
        if (d <= heap.threshold_sq) {
            heap.push(id, d);
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

        __m128 d4 = l2_sq_avx2_4x_aos(q,
                                      base_v + static_cast<size_t>(id0) * dim,
                                      base_v + static_cast<size_t>(id1) * dim,
                                      base_v + static_cast<size_t>(id2) * dim,
                                      base_v + static_cast<size_t>(id3) * dim,
                                      dim);

        __m128 v_thresh = _mm_set1_ps(heap.threshold_sq);
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, v_thresh));
        if (mask != 0) {
            alignas(16) float dists[4];
            _mm_store_ps(dists, d4);
            if (mask & 1) heap.push(id0, dists[0]);
            if (mask & 2) heap.push(id1, dists[1]);
            if (mask & 4) heap.push(id2, dists[2]);
            if (mask & 8) heap.push(id3, dists[3]);
        }
    }

    for (; i < n_el; ++i) {
        uint32_t id = el_ids[i];
        float d = l2_sq_avx2_1x_aos(q, base_v + static_cast<size_t>(id) * dim, dim);
        if (d <= heap.threshold_sq) {
            heap.push(id, d);
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
        __m256 v_thresh = _mm256_set1_ps(heap.threshold_sq);
        int hit_mask = _mm256_movemask_ps(_mm256_cmp_ps(d8, v_thresh, _CMP_LE_OS)) & elig_mask;

        if (hit_mask != 0) {
            alignas(32) float dists[8];
            _mm256_store_ps(dists, d8);
            for (uint32_t lane = 0; lane < 8U; ++lane) {
                if (hit_mask & (1 << lane)) {
                    heap.push(idx.ids[vec_idx + lane], dists[lane]);
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
        __m256 v_thresh = _mm256_set1_ps(heap.threshold_sq);
        int hit_mask = _mm256_movemask_ps(_mm256_cmp_ps(d8, v_thresh, _CMP_LE_OS)) & elig_mask;

        if (hit_mask != 0) {
            alignas(32) float dists[8];
            _mm256_store_ps(dists, d8);
            for (uint32_t lane = 0; lane < 8U; ++lane) {
                if (hit_mask & (1 << lane)) {
                    heap.push(idx.ids[vec_idx + lane], dists[lane]);
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

static inline void rotate_query_avx2(const float* qt, const float* q, float* q_rot, uint32_t dim) {
    for (uint32_t r = 0; r < dim; ++r) {
        const float* row = qt + static_cast<size_t>(r) * dim;
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        for (uint32_t d = 0; d < dim; d += 16) {
            acc0 = _mm256_fmadd_ps(_mm256_load_ps(row + d), _mm256_loadu_ps(q + d), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_load_ps(row + d + 8), _mm256_loadu_ps(q + d + 8), acc1);
        }
        q_rot[r] = hsum256_ps_fast(_mm256_add_ps(acc0, acc1));
    }
}

struct PartitionOrder {
    uint32_t part_id;
    float dc;       /* ||q - c_p||_2 */
    float lb;       /* max(0, dc - r_max, r_min - dc) */
};

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
    alignas(64) float q_rot[64];
    rotate_query_avx2(idx.ortho_basis, q, q_rot, dim);

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q_rot, idx.rotated_centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        p_order[p] = {p, dc, dc};
    }
    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_v = idx.part_aos_vectors;

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

            dists_eval += 4;
            __m128 d4 = l2_sq_avx2_4x_aos(q_rot,
                                          base_v + static_cast<size_t>(iv0) * dim,
                                          base_v + static_cast<size_t>(iv1) * dim,
                                          base_v + static_cast<size_t>(iv2) * dim,
                                          base_v + static_cast<size_t>(iv3) * dim,
                                          dim);
            int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(heap.threshold_sq)));
            if (mask != 0) {
                alignas(16) float dists[4];
                _mm_store_ps(dists, d4);
                if (mask & 1) heap.push(idx.part_aos_ids[iv0], dists[0]);
                if (mask & 2) heap.push(idx.part_aos_ids[iv1], dists[1]);
                if (mask & 4) heap.push(idx.part_aos_ids[iv2], dists[2]);
                if (mask & 8) heap.push(idx.part_aos_ids[iv3], dists[3]);
            }
        }
        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            dists_eval++;
            float d = l2_sq_avx2_1x_aos(q_rot, base_v + static_cast<size_t>(iv) * dim, dim);
            if (d <= heap.threshold_sq) {
                heap.push(idx.part_aos_ids[iv], d);
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
    alignas(64) float q_rot[64];
    rotate_query_avx2(idx.ortho_basis, q, q_rot, dim);

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q_rot, idx.rotated_centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        const GeometricPartition& gp = idx.partitions[p];
        float lb = 0.0f;
        if (dc > gp.max_radius) lb = dc - gp.max_radius;
        else if (dc < gp.min_radius) lb = gp.min_radius - dc;
        p_order[p] = {p, dc, lb};
    }
    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_v = idx.part_aos_vectors;

    uint32_t dists_eval = 0;
    uint32_t parts_visited = 0;

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        if (heap.size == k) {
            float safety_rho = sqrtf(heap.threshold_sq) * 1.0002f;
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

            dists_eval += 4;
            __m128 d4 = l2_sq_avx2_4x_aos(q_rot,
                                          base_v + static_cast<size_t>(iv0) * dim,
                                          base_v + static_cast<size_t>(iv1) * dim,
                                          base_v + static_cast<size_t>(iv2) * dim,
                                          base_v + static_cast<size_t>(iv3) * dim,
                                          dim);
            int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(heap.threshold_sq)));
            if (mask != 0) {
                alignas(16) float dists[4];
                _mm_store_ps(dists, d4);
                if (mask & 1) heap.push(idx.part_aos_ids[iv0], dists[0]);
                if (mask & 2) heap.push(idx.part_aos_ids[iv1], dists[1]);
                if (mask & 4) heap.push(idx.part_aos_ids[iv2], dists[2]);
                if (mask & 8) heap.push(idx.part_aos_ids[iv3], dists[3]);
            }
        }
        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            dists_eval++;
            float d = l2_sq_avx2_1x_aos(q_rot, base_v + static_cast<size_t>(iv) * dim, dim);
            if (d <= heap.threshold_sq) {
                heap.push(idx.part_aos_ids[iv], d);
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
 * Kernel 4x1 Progresivo en Base Rotada PCA (16D + Cota Exacta Tail-48D, luego 48D restantes solo para sobrevivientes)
 */
static inline void eval_4x_pca_progressive(const float* q_rot,
                                           float q_tail48,
                                           float q_tail32,
                                           const IVFPartitionedIndex& idx,
                                           uint32_t iv0, uint32_t iv1, uint32_t iv2, uint32_t iv3,
                                           TopKHeap& heap,
                                           uint32_t& quarter_evals,
                                           uint64_t& bytes_read) {
    const float* v0 = idx.part_aos_vectors + static_cast<size_t>(iv0) * 64U;
    const float* v1 = idx.part_aos_vectors + static_cast<size_t>(iv1) * 64U;
    const float* v2 = idx.part_aos_vectors + static_cast<size_t>(iv2) * 64U;
    const float* v3 = idx.part_aos_vectors + static_cast<size_t>(iv3) * 64U;

    /* Etapa 1: Primeras 16 dimensiones principales (1 cache line = 64B por vector) */
    __m256 q0 = _mm256_load_ps(q_rot + 0);
    __m256 q1 = _mm256_load_ps(q_rot + 8);

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

    quarter_evals += 4; /* 4 x (1/4 de vector) = 1 vector equivalente */
    bytes_read += 4 * 64 + 4 * sizeof(float);

    /* Sumar cota inferior exacta de las 48 dimensiones restantes: (||q_48|| - ||x_48||)^2 */
    __m128 b4 = _mm_set_ps(idx.part_aos_tail48_norm[iv3],
                           idx.part_aos_tail48_norm[iv2],
                           idx.part_aos_tail48_norm[iv1],
                           idx.part_aos_tail48_norm[iv0]);
    __m128 diff_b = _mm_sub_ps(_mm_set1_ps(q_tail48), b4);
    __m128 lb64_4 = _mm_fmadd_ps(diff_b, diff_b, d16_4);

    float safe_thresh = heap.threshold_sq * 1.0002f;
    int mask16 = _mm_movemask_ps(_mm_cmple_ps(lb64_4, _mm_set1_ps(safe_thresh)));
    if (mask16 == 0) {
        return; /* Los 4 vectores quedan podados exactamente tras leer solo 1 cache line cada uno */
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
        __m256 qa = _mm256_sub_ps(_mm256_load_ps(q_rot + 16), _mm256_load_ps(vp + 16));
        __m256 qb = _mm256_sub_ps(_mm256_load_ps(q_rot + 24), _mm256_load_ps(vp + 24));
        float d32 = p16[lane] + hsum256_ps_fast(_mm256_fmadd_ps(qa, qa, _mm256_mul_ps(qb, qb)));
        quarter_evals += 1;
        bytes_read += 64;

        float diff_t32 = q_tail32 - idx.part_aos_tail32_norm[iv];
        if (d32 + diff_t32 * diff_t32 > safe_thresh) {
            continue; /* Podado exactamente en 32D (ahorra las ultimas 2 cache lines) */
        }

        /* Etapa 3: Completar dimensiones 32..63 (ultimas 2 cache lines) */
        __m256 qc = _mm256_sub_ps(_mm256_load_ps(q_rot + 32), _mm256_load_ps(vp + 32));
        __m256 qd = _mm256_sub_ps(_mm256_load_ps(q_rot + 40), _mm256_load_ps(vp + 40));
        __m256 acc_tail0 = _mm256_fmadd_ps(qc, qc, _mm256_mul_ps(qd, qd));

        __m256 qe = _mm256_sub_ps(_mm256_load_ps(q_rot + 48), _mm256_load_ps(vp + 48));
        __m256 qf = _mm256_sub_ps(_mm256_load_ps(q_rot + 56), _mm256_load_ps(vp + 56));
        __m256 acc_tail1 = _mm256_fmadd_ps(qe, qe, _mm256_mul_ps(qf, qf));

        float d64 = d32 + hsum256_ps_fast(_mm256_add_ps(acc_tail0, acc_tail1));
        quarter_evals += 2;
        bytes_read += 128;

        if (d64 <= heap.threshold_sq) {
            heap.push(idx.part_aos_ids[iv], d64);
            safe_thresh = heap.threshold_sq * 1.0002f;
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
    rotate_query_avx2(idx.ortho_basis, q, q_rot, dim);

    double qt32_sq = 0.0, qt16_31_sq = 0.0;
    for (uint32_t d = 32; d < dim; ++d) qt32_sq += static_cast<double>(q_rot[d]) * q_rot[d];
    for (uint32_t d = 16; d < 32; ++d) qt16_31_sq += static_cast<double>(q_rot[d]) * q_rot[d];
    float q_tail32 = static_cast<float>(std::sqrt(qt32_sq));
    float q_tail48 = static_cast<float>(std::sqrt(qt32_sq + qt16_31_sq));

    PartitionOrder p_order[512];
    for (uint32_t p = 0; p < n_parts; ++p) {
        float d2 = l2_sq_avx2_1x_aos(q_rot, idx.rotated_centroids + static_cast<size_t>(p) * dim, dim);
        float dc = sqrtf(d2);
        const GeometricPartition& gp = idx.partitions[p];
        float lb = 0.0f;
        if (dc > gp.max_radius) lb = dc - gp.max_radius;
        else if (dc < gp.min_radius) lb = gp.min_radius - dc;
        p_order[p] = {p, dc, lb};
    }

    std::sort(p_order, p_order + n_parts, [](const PartitionOrder& a, const PartitionOrder& b) {
        return a.dc < b.dc;
    });

    ts.mark_prep_done();

    TopKHeap heap;
    heap.init(k);

    const uint32_t* el_ivf_ids = idx.eligible_ivf_ids_per_sel[pred.sel_index].data();
    const uint32_t* part_offs = idx.eligible_part_offsets_per_sel[pred.sel_index].data();
    const float* base_v = idx.part_aos_vectors;

    uint32_t quarter_evals = 0;
    uint32_t parts_visited = 0;
    uint64_t total_bytes = 0;

    for (uint32_t pi = 0; pi < n_parts; ++pi) {
        uint32_t pid = p_order[pi].part_id;
        uint32_t e_start = part_offs[pid];
        uint32_t e_end = part_offs[pid + 1];
        if (e_start >= e_end) continue;

        float dc = p_order[pi].dc;
        float rho = FLT_MAX;

        if (heap.size == k) {
            rho = sqrtf(heap.threshold_sq) * 1.0002f;
            if (p_order[pi].lb > rho) {
                continue; /* Nivel 1: Particion entera podada */
            }

            /* Nivel 2: Poda Exacta de Cascara Radial [dc - rho, dc + rho] sobre los elegibles de la particion */
            const GeometricPartition& gp = idx.partitions[pid];
            float r_low = dc - rho;
            float r_high = dc + rho;

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

        /* En la primera particion mientas se llena el heap, evaluar 4x1 directo */
        while (e + 4 <= e_end && heap.size < k) {
            uint32_t iv0 = el_ivf_ids[e];
            uint32_t iv1 = el_ivf_ids[e + 1];
            uint32_t iv2 = el_ivf_ids[e + 2];
            uint32_t iv3 = el_ivf_ids[e + 3];
            quarter_evals += 16;
            total_bytes += 4 * 256;
            __m128 d4 = l2_sq_avx2_4x_aos(q_rot,
                                          base_v + static_cast<size_t>(iv0) * dim,
                                          base_v + static_cast<size_t>(iv1) * dim,
                                          base_v + static_cast<size_t>(iv2) * dim,
                                          base_v + static_cast<size_t>(iv3) * dim,
                                          dim);
            alignas(16) float dists[4];
            _mm_store_ps(dists, d4);
            heap.push(idx.part_aos_ids[iv0], dists[0]);
            heap.push(idx.part_aos_ids[iv1], dists[1]);
            heap.push(idx.part_aos_ids[iv2], dists[2]);
            heap.push(idx.part_aos_ids[iv3], dists[3]);
            e += 4;
        }

        /* Con el umbral rho_K activo, usar el kernel 4x1 Progresivo PCA + Tail-48D/32D */
        for (; e + 4 <= e_end; e += 4) {
            eval_4x_pca_progressive(q_rot, q_tail48, q_tail32, idx,
                                    el_ivf_ids[e], el_ivf_ids[e + 1], el_ivf_ids[e + 2], el_ivf_ids[e + 3],
                                    heap, quarter_evals, total_bytes);
        }

        for (; e < e_end; ++e) {
            uint32_t iv = el_ivf_ids[e];
            quarter_evals += 4;
            total_bytes += 256;
            float d = l2_sq_avx2_1x_aos(q_rot, base_v + static_cast<size_t>(iv) * dim, dim);
            if (d <= heap.threshold_sq) {
                heap.push(idx.part_aos_ids[iv], d);
            }
        }
    }

    heap.sort_ascending();
    *out_count = heap.size;
    memcpy(out_results, heap.data, heap.size * sizeof(Neighbor));

    ts.stop(metrics);
    metrics.dists_evaluated = (quarter_evals + 2) / 4; /* Equivalente en evaluaciones completas de 64D */
    metrics.eligible_total = pred.eligible_count;
    metrics.partitions_visited = parts_visited;
    metrics.partitions_total = n_parts;
    metrics.bytes_read = total_bytes;
}

} // namespace alg_flat
