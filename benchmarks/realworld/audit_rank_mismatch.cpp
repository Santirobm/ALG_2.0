// audit_rank_mismatch.cpp
#include "../faiss_comparison/faiss_c_shim.h"
#include "../../include/dataset_loader.h"
#include "../../include/physical_layouts.h"
#include "../../include/execution_plans.h"
#include "../../include/ieee754_bounds.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

using namespace alg_flat;

struct DoubleNeighbor {
    uint32_t id;
    double dist_sq;
};

struct DoubleTopKHeap {
    DoubleNeighbor data[MAX_TOPK_CAPACITY];
    uint32_t size;
    uint32_t k;
    double threshold_sq;

    inline void init(uint32_t target_k) {
        size = 0;
        k = (target_k <= MAX_TOPK_CAPACITY) ? target_k : MAX_TOPK_CAPACITY;
        threshold_sq = 1.0e300;
    }

    static inline bool less(const DoubleNeighbor& a, const DoubleNeighbor& b) {
        if (a.dist_sq != b.dist_sq) return a.dist_sq < b.dist_sq;
        return a.id < b.id;
    }

    inline void push(uint32_t id, double dist_sq) {
        if (size < k) {
            uint32_t i = size++;
            while (i > 0) {
                uint32_t p = (i - 1) >> 1;
                DoubleNeighbor cand{id, dist_sq};
                if (!less(data[p], cand)) break;
                data[i] = data[p];
                i = p;
            }
            data[i] = {id, dist_sq};
            if (size == k) threshold_sq = data[0].dist_sq;
            return;
        }
        if (dist_sq >= threshold_sq) return;

        uint32_t i = 0;
        DoubleNeighbor cand{id, dist_sq};
        while (true) {
            uint32_t left = (i << 1) + 1;
            uint32_t right = left + 1;
            if (left >= k) break;
            uint32_t best = left;
            if (right < k && less(data[left], data[right])) best = right;
            if (!less(cand, data[best])) break;
            data[i] = data[best];
            i = best;
        }
        data[i] = cand;
        threshold_sq = data[0].dist_sq;
    }

    inline void sort_ascending(DoubleNeighbor* out) {
        uint32_t n = size;
        for (uint32_t i = 0; i < n; ++i) {
            out[n - 1 - i] = data[0];
            DoubleNeighbor last = data[--size];
            if (size == 0) break;
            uint32_t cur = 0;
            while (true) {
                uint32_t left = (cur << 1) + 1;
                uint32_t right = left + 1;
                if (left >= size) break;
                uint32_t best = left;
                if (right < size && less(data[left], data[right])) best = right;
                if (!less(last, data[best])) break;
                data[cur] = data[best];
                cur = best;
            }
            data[cur] = last;
        }
    }
};

int main() {
    const std::string corpus_file = "data/hito3_3/D64/corpus.f32";
    const std::string queries_file = "data/hito3_3/D64/queries.f32";
    const uint32_t N = 90000;
    const uint32_t dim = 64;
    const uint32_t num_queries_eval = 100;
    const uint32_t k = 100;
    const uint32_t target_cat = 1000; // 10% selectivity

    std::ifstream f_corp(corpus_file, std::ios::binary);
    BenchmarkDataset ds;
    ds.header.num_vectors = N;
    ds.header.dimension = dim;
    ds.header.num_queries = num_queries_eval * 2;
    ds.vectors = static_cast<float*>(_aligned_malloc(static_cast<size_t>(N) * dim * sizeof(float), 64));
    ds.categories = static_cast<uint32_t*>(_aligned_malloc(static_cast<size_t>(N) * sizeof(uint32_t), 64));
    ds.queries.resize(num_queries_eval * 2);
    f_corp.read(reinterpret_cast<char*>(ds.vectors), static_cast<size_t>(N) * dim * sizeof(float));
    f_corp.close();

    for (uint32_t i = 0; i < N; ++i) ds.categories[i] = i % 10000;

    std::ifstream f_q(queries_file, std::ios::binary);
    for (uint32_t q = 0; q < num_queries_eval * 2; ++q) {
        f_q.read(reinterpret_cast<char*>(ds.queries[q].values), dim * sizeof(float));
    }
    f_q.close();

    IVFPartitionedIndex idx_alg;
    build_ivf_partitioned_index(ds, 128U, 15U, idx_alg);

    // Slice 0 for 10%
    if (idx_alg.eligible_ivf_ids_per_sel.size() <= 0) {
        idx_alg.eligible_ivf_ids_per_sel.resize(1);
        idx_alg.eligible_part_offsets_per_sel.resize(1);
    }
    auto& el_ids = idx_alg.eligible_ivf_ids_per_sel[0];
    auto& part_offs = idx_alg.eligible_part_offsets_per_sel[0];
    el_ids.clear();
    part_offs.resize(idx_alg.num_partitions + 1);
    for (uint32_t c = 0; c < idx_alg.num_partitions; ++c) {
        part_offs[c] = static_cast<uint32_t>(el_ids.size());
        uint32_t p_start = idx_alg.partitions[c].aos_offset;
        uint32_t p_end = p_start + idx_alg.partitions[c].count;
        for (uint32_t idx_v = p_start; idx_v < p_end; ++idx_v) {
            if (idx_alg.part_aos_categories[idx_v] < target_cat) {
                el_ids.push_back(idx_v);
            }
        }
    }
    part_offs[idx_alg.num_partitions] = static_cast<uint32_t>(el_ids.size());

    std::vector<uint32_t> eligible_ids;
    for (uint32_t id = 0; id < N; ++id) {
        if (ds.categories[id] < target_cat) eligible_ids.push_back(id);
    }

    QueryPredicate pred;
    pred.target_category = target_cat;
    pred.eligible_ids = eligible_ids.data();
    pred.eligible_count = static_cast<uint32_t>(eligible_ids.size());
    pred.sel_index = 0;

    DoubleNeighbor gt_topk[512];
    Neighbor res_alg[512];
    uint32_t cnt_alg = 0;
    QueryExecutionMetrics met_alg;

    std::cout << "Buscando consultas con discrepancia posicional en COLD (Top-100, Sel=10%)..." << std::endl;
    for (uint32_t q = 0; q < num_queries_eval; ++q) {
        uint32_t q_idx = num_queries_eval + q; // COLD queries
        const float* q_vec = ds.queries[q_idx].values;

        DoubleTopKHeap heap;
        heap.init(k);
        for (uint32_t id : eligible_ids) {
            const float* v = ds.vectors + static_cast<size_t>(id) * dim;
            double sum_sq = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                double diff = static_cast<double>(q_vec[d]) - static_cast<double>(v[d]);
                sum_sq += diff * diff;
            }
            heap.push(id, sum_sq);
        }
        heap.sort_ascending(gt_topk);

        execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);

        // Comparar
        bool match = true;
        for (uint32_t i = 0; i < k; ++i) {
            if (res_alg[i].id != gt_topk[i].id) {
                match = false;
                break;
            }
        }

        if (!match) {
            std::cout << ">>> Discrepancia encontrada en Consulta #" << q << " (q_idx=" << q_idx << "):" << std::endl;
            for (uint32_t i = 0; i < k; ++i) {
                if (res_alg[i].id != gt_topk[i].id) {
                    uint32_t id_alg = res_alg[i].id;
                    uint32_t id_gt = gt_topk[i].id;

                    // Calcular distancias en double exacto
                    double d_alg_double = 0.0, d_gt_double = 0.0;
                    float d_alg_float = 0.0f, d_gt_float = 0.0f;
                    for (uint32_t d = 0; d < dim; ++d) {
                        double diff_a = static_cast<double>(q_vec[d]) - static_cast<double>(ds.vectors[id_alg * dim + d]);
                        double diff_g = static_cast<double>(q_vec[d]) - static_cast<double>(ds.vectors[id_gt * dim + d]);
                        d_alg_double += diff_a * diff_a;
                        d_gt_double += diff_g * diff_g;

                        float diff_af = q_vec[d] - ds.vectors[id_alg * dim + d];
                        float diff_gf = q_vec[d] - ds.vectors[id_gt * dim + d];
                        d_alg_float += diff_af * diff_af;
                        d_gt_float += diff_gf * diff_gf;
                    }

                    std::cout << "   Posicion " << i << ":" << std::endl;
                    std::cout << "     ALG id=" << id_alg << " | dist_float=" << std::setprecision(9) << res_alg[i].dist_sq
                              << " | dist_recalc_float=" << d_alg_float
                              << " | dist_double=" << std::setprecision(17) << d_alg_double << std::endl;
                    std::cout << "     GT  id=" << id_gt << " | dist_double=" << std::setprecision(17) << gt_topk[i].dist_sq
                              << " | dist_recalc_float=" << d_gt_float << std::endl;
                    std::cout << "     Delta float: " << std::abs(d_alg_float - d_gt_float)
                              << " | Delta double: " << std::abs(d_alg_double - d_gt_double) << std::endl;
                }
            }
        }
    }
    return 0;
}
