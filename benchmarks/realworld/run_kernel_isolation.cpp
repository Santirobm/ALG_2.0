// ==============================================================================
// run_kernel_isolation.cpp
// Hito 3.3.2: Experimento de Decisión de Aislamiento de Kernel y Escalabilidad Dimensional
//
// Comparación Simétrica Pura:
//   - A1: ExactScan ALG AVX2 sobre corpus original vía lista de IDs (gather/strided)
//   - A2: ExactScan ALG AVX2 sobre buffer compacto (localidad física idéntica a C1)
//   - C1: FAISS Flat L2 sobre exactamente el mismo buffer compacto
//
// Verificaciones y Controles:
//   - Pre-check de consistencia binaria byte a byte entre A2 y C1
//   - Idéntico orden y mapeo de identificadores locales a globales
//   - Oráculo binary64 con comparador estricto determinista ante empates
//   - Medición end-to-end primaria sin instrumentación interna
//   - Micro-benchmark separado para evaluación pura de distancia SIMD
//   - Métricas completas: Media, DesvStd, p50, p95, p99, QPS, IDSetRecall, StrictRankMatch
// ==============================================================================

#include "../faiss_comparison/faiss_c_shim.h"
#include "../../include/alg_flat_types.h"
#include "../../include/dataset_loader.h"
#include "../../include/physical_layouts.h"
#include "../../include/simd_kernels.h"
#include "../../include/ieee754_bounds.h"

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <map>
#include <windows.h>

using namespace alg_flat;

static inline uint64_t get_highres_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

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

        // Condición estricta de desempate determinista por ID
        DoubleNeighbor cand{id, dist_sq};
        if (!less(cand, data[0])) return;

        uint32_t i = 0;
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

static void compute_ground_truth_double(
    const float* query_values,
    uint32_t dim,
    const float* corpus_vectors,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    uint32_t k,
    DoubleNeighbor* out_topk,
    uint32_t* out_count) {

    DoubleTopKHeap heap;
    heap.init(k);

    for (uint32_t i = 0; i < eligible_count; ++i) {
        uint32_t id = eligible_ids[i];
        const float* v = corpus_vectors + static_cast<size_t>(id) * dim;
        double sum_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            double diff = static_cast<double>(query_values[d]) - static_cast<double>(v[d]);
            sum_sq += diff * diff;
        }
        heap.push(id, sum_sq);
    }

    *out_count = heap.size;
    heap.sort_ascending(out_topk);
}

// Variante A1: ExactScan AVX2 sobre corpus original vía array de IDs (acceso no contiguo/gather)
static inline void run_variant_a1_exact_scan(
    const float* corpus_vectors,
    uint32_t dim,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    const float* query_values,
    uint32_t k,
    Neighbor* out_topk,
    uint32_t* out_count) {

    TopKHeap heap;
    heap.init(k);

    uint32_t i = 0;
    for (; i + 4 <= eligible_count; i += 4) {
        uint32_t id0 = eligible_ids[i];
        uint32_t id1 = eligible_ids[i + 1];
        uint32_t id2 = eligible_ids[i + 2];
        uint32_t id3 = eligible_ids[i + 3];

        const float* v0 = corpus_vectors + static_cast<size_t>(id0) * dim;
        const float* v1 = corpus_vectors + static_cast<size_t>(id1) * dim;
        const float* v2 = corpus_vectors + static_cast<size_t>(id2) * dim;
        const float* v3 = corpus_vectors + static_cast<size_t>(id3) * dim;

        __m128 d4 = l2_sq_avx2_4x_aos(query_values, v0, v1, v2, v3, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
        if (mask != 0) {
            alignas(16) float d[4];
            _mm_store_ps(d, d4);
            if (mask & 1) heap.push(id0, d[0]);
            if (mask & 2) heap.push(id1, d[1]);
            if (mask & 4) heap.push(id2, d[2]);
            if (mask & 8) heap.push(id3, d[3]);
        }
    }
    for (; i < eligible_count; ++i) {
        uint32_t id = eligible_ids[i];
        const float* v = corpus_vectors + static_cast<size_t>(id) * dim;
        float d_sq = l2_sq_avx2_1x_aos(query_values, v, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d_sq <= gate) {
            heap.push(id, d_sq);
        }
    }

    *out_count = heap.size;
    heap.sort_ascending();
    for (uint32_t j = 0; j < heap.size; ++j) out_topk[j] = heap.data[j];
}

// Variante A2: ExactScan ALG AVX2 sobre buffer compacto 100% contiguo (idéntico layout a C1)
static inline void run_variant_a2_exact_scan(
    const float* compact_vectors,
    uint32_t dim,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    const float* query_values,
    uint32_t k,
    Neighbor* out_topk,
    uint32_t* out_count) {

    TopKHeap heap;
    heap.init(k);

    uint32_t i = 0;
    for (; i + 4 <= eligible_count; i += 4) {
        const float* v0 = compact_vectors + static_cast<size_t>(i + 0) * dim;
        const float* v1 = compact_vectors + static_cast<size_t>(i + 1) * dim;
        const float* v2 = compact_vectors + static_cast<size_t>(i + 2) * dim;
        const float* v3 = compact_vectors + static_cast<size_t>(i + 3) * dim;

        __m128 d4 = l2_sq_avx2_4x_aos(query_values, v0, v1, v2, v3, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        int mask = _mm_movemask_ps(_mm_cmple_ps(d4, _mm_set1_ps(gate)));
        if (mask != 0) {
            alignas(16) float d[4];
            _mm_store_ps(d, d4);
            if (mask & 1) heap.push(eligible_ids[i + 0], d[0]);
            if (mask & 2) heap.push(eligible_ids[i + 1], d[1]);
            if (mask & 4) heap.push(eligible_ids[i + 2], d[2]);
            if (mask & 8) heap.push(eligible_ids[i + 3], d[3]);
        }
    }
    for (; i < eligible_count; ++i) {
        const float* v = compact_vectors + static_cast<size_t>(i) * dim;
        float d_sq = l2_sq_avx2_1x_aos(query_values, v, dim);
        float gate = (heap.size < k) ? FLT_MAX : (heap.threshold_sq * (1.0f + SIMD_TREE_GATE_REL_EPS));
        if (d_sq <= gate) {
            heap.push(eligible_ids[i], d_sq);
        }
    }

    *out_count = heap.size;
    heap.sort_ascending();
    for (uint32_t j = 0; j < heap.size; ++j) out_topk[j] = heap.data[j];
}

// Micro-benchmark de Distancia Pura (sin lógica de Top-K heap)
static inline void run_pure_distance_avx2(
    const float* compact_vectors,
    uint32_t dim,
    uint32_t count,
    const float* query_values,
    float* out_dists) {

    uint32_t i = 0;
    for (; i + 4 <= count; i += 4) {
        const float* v0 = compact_vectors + static_cast<size_t>(i + 0) * dim;
        const float* v1 = compact_vectors + static_cast<size_t>(i + 1) * dim;
        const float* v2 = compact_vectors + static_cast<size_t>(i + 2) * dim;
        const float* v3 = compact_vectors + static_cast<size_t>(i + 3) * dim;
        __m128 d4 = l2_sq_avx2_4x_aos(query_values, v0, v1, v2, v3, dim);
        _mm_storeu_ps(out_dists + i, d4);
    }
    for (; i < count; ++i) {
        const float* v = compact_vectors + static_cast<size_t>(i) * dim;
        out_dists[i] = l2_sq_avx2_1x_aos(query_values, v, dim);
    }
}

static double compute_type7_quantile(const std::vector<double>& sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    if (sorted_v.size() == 1) return sorted_v[0];
    if (p <= 0.0) return sorted_v.front();
    if (p >= 1.0) return sorted_v.back();
    double h = static_cast<double>(sorted_v.size() - 1) * p;
    size_t i = static_cast<size_t>(h);
    double f = h - static_cast<double>(i);
    return (1.0 - f) * sorted_v[i] + f * sorted_v[i + 1];
}

// Estructura de métricas estadísticas
struct LatencyStats {
    double mean;
    double stddev;
    double p50;
    double p95;
    double p99;
    double qps;
};

static LatencyStats calc_stats(std::vector<double>& latencies) {
    LatencyStats st{0, 0, 0, 0, 0, 0};
    if (latencies.empty()) return st;
    size_t n = latencies.size();
    st.mean = std::accumulate(latencies.begin(), latencies.end(), 0.0) / n;
    double var = 0.0;
    for (double x : latencies) var += (x - st.mean) * (x - st.mean);
    st.stddev = std::sqrt(var / n);
    std::sort(latencies.begin(), latencies.end());
    st.p50 = compute_type7_quantile(latencies, 0.50);
    st.p95 = compute_type7_quantile(latencies, 0.95);
    st.p99 = compute_type7_quantile(latencies, 0.99);
    st.qps = (st.mean > 0.0) ? (1.0e6 / st.mean) : 0.0;
    return st;
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "HITO 3.3.2: EXPERIMENTO DE DECISION - AISLAMIENTO DE KERNEL (A1 vs A2 vs C1)" << std::endl;
    std::cout << "  - A1: ExactScan ALG AVX2 sobre corpus original vía array de IDs (gather)" << std::endl;
    std::cout << "  - A2: ExactScan ALG AVX2 sobre buffer compacto contiguo (localidad pura)" << std::endl;
    std::cout << "  - C1: FAISS Flat L2 sobre exactamente el mismo buffer compacto" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    std::vector<uint32_t> test_dims = {64, 128, 256};
    if (argc > 1) {
        test_dims.clear();
        for (int i = 1; i < argc; ++i) {
            test_dims.push_back(std::stoul(argv[i]));
        }
    }

    CreateDirectoryA("results/hito3_3_2", NULL);
    std::ofstream out_summary("results/hito3_3_2/kernel_isolation_summary.csv");
    out_summary << "Dimension,FilterFamily,Selectivity,EligibleCount,TopK,CacheMode,"
                << "Mean_A1_us,StdDev_A1,P50_A1_us,P95_A1_us,P99_A1_us,QPS_A1,"
                << "Mean_A2_us,StdDev_A2,P50_A2_us,P95_A2_us,P99_A2_us,QPS_A2,"
                << "Mean_C1_us,StdDev_C1,P50_C1_us,P95_C1_us,P99_C1_us,QPS_C1,"
                << "Speedup_A2_vs_C1,Speedup_A2_vs_A1,Speedup_A1_vs_C1,"
                << "PureDist_A2_us,PureDist_FAISS_us,PureDist_Speedup,"
                << "Recall_A1,RankMatch_A1,Recall_A2,RankMatch_A2,Recall_C1,RankMatch_C1\n";

    ColdCacheEvictor evictor(36);
    const uint32_t N = 90000;
    const uint32_t num_queries_eval = 1000; // 1.000 consultas por celda para análisis completo

    std::vector<float> selectivities = {1.0f, 10.0f, 100.0f};
    std::vector<uint32_t> top_ks = {10U, 100U};
    std::vector<std::string> cache_modes = {"WARM_CACHE", "COLD_LLC_FLUSH"};

    DoubleNeighbor gt[512];
    Neighbor res_a1[512];
    Neighbor res_a2[512];
    float dists_c1[512];
    int64_t labels_c1[512];
    uint32_t cnt_a1 = 0, cnt_a2 = 0;

    for (uint32_t dim : test_dims) {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << ">>> EVALUANDO DIMENSION D = " << dim << std::endl;
        std::cout << "================================================================================" << std::endl;

        std::string dim_str = "D" + std::to_string(dim);
        std::string corpus_file = "data/hito3_3/" + dim_str + "/corpus.f32";
        std::string queries_file = "data/hito3_3/" + dim_str + "/queries.f32";

        std::vector<float> corpus(static_cast<size_t>(N) * dim);
        std::ifstream f_corp(corpus_file, std::ios::binary);
        if (!f_corp) {
            std::cerr << "Error cargando " << corpus_file << std::endl;
            continue;
        }
        f_corp.read(reinterpret_cast<char*>(corpus.data()), corpus.size() * sizeof(float));
        f_corp.close();

        std::vector<float> queries(10000 * dim);
        std::ifstream f_q(queries_file, std::ios::binary);
        if (!f_q) {
            std::cerr << "Error cargando " << queries_file << std::endl;
            continue;
        }
        f_q.read(reinterpret_cast<char*>(queries.data()), queries.size() * sizeof(float));
        f_q.close();

        // Categorías sintéticas deterministas para selectividad
        std::vector<uint32_t> categories(N);
        for (uint32_t i = 0; i < N; ++i) categories[i] = i % 10000;

        for (float sel : selectivities) {
            std::vector<uint32_t> eligible_ids;
            eligible_ids.reserve(N);
            uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);
            if (sel >= 100.0f) target_cat = 10001U;
            for (uint32_t id = 0; id < N; ++id) {
                if (categories[id] < target_cat) {
                    eligible_ids.push_back(id);
                }
            }
            uint32_t eligible_count = static_cast<uint32_t>(eligible_ids.size());

            // Construir buffer físicamente contiguo compact_vectors
            std::vector<float> compact_vectors(static_cast<size_t>(eligible_count) * dim);
            for (uint32_t i = 0; i < eligible_count; ++i) {
                uint32_t orig_id = eligible_ids[i];
                memcpy(&compact_vectors[static_cast<size_t>(i) * dim],
                       &corpus[static_cast<size_t>(orig_id) * dim],
                       dim * sizeof(float));
            }

            // Construir FAISS Flat compacto sobre el MISMO buffer fuera del cronómetro
            FaissFlatHandle faiss_compact = faiss_shim_create_flat_l2(dim);
            faiss_shim_flat_add(faiss_compact, eligible_count, compact_vectors.data());

            // -------------------------------------------------------------------------
            // PRUEBA DE CONSISTENCIA NO NEGOCIABLE:
            // Validar identidad binaria byte a byte y exactitud de mapeo antes de medir
            // -------------------------------------------------------------------------
            bool consistency_passed = true;
            for (uint32_t i = 0; i < eligible_count; ++i) {
                uint32_t orig_id = eligible_ids[i];
                const float* p_orig = &corpus[static_cast<size_t>(orig_id) * dim];
                const float* p_comp = &compact_vectors[static_cast<size_t>(i) * dim];
                if (memcmp(p_orig, p_comp, dim * sizeof(float)) != 0) {
                    consistency_passed = false;
                    break;
                }
            }
            if (!consistency_passed) {
                std::cerr << " [ERROR CRITICO] Fallo de consistencia binaria en D=" << dim << ", Sel=" << sel << std::endl;
                faiss_shim_free_flat(faiss_compact);
                return 1;
            }

            // -------------------------------------------------------------------------
            // MICRO-BENCHMARK: Pasada de Distancia Pura (sin lógica de Top-K heap)
            // -------------------------------------------------------------------------
            std::vector<float> pure_dists_alg(eligible_count);
            std::vector<float> pure_dists_faiss(eligible_count);
            std::vector<int64_t> dummy_labels(eligible_count);
            uint32_t pure_test_queries = 100;
            double pure_time_alg_total = 0.0;
            double pure_time_faiss_total = 0.0;

            for (uint32_t q = 0; q < pure_test_queries; ++q) {
                const float* q_vec = &queries[q * dim];
                uint64_t t0 = get_highres_time_ns();
                run_pure_distance_avx2(compact_vectors.data(), dim, eligible_count, q_vec, pure_dists_alg.data());
                uint64_t t1 = get_highres_time_ns();
                pure_time_alg_total += (t1 - t0) / 1000.0;

                uint64_t tf0 = get_highres_time_ns();
                faiss_shim_search_flat(faiss_compact, 1, q_vec, eligible_count, pure_dists_faiss.data(), dummy_labels.data(), nullptr);
                uint64_t tf1 = get_highres_time_ns();
                pure_time_faiss_total += (tf1 - tf0) / 1000.0;
            }
            double mean_pure_dist_a2 = pure_time_alg_total / pure_test_queries;
            double mean_pure_dist_faiss = pure_time_faiss_total / pure_test_queries;
            double pure_dist_speedup = mean_pure_dist_faiss / mean_pure_dist_a2;

            // -------------------------------------------------------------------------
            // MEDICION PRIMARIA END-TO-END (Sin instrumentación interna contaminante)
            // -------------------------------------------------------------------------
            for (uint32_t k : top_ks) {
                for (const auto& cache_mode : cache_modes) {
                    bool is_cold = (cache_mode == "COLD_LLC_FLUSH");
                    uint32_t q_offset = is_cold ? num_queries_eval : 0;

                    std::vector<double> lat_a1, lat_a2, lat_c1;
                    lat_a1.reserve(num_queries_eval);
                    lat_a2.reserve(num_queries_eval);
                    lat_c1.reserve(num_queries_eval);

                    uint32_t match_rec_a1 = 0, match_rec_a2 = 0, match_rec_c1 = 0;
                    uint32_t match_rnk_a1 = 0, match_rnk_a2 = 0, match_rnk_c1 = 0;

                    for (uint32_t q = 0; q < num_queries_eval; ++q) {
                        uint32_t q_idx = q_offset + q;
                        const float* q_vec = &queries[q_idx * dim];

                        // Oráculo Ground Truth binary64 corregido (con comparador estricto)
                        uint32_t gt_cnt = 0;
                        compute_ground_truth_double(q_vec, dim, corpus.data(),
                                                    eligible_ids.data(), eligible_count,
                                                    k, gt, &gt_cnt);

                        // Esquema de rotación balanceada para eliminar sesgo de orden
                        auto run_A1 = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            run_variant_a1_exact_scan(corpus.data(), dim, eligible_ids.data(), eligible_count, q_vec, k, res_a1, &cnt_a1);
                            uint64_t t1 = get_highres_time_ns();
                            lat_a1.push_back((t1 - t0) / 1000.0);
                        };

                        auto run_A2 = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            run_variant_a2_exact_scan(compact_vectors.data(), dim, eligible_ids.data(), eligible_count, q_vec, k, res_a2, &cnt_a2);
                            uint64_t t1 = get_highres_time_ns();
                            lat_a2.push_back((t1 - t0) / 1000.0);
                        };

                        auto run_C1 = [&]() {
                            if (is_cold) evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(faiss_compact, 1, q_vec, k, dists_c1, labels_c1, nullptr);
                            uint64_t t1 = get_highres_time_ns();
                            lat_c1.push_back((t1 - t0) / 1000.0);
                        };

                        uint32_t rot = q % 3;
                        if (rot == 0) { run_A1(); run_A2(); run_C1(); }
                        else if (rot == 1) { run_A2(); run_C1(); run_A1(); }
                        else { run_C1(); run_A1(); run_A2(); }

                        // Validar exactitud
                        std::vector<uint32_t> ids_a1(cnt_a1), ids_a2(cnt_a2), ids_c1(k);
                        for (uint32_t i = 0; i < cnt_a1; ++i) ids_a1[i] = res_a1[i].id;
                        for (uint32_t i = 0; i < cnt_a2; ++i) ids_a2[i] = res_a2[i].id;
                        for (uint32_t i = 0; i < k; ++i) {
                            if (labels_c1[i] >= 0 && labels_c1[i] < static_cast<int64_t>(eligible_count)) {
                                ids_c1[i] = eligible_ids[labels_c1[i]]; // Remapeo estricto
                            }
                        }

                        auto eval_accuracy = [&](const std::vector<uint32_t>& res_ids, uint32_t cnt, uint32_t& rec_ok, uint32_t& rnk_ok) {
                            uint32_t m_set = 0;
                            for (uint32_t i = 0; i < cnt; ++i) {
                                for (uint32_t g = 0; g < gt_cnt; ++g) {
                                    if (res_ids[i] == gt[g].id) { m_set++; break; }
                                }
                            }
                            if (m_set == gt_cnt) rec_ok++;

                            bool rnk_match = (cnt == gt_cnt);
                            if (rnk_match) {
                                for (uint32_t i = 0; i < gt_cnt; ++i) {
                                    if (res_ids[i] != gt[i].id) { rnk_match = false; break; }
                                }
                            }
                            if (rnk_match) rnk_ok++;
                        };

                        eval_accuracy(ids_a1, cnt_a1, match_rec_a1, match_rnk_a1);
                        eval_accuracy(ids_a2, cnt_a2, match_rec_a2, match_rnk_a2);
                        eval_accuracy(ids_c1, k, match_rec_c1, match_rnk_c1);
                    }

                    LatencyStats st_a1 = calc_stats(lat_a1);
                    LatencyStats st_a2 = calc_stats(lat_a2);
                    LatencyStats st_c1 = calc_stats(lat_c1);

                    double speedup_a2_vs_c1 = st_c1.mean / st_a2.mean;
                    double speedup_a2_vs_a1 = st_a1.mean / st_a2.mean;
                    double speedup_a1_vs_c1 = st_c1.mean / st_a1.mean;

                    double rec_a1 = 100.0 * match_rec_a1 / num_queries_eval;
                    double rnk_a1 = 100.0 * match_rnk_a1 / num_queries_eval;
                    double rec_a2 = 100.0 * match_rec_a2 / num_queries_eval;
                    double rnk_a2 = 100.0 * match_rnk_a2 / num_queries_eval;
                    double rec_c1 = 100.0 * match_rec_c1 / num_queries_eval;
                    double rnk_c1 = 100.0 * match_rnk_c1 / num_queries_eval;

                    out_summary << std::fixed << std::setprecision(4)
                                << dim << ",Uniform," << sel << "," << eligible_count << ","
                                << k << "," << cache_mode << ","
                                << st_a1.mean << "," << st_a1.stddev << "," << st_a1.p50 << "," << st_a1.p95 << "," << st_a1.p99 << "," << st_a1.qps << ","
                                << st_a2.mean << "," << st_a2.stddev << "," << st_a2.p50 << "," << st_a2.p95 << "," << st_a2.p99 << "," << st_a2.qps << ","
                                << st_c1.mean << "," << st_c1.stddev << "," << st_c1.p50 << "," << st_c1.p95 << "," << st_c1.p99 << "," << st_c1.qps << ","
                                << speedup_a2_vs_c1 << "," << speedup_a2_vs_a1 << "," << speedup_a1_vs_c1 << ","
                                << mean_pure_dist_a2 << "," << mean_pure_dist_faiss << "," << pure_dist_speedup << ","
                                << rec_a1 << "," << rnk_a1 << ","
                                << rec_a2 << "," << rnk_a2 << ","
                                << rec_c1 << "," << rnk_c1 << "\n";
                    out_summary.flush();

                    std::cout << "[D=" << std::setw(3) << dim << " | Sel=" << std::setw(4) << sel << "% | "
                              << std::setw(15) << cache_mode << " | Top-" << std::setw(3) << k << "] "
                              << "A1=" << std::setw(7) << st_a1.mean << " us | "
                              << "A2=" << std::setw(7) << st_a2.mean << " us | "
                              << "C1=" << std::setw(7) << st_c1.mean << " us | "
                              << "A2/C1=" << std::setw(5) << speedup_a2_vs_c1 << "x | "
                              << "A2/A1=" << std::setw(5) << speedup_a2_vs_a1 << "x | "
                              << "PureDist A2/FAISS=" << std::setw(5) << pure_dist_speedup << "x" << std::endl;
                }
            }

            faiss_shim_free_flat(faiss_compact);
        }
    }

    out_summary.close();
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "EXPERIMENTO DE DECISION HITO 3.3.2 COMPLETADO EXITOSAMENTE" << std::endl;
    std::cout << "Resultados: results/hito3_3_2/kernel_isolation_summary.csv" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
