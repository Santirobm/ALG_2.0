#include "faiss_common.h"
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
#include <numeric>

using namespace alg_flat;

/*
 * ============================================================================
 * ORACULO INDEPENDIENTE DE REFERENCIA EN DOBLE PRECISION (IEEE-754 binary64)
 * ============================================================================
 * Mantiene la distancia acumulada, comparaciones y heap en double (64 bits)
 * sin truncar a float32 durante la seleccion o el ordenamiento.
 */
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
        return a.id < b.id; // Politica estricta y reproducible de desempate por menor ID
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
            uint32_t largest = left;
            if (right < k && less(data[left], data[right])) {
                largest = right;
            }
            if (!less(cand, data[largest])) break;
            data[i] = data[largest];
            i = largest;
        }
        data[i] = cand;
        threshold_sq = data[0].dist_sq;
    }

    inline void sort_ascending() {
        std::sort(data, data + size, less);
    }
};

static void compute_ground_truth_double(
    const float* raw_vectors,
    uint32_t num_vectors,
    uint32_t dim,
    const float* q,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    uint32_t k,
    DoubleNeighbor* out_gt) {

    DoubleTopKHeap heap;
    heap.init(k);

    for (uint32_t i = 0; i < eligible_count; ++i) {
        uint32_t id = eligible_ids[i];
        const float* v = raw_vectors + static_cast<size_t>(id) * dim;
        double dist = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            double diff = static_cast<double>(q[d]) - static_cast<double>(v[d]);
            dist += diff * diff;
        }
        heap.push(id, dist);
    }

    heap.sort_ascending();
    memcpy(out_gt, heap.data, heap.size * sizeof(DoubleNeighbor));
}

/*
 * ============================================================================
 * AUDITORES DE EXACTITUD Y CONCORDANCIA
 * ============================================================================
 */

/* RankMatch Estricto: compara posicion por posicion exactamente el ID */
static bool verify_strict_rank_match(const Neighbor* a, const DoubleNeighbor* gt, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) {
        if (a[i].id != gt[i].id) return false;
    }
    return true;
}

static bool verify_strict_rank_match(const Neighbor* a, const Neighbor* b, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) {
        if (a[i].id != b[i].id) return false;
    }
    return true;
}

/* IDSetRecall: recall de pertenencia no ordenado del Top-K */
static double compute_id_set_recall(const Neighbor* test, const DoubleNeighbor* gt, uint32_t k) {
    uint32_t matches = 0;
    for (uint32_t i = 0; i < k; ++i) {
        for (uint32_t j = 0; j < k; ++j) {
            if (test[i].id == gt[j].id) {
                matches++;
                break;
            }
        }
    }
    return (100.0 * matches) / k;
}

/* Error maximo absoluto en distancias al cuadrado frente a Double GT */
static double compute_max_dist_error(const Neighbor* test, const DoubleNeighbor* gt, uint32_t k) {
    double max_err = 0.0;
    for (uint32_t i = 0; i < k; ++i) {
        double err = std::fabs(static_cast<double>(test[i].dist_sq) - gt[i].dist_sq);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/*
 * ============================================================================
 * ESTIMADOR CANONICO DE CUANTILES TIPO 7 (Standard en NumPy, R, SciPy)
 * ============================================================================
 * Interpolacion lineal continua: h = (N - 1) * p; Q(p) = (1-f)*v[i] + f*v[i+1]
 */
static double compute_quantile(const std::vector<double>& sorted_v, double p) {
    if (sorted_v.empty()) return 0.0;
    if (sorted_v.size() == 1) return sorted_v[0];
    if (p <= 0.0) return sorted_v.front();
    if (p >= 1.0) return sorted_v.back();
    double h = static_cast<double>(sorted_v.size() - 1) * p;
    size_t i = static_cast<size_t>(h);
    double f = h - static_cast<double>(i);
    return (1.0 - f) * sorted_v[i] + f * sorted_v[i + 1];
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "ALG-Flat 2.0 vs FAISS (Hito 3.1: Integridad Nativa de Rendimiento y Exactitud)" << std::endl;
    std::cout << "  - Enlace: Directo C++ nativo (faiss.dll) sin CPython en el cronometro" << std::endl;
    std::cout << "  - Protocolo: 1 Hilo OpenMP forzado, CPU Pinning P-Core 0, Priority HIGH" << std::endl;
    std::cout << "  - Ground Truth: Double (binary64) estricto con heap independiente y ordenacion 64-bit" << std::endl;
    std::cout << "  - Metricas: StrictRankMatch@K (ID exacto), IDSetRecall@K, MaxDistError" << std::endl;
    std::cout << "  - Estadistica: Cuantiles Tipo-7 estandar (NumPy/R) y 100 consultas COLD por punto" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    ColdCacheEvictor evictor(36);

    std::string out_raw_csv = "results/hito3/raw_queries.csv";
    std::string out_summary_csv = "results/hito3/summary.csv";

    std::ofstream csv_raw(out_raw_csv);
    std::ofstream csv_sum(out_summary_csv);

    if (!csv_raw || !csv_sum) {
        std::cerr << "Error creando archivos de resultados en results/hito3/" << std::endl;
        return 1;
    }

    csv_raw << "seed,scenario,sigma,selectivity,top_k,cache_mode,query_id,execution_order,"
            << "time_alg_5a_us,time_faiss_flat_us,time_faiss_ivf16_us,time_faiss_ivf32_us,"
            << "strict_rankmatch_alg_vs_gt,strict_rankmatch_alg_vs_flat,strict_rankmatch_flat_vs_gt,"
            << "recall_alg_vs_gt,recall_flat_vs_gt,recall_ivf16_vs_gt,recall_ivf32_vs_gt,"
            << "max_dist_err_alg,max_dist_err_flat\n";

    csv_sum << "Scenario,Selectivity,TopK,CacheMode,Engine,Mean_us,p50_us,p95_us,p99_us,"
            << "Speedup_vs_Flat,StrictRankMatch_vs_GT_Pct,StrictRankMatch_vs_Flat_Pct,IDSetRecall_Pct,MaxDistErr\n";

    const uint32_t N = 100000U;
    const uint32_t dim = 64U;
    const uint32_t num_queries_warm = 100U;
    const uint32_t num_queries_cold = 100U;

    struct ScenarioConfig {
        std::string name;
        uint64_t seed;
        float sigma;
        MixMode mix;
    };

    std::vector<ScenarioConfig> scenarios = {
        {"Concentrated_s003", 8ULL, 0.03f, MIX_R},
        {"Medium_s009",       8ULL, 0.09f, MIX_R},
        {"Diffuse_s015",      9ULL, 0.15f, MIX_R},
        {"Bimodal_5050",      9ULL, 0.06f, MIX_50_50},
        {"Adversarial_OOD",   10ULL, 0.15f, MIX_D}
    };

    std::vector<float> selectivities = {1.0f, 5.0f, 10.0f, 15.0f, 25.0f};
    std::vector<uint32_t> top_k_list = {10U, 100U};

    DoubleNeighbor gt_topk[512];
    Neighbor res_alg[512], res_flat[512], res_ivf16[512], res_ivf32[512];
    uint32_t cnt_alg = 0, cnt_flat = 0, cnt_ivf16 = 0, cnt_ivf32 = 0;
    QueryExecutionMetrics met_alg;

    for (const auto& sc : scenarios) {
        std::cout << "\n>>> Cargando/Generando dataset: " << sc.name 
                  << " (Seed=" << sc.seed << ", sigma=" << sc.sigma << ")..." << std::endl;

        BenchmarkDataset ds;
        if (!generate_dataset_deterministic(sc.seed, N, num_queries_warm + num_queries_cold, 64U, sc.sigma, sc.mix, ds)) {
            std::cerr << "Error al generar dataset." << std::endl;
            return 1;
        }

        /* Configurar exactamente las 5 selectividades requeridas para Hito 3 */
        ds.header.num_selectivities = static_cast<uint32_t>(selectivities.size());
        for (size_t s = 0; s < selectivities.size(); ++s) {
            ds.header.selectivities_deci_percent[s] = static_cast<uint32_t>(selectivities[s] * 10.0f + 0.5f);
        }

        std::cout << ">>> Construyendo indice ALG Plan 5A (IVFPartitionedIndex, P=128)..." << std::endl;
        IVFPartitionedIndex idx_alg;
        if (!build_ivf_partitioned_index(ds, 128U, 15U, idx_alg)) {
            std::cerr << "Error al construir IVFPartitionedIndex." << std::endl;
            return 1;
        }

        std::cout << ">>> Inicializando motor FAISS C++ Nativo (IndexFlatL2 + IndexIVFFlat)..." << std::endl;
        FaissNativeEngine faiss_engine;
        if (!faiss_engine.init(dim, N, ds.vectors)) {
            std::cerr << "Error al inicializar FaissNativeEngine." << std::endl;
            return 1;
        }

        for (size_t s_idx = 0; s_idx < selectivities.size(); ++s_idx) {
            float sel = selectivities[s_idx];
            uint32_t target_cat = ds.header.selectivities_deci_percent[s_idx] * 10U;
            std::vector<uint32_t> eligible_ids;
            eligible_ids.reserve(N / 3);
            for (uint32_t id = 0; id < N; ++id) {
                if (ds.categories[id] < target_cat) {
                    eligible_ids.push_back(id);
                }
            }

            QueryPredicate pred;
            pred.target_category = target_cat;
            pred.eligible_ids = eligible_ids.data();
            pred.eligible_count = static_cast<uint32_t>(eligible_ids.size());
            pred.sel_index = static_cast<uint32_t>(s_idx);

            /* Actualizar IDSelectorBitmap en FAISS fuera del bucle de cronometraje */
            size_t bmp_bytes = (N + 7) / 8;
            std::vector<uint8_t> bitmap(bmp_bytes, 0);
            for (uint32_t id : eligible_ids) {
                bitmap[id >> 3] |= (1U << (id & 7));
            }
            faiss_engine.update_bitmap_filter(bitmap.data(), bmp_bytes);

            for (uint32_t k : top_k_list) {
                for (const std::string& cache_mode : {"WARM_CACHE", "COLD_LLC_FLUSH"}) {
                    uint32_t q_start = (cache_mode == "WARM_CACHE") ? 0 : num_queries_warm;
                    uint32_t q_count = (cache_mode == "WARM_CACHE") ? num_queries_warm : num_queries_cold;

                    std::vector<double> times_alg, times_flat, times_ivf16, times_ivf32;
                    times_alg.reserve(q_count);
                    times_flat.reserve(q_count);
                    times_ivf16.reserve(q_count);
                    times_ivf32.reserve(q_count);

                    uint32_t strict_match_alg_gt = 0;
                    uint32_t strict_match_alg_flat = 0;
                    uint32_t strict_match_flat_gt = 0;

                    double sum_rec_alg_gt = 0.0;
                    double sum_rec_flat_gt = 0.0;
                    double sum_rec_ivf16_gt = 0.0;
                    double sum_rec_ivf32_gt = 0.0;

                    double max_dist_err_alg_total = 0.0;
                    double max_dist_err_flat_total = 0.0;

                    for (uint32_t q = q_start; q < q_start + q_count; ++q) {
                        const float* q_vec = ds.queries[q].values;

                        /* Ground Truth independiente en double (64 bits) */
                        compute_ground_truth_double(ds.vectors, N, dim, q_vec, eligible_ids.data(), pred.eligible_count, k, gt_topk);

                        uint32_t order_mode = q % 3U;
                        std::string order_str;
                        double t_alg = 0.0, t_flat = 0.0, t_ivf16 = 0.0, t_ivf32 = 0.0;

                        if (cache_mode == "WARM_CACHE") {
                            if (order_mode == 0) {
                                order_str = "ALG->Flat->IVF";
                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;

                                faiss_engine.search_flat(q_vec, k, res_flat, &cnt_flat, t_flat);

                                faiss_engine.set_ivf_nprobe(16);
                                faiss_engine.search_ivf(q_vec, k, res_ivf16, &cnt_ivf16, t_ivf16);

                                faiss_engine.set_ivf_nprobe(32);
                                faiss_engine.search_ivf(q_vec, k, res_ivf32, &cnt_ivf32, t_ivf32);
                            } else if (order_mode == 1) {
                                order_str = "Flat->IVF->ALG";
                                faiss_engine.search_flat(q_vec, k, res_flat, &cnt_flat, t_flat);

                                faiss_engine.set_ivf_nprobe(16);
                                faiss_engine.search_ivf(q_vec, k, res_ivf16, &cnt_ivf16, t_ivf16);

                                faiss_engine.set_ivf_nprobe(32);
                                faiss_engine.search_ivf(q_vec, k, res_ivf32, &cnt_ivf32, t_ivf32);

                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;
                            } else {
                                order_str = "IVF->ALG->Flat";
                                faiss_engine.set_ivf_nprobe(16);
                                faiss_engine.search_ivf(q_vec, k, res_ivf16, &cnt_ivf16, t_ivf16);

                                faiss_engine.set_ivf_nprobe(32);
                                faiss_engine.search_ivf(q_vec, k, res_ivf32, &cnt_ivf32, t_ivf32);

                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;

                                faiss_engine.search_flat(q_vec, k, res_flat, &cnt_flat, t_flat);
                            }
                        } else {
                            /* COLD_LLC_FLUSH con eviccion estricta de 36 MB entre cada motor */
                            if (order_mode == 0) {
                                order_str = "ALG->Flat->IVF";
                                evictor.flush_llc();
                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;

                                evictor.flush_llc();
                                faiss_engine.search_flat(q_vec, k, res_flat, &cnt_flat, t_flat);

                                evictor.flush_llc();
                                faiss_engine.set_ivf_nprobe(16);
                                faiss_engine.search_ivf(q_vec, k, res_ivf16, &cnt_ivf16, t_ivf16);

                                evictor.flush_llc();
                                faiss_engine.set_ivf_nprobe(32);
                                faiss_engine.search_ivf(q_vec, k, res_ivf32, &cnt_ivf32, t_ivf32);
                            } else if (order_mode == 1) {
                                order_str = "Flat->IVF->ALG";
                                evictor.flush_llc();
                                faiss_engine.search_flat(q_vec, k, res_flat, &cnt_flat, t_flat);

                                evictor.flush_llc();
                                faiss_engine.set_ivf_nprobe(16);
                                faiss_engine.search_ivf(q_vec, k, res_ivf16, &cnt_ivf16, t_ivf16);

                                evictor.flush_llc();
                                faiss_engine.set_ivf_nprobe(32);
                                faiss_engine.search_ivf(q_vec, k, res_ivf32, &cnt_ivf32, t_ivf32);

                                evictor.flush_llc();
                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;
                            } else {
                                order_str = "IVF->ALG->Flat";
                                evictor.flush_llc();
                                faiss_engine.set_ivf_nprobe(16);
                                faiss_engine.search_ivf(q_vec, k, res_ivf16, &cnt_ivf16, t_ivf16);

                                evictor.flush_llc();
                                faiss_engine.set_ivf_nprobe(32);
                                faiss_engine.search_ivf(q_vec, k, res_ivf32, &cnt_ivf32, t_ivf32);

                                evictor.flush_llc();
                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;

                                evictor.flush_llc();
                                faiss_engine.search_flat(q_vec, k, res_flat, &cnt_flat, t_flat);
                            }
                        }

                        /* Auditorias estrictas */
                        bool m_alg_gt = verify_strict_rank_match(res_alg, gt_topk, k);
                        bool m_alg_flat = verify_strict_rank_match(res_alg, res_flat, k);
                        bool m_flat_gt = verify_strict_rank_match(res_flat, gt_topk, k);

                        double rec_alg_gt = compute_id_set_recall(res_alg, gt_topk, k);
                        double rec_flat_gt = compute_id_set_recall(res_flat, gt_topk, k);
                        double rec_ivf16_gt = compute_id_set_recall(res_ivf16, gt_topk, k);
                        double rec_ivf32_gt = compute_id_set_recall(res_ivf32, gt_topk, k);

                        double err_alg = compute_max_dist_error(res_alg, gt_topk, k);
                        double err_flat = compute_max_dist_error(res_flat, gt_topk, k);

                        if (m_alg_gt) strict_match_alg_gt++;
                        if (m_alg_flat) strict_match_alg_flat++;
                        if (m_flat_gt) strict_match_flat_gt++;

                        sum_rec_alg_gt += rec_alg_gt;
                        sum_rec_flat_gt += rec_flat_gt;
                        sum_rec_ivf16_gt += rec_ivf16_gt;
                        sum_rec_ivf32_gt += rec_ivf32_gt;

                        if (err_alg > max_dist_err_alg_total) max_dist_err_alg_total = err_alg;
                        if (err_flat > max_dist_err_flat_total) max_dist_err_flat_total = err_flat;

                        times_alg.push_back(t_alg);
                        times_flat.push_back(t_flat);
                        times_ivf16.push_back(t_ivf16);
                        times_ivf32.push_back(t_ivf32);

                        csv_raw << sc.seed << "," << sc.name << "," << sc.sigma << ","
                                << sel << "," << k << "," << cache_mode << "," << q << "," << order_str << ","
                                << t_alg << "," << t_flat << "," << t_ivf16 << "," << t_ivf32 << ","
                                << (m_alg_gt ? 1 : 0) << "," << (m_alg_flat ? 1 : 0) << "," << (m_flat_gt ? 1 : 0) << ","
                                << rec_alg_gt << "," << rec_flat_gt << "," << rec_ivf16_gt << "," << rec_ivf32_gt << ","
                                << err_alg << "," << err_flat << "\n";
                    }

                    auto get_stats = [](std::vector<double>& v, double& mean, double& p50, double& p95, double& p99) {
                        std::sort(v.begin(), v.end());
                        double sum = std::accumulate(v.begin(), v.end(), 0.0);
                        mean = sum / v.size();
                        p50 = compute_quantile(v, 0.50);
                        p95 = compute_quantile(v, 0.95);
                        p99 = compute_quantile(v, 0.99);
                    };

                    double m_alg, p50_alg, p95_alg, p99_alg;
                    double m_flat, p50_flat, p95_flat, p99_flat;
                    double m_ivf16, p50_ivf16, p95_ivf16, p99_ivf16;
                    double m_ivf32, p50_ivf32, p95_ivf32, p99_ivf32;

                    get_stats(times_alg, m_alg, p50_alg, p95_alg, p99_alg);
                    get_stats(times_flat, m_flat, p50_flat, p95_flat, p99_flat);
                    get_stats(times_ivf16, m_ivf16, p50_ivf16, p95_ivf16, p99_ivf16);
                    get_stats(times_ivf32, m_ivf32, p50_ivf32, p95_ivf32, p99_ivf32);

                    double speedup_alg = m_flat / m_alg;
                    double speedup_ivf16 = m_flat / m_ivf16;
                    double speedup_ivf32 = m_flat / m_ivf32;

                    double pct_strict_alg_gt = (100.0 * strict_match_alg_gt) / q_count;
                    double pct_strict_alg_flat = (100.0 * strict_match_alg_flat) / q_count;
                    double pct_strict_flat_gt = (100.0 * strict_match_flat_gt) / q_count;

                    double mean_rec_alg = sum_rec_alg_gt / q_count;
                    double mean_rec_flat = sum_rec_flat_gt / q_count;
                    double mean_rec_ivf16 = sum_rec_ivf16_gt / q_count;
                    double mean_rec_ivf32 = sum_rec_ivf32_gt / q_count;

                    /* Escribir filas al summary CSV */
                    csv_sum << sc.name << "," << sel << "," << k << "," << cache_mode << ",ALG_Plan5A,"
                            << m_alg << "," << p50_alg << "," << p95_alg << "," << p99_alg << ","
                            << speedup_alg << "," << pct_strict_alg_gt << "," << pct_strict_alg_flat << ","
                            << mean_rec_alg << "," << max_dist_err_alg_total << "\n";

                    csv_sum << sc.name << "," << sel << "," << k << "," << cache_mode << ",FAISS_Flat,"
                            << m_flat << "," << p50_flat << "," << p95_flat << "," << p99_flat << ","
                            << 1.0 << "," << pct_strict_flat_gt << ",100.0,"
                            << mean_rec_flat << "," << max_dist_err_flat_total << "\n";

                    csv_sum << sc.name << "," << sel << "," << k << "," << cache_mode << ",FAISS_IVF16,"
                            << m_ivf16 << "," << p50_ivf16 << "," << p95_ivf16 << "," << p99_ivf16 << ","
                            << speedup_ivf16 << ",0.0,0.0," << mean_rec_ivf16 << ",0.0\n";

                    csv_sum << sc.name << "," << sel << "," << k << "," << cache_mode << ",FAISS_IVF32,"
                            << m_ivf32 << "," << p50_ivf32 << "," << p95_ivf32 << "," << p99_ivf32 << ","
                            << speedup_ivf32 << ",0.0,0.0," << mean_rec_ivf32 << ",0.0\n";

                    std::cout << "[" << sc.name << " | " << cache_mode << " | Sel=" << sel << "% | Top-" << k << "] "
                              << "ALG 5A=" << std::fixed << std::setprecision(1) << m_alg << " us (p95=" << p95_alg << ") | "
                              << "FAISS Flat=" << m_flat << " us | "
                              << "Speedup=" << std::setprecision(2) << speedup_alg << "x | "
                              << "StrictMatch(ALG,GT)=" << std::setprecision(1) << pct_strict_alg_gt << "% | "
                              << "StrictMatch(ALG,Flat)=" << pct_strict_alg_flat << "% | "
                              << "IVF32 Rec=" << mean_rec_ivf32 << "% (" << m_ivf32 << " us)"
                              << std::endl;
                }
            }
        }
    }

    csv_raw.close();
    csv_sum.close();

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "BENCHMARK HITO 3.1 NATIVO COMPLETADO EXITOSAMENTE" << std::endl;
    std::cout << "Resultados crudos: " << out_raw_csv << std::endl;
    std::cout << "Resumen canonico: " << out_summary_csv << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
