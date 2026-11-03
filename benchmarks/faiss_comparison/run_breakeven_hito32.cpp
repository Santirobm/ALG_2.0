#include "faiss_c_shim.h"
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

static inline uint64_t get_highres_time_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

/*
 * Oraculo Double (binary64) para certificar exactitud
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
            uint32_t largest = left;
            if (right < k && less(data[left], data[right])) largest = right;
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

static inline void set_dynamic_selectivity_slice(
    IVFPartitionedIndex& idx,
    uint32_t sel_idx,
    uint32_t target_cat) {

    if (idx.eligible_ivf_ids_per_sel.size() <= sel_idx) {
        idx.eligible_ivf_ids_per_sel.resize(sel_idx + 1);
        idx.eligible_part_offsets_per_sel.resize(sel_idx + 1);
    }
    auto& el_ids = idx.eligible_ivf_ids_per_sel[sel_idx];
    auto& part_offs = idx.eligible_part_offsets_per_sel[sel_idx];
    el_ids.clear();
    el_ids.reserve(idx.num_vectors / 2);
    part_offs.resize(idx.num_partitions + 1);

    for (uint32_t c = 0; c < idx.num_partitions; ++c) {
        part_offs[c] = static_cast<uint32_t>(el_ids.size());
        uint32_t p_start = idx.partitions[c].aos_offset;
        uint32_t p_end = p_start + idx.partitions[c].count;
        for (uint32_t idx_v = p_start; idx_v < p_end; ++idx_v) {
            if (idx.part_aos_categories[idx_v] < target_cat) {
                el_ids.push_back(idx_v);
            }
        }
    }
    part_offs[idx.num_partitions] = static_cast<uint32_t>(el_ids.size());
}

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
    std::cout << "HITO 3.2: CURVA DE BREAK-EVEN SELECTIVITY Y ESCALABILIDAD (ALG vs FAISS SHIM)" << std::endl;
    std::cout << "  - Grilla de 16 selectividades: 0% .. 100%" << std::endl;
    std::cout << "  - Motor FAISS: C++ Shim tipado (faiss_shim_search_flat con IDSelectorBitmap)" << std::endl;
    std::cout << "  - Metricas: Latencias (p50/p95/p99), QPS, Pruning Ratio, Particiones, Bytes" << std::endl;
    std::cout << "  - Protocolo: P-Core 0 pinning, HIGH priority, 1 hilo OpenMP, LLC 36MB flush" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    ColdCacheEvictor evictor(36);

    std::string out_raw_csv = "results/hito3/breakeven_raw.csv";
    std::string out_summary_csv = "results/hito3/breakeven_summary.csv";

    std::ofstream csv_raw(out_raw_csv);
    std::ofstream csv_sum(out_summary_csv);

    if (!csv_raw || !csv_sum) {
        std::cerr << "Error creando archivos en results/hito3/" << std::endl;
        return 1;
    }

    csv_raw << "selectivity,top_k,cache_mode,query_id,order_str,"
            << "time_alg_us,time_faiss_us,dists_eval_alg,eligible_total,parts_visited,bytes_read_alg,"
            << "strict_rankmatch,id_set_recall,max_dist_err\n";

    csv_sum << "Selectivity,TopK,CacheMode,Speedup_vs_Flat,Mean_ALG_us,p50_ALG_us,p95_ALG_us,p99_ALG_us,"
            << "Mean_FAISS_us,p50_FAISS_us,p95_FAISS_us,p99_FAISS_us,"
            << "QPS_ALG,QPS_FAISS,Avg_Dists_Eval,Eligible_Total,Pruning_Ratio,Parts_Visited_Avg,StrictRankMatch_Pct,IDSetRecall_Pct\n";

    const uint32_t N = 100000U;
    const uint32_t dim = 64U;
    const uint32_t num_queries_warm = 100U;
    const uint32_t num_queries_cold = 100U;

    std::vector<float> selectivities = {
        0.0f, 0.05f, 0.1f, 0.25f, 0.5f, 1.0f, 2.0f, 5.0f,
        10.0f, 15.0f, 20.0f, 30.0f, 40.0f, 50.0f, 75.0f, 100.0f
    };
    std::vector<uint32_t> top_k_list = {10U, 100U};

    // Dataset estandar de evaluacion (Medium_s009, semilla 8)
    std::cout << "\n>>> Generando dataset base determinista (N=100K, sigma=0.09)..." << std::endl;
    BenchmarkDataset ds;
    if (!generate_dataset_deterministic(8ULL, N, num_queries_warm + num_queries_cold, dim, 0.09f, MIX_R, ds)) {
        std::cerr << "Error generando dataset." << std::endl;
        return 1;
    }

    std::cout << ">>> Construyendo indice ALG Plan 5A (IVFPartitionedIndex, P=128)..." << std::endl;
    IVFPartitionedIndex idx_alg;
    if (!build_ivf_partitioned_index(ds, 128U, 15U, idx_alg)) {
        std::cerr << "Error construyendo IVFPartitionedIndex." << std::endl;
        return 1;
    }

    std::cout << ">>> Inicializando FAISS IndexFlatL2 a traves de la capa C Shim..." << std::endl;
    FaissFlatHandle faiss_flat = faiss_shim_create_flat_l2(dim);
    if (!faiss_flat || !faiss_shim_flat_add(faiss_flat, N, ds.vectors)) {
        std::cerr << "Error inicializando FAISS Flat en C Shim." << std::endl;
        return 1;
    }

    // Pre-indexar todos los 16 slices de selectividad en ALG
    for (size_t s_idx = 0; s_idx < selectivities.size(); ++s_idx) {
        float sel = selectivities[s_idx];
        uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);
        if (sel >= 100.0f) target_cat = 10001U; // Selecciona el 100%
        set_dynamic_selectivity_slice(idx_alg, static_cast<uint32_t>(s_idx), target_cat);
    }

    DoubleNeighbor gt_topk[512];
    Neighbor res_alg[512];
    float dists_faiss[512];
    int64_t labels_faiss[512];
    uint32_t cnt_alg = 0;
    QueryExecutionMetrics met_alg;

    size_t bmp_bytes = (N + 7) / 8;
    std::vector<uint8_t> bitmap(bmp_bytes, 0);

    for (size_t s_idx = 0; s_idx < selectivities.size(); ++s_idx) {
        float sel = selectivities[s_idx];
        uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);
        if (sel >= 100.0f) target_cat = 10001U;

        std::vector<uint32_t> eligible_ids;
        eligible_ids.reserve(N);
        memset(bitmap.data(), 0, bmp_bytes);

        for (uint32_t id = 0; id < N; ++id) {
            if (ds.categories[id] < target_cat) {
                eligible_ids.push_back(id);
                bitmap[id >> 3] |= (1U << (id & 7));
            }
        }

        QueryPredicate pred;
        pred.target_category = target_cat;
        pred.eligible_ids = eligible_ids.data();
        pred.eligible_count = static_cast<uint32_t>(eligible_ids.size());
        pred.sel_index = static_cast<uint32_t>(s_idx);

        // Actualizar selector y parametros FAISS
        FaissSelectorHandle sel_handle = faiss_shim_create_bitmap_selector(N, bitmap.data());
        FaissParamsHandle params_flat = faiss_shim_create_flat_params(sel_handle);

        for (uint32_t k : top_k_list) {
            for (const std::string& cache_mode : {"WARM_CACHE", "COLD_LLC_FLUSH"}) {
                uint32_t q_start = (cache_mode == "WARM_CACHE") ? 0 : num_queries_warm;
                uint32_t q_count = (cache_mode == "WARM_CACHE") ? num_queries_warm : num_queries_cold;

                std::vector<double> times_alg, times_faiss;
                times_alg.reserve(q_count);
                times_faiss.reserve(q_count);

                uint64_t sum_dists_eval = 0;
                uint64_t sum_parts_visited = 0;
                uint32_t strict_matches = 0;
                double sum_recall = 0.0;
                double max_err_total = 0.0;

                for (uint32_t q = q_start; q < q_start + q_count; ++q) {
                    const float* q_vec = ds.queries[q].values;

                    // Ground Truth en double
                    compute_ground_truth_double(ds.vectors, N, dim, q_vec, eligible_ids.data(), pred.eligible_count, k, gt_topk);

                    uint32_t order_mode = q % 2U;
                    std::string order_str;
                    double t_alg = 0.0, t_faiss = 0.0;

                    if (cache_mode == "WARM_CACHE") {
                        if (order_mode == 0) {
                            order_str = "ALG->FAISS";
                            execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                            t_alg = met_alg.total_us;

                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                            uint64_t t1 = get_highres_time_ns();
                            t_faiss = static_cast<double>(t1 - t0) / 1000.0;
                        } else {
                            order_str = "FAISS->ALG";
                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                            uint64_t t1 = get_highres_time_ns();
                            t_faiss = static_cast<double>(t1 - t0) / 1000.0;

                            execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                            t_alg = met_alg.total_us;
                        }
                    } else {
                        // COLD_LLC_FLUSH con eviccion de 36 MB
                        if (order_mode == 0) {
                            order_str = "ALG->FAISS";
                            evictor.flush_llc();
                            execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                            t_alg = met_alg.total_us;

                            evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                            uint64_t t1 = get_highres_time_ns();
                            t_faiss = static_cast<double>(t1 - t0) / 1000.0;
                        } else {
                            order_str = "FAISS->ALG";
                            evictor.flush_llc();
                            uint64_t t0 = get_highres_time_ns();
                            faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                            uint64_t t1 = get_highres_time_ns();
                            t_faiss = static_cast<double>(t1 - t0) / 1000.0;

                            evictor.flush_llc();
                            execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                            t_alg = met_alg.total_us;
                        }
                    }

                    times_alg.push_back(t_alg);
                    times_faiss.push_back(t_faiss);
                    sum_dists_eval += met_alg.dists_evaluated;
                    sum_parts_visited += met_alg.partitions_visited;

                    // Audit exactness vs double GT
                    uint32_t effective_k = std::min(k, pred.eligible_count);
                    bool strict_ok = true;
                    uint32_t matches = 0;
                    double q_max_err = 0.0;

                    for (uint32_t i = 0; i < effective_k; ++i) {
                        if (res_alg[i].id != gt_topk[i].id) strict_ok = false;
                        for (uint32_t j = 0; j < effective_k; ++j) {
                            if (res_alg[i].id == gt_topk[j].id) { matches++; break; }
                        }
                        double err = std::fabs(static_cast<double>(res_alg[i].dist_sq) - gt_topk[i].dist_sq);
                        if (err > q_max_err) q_max_err = err;
                    }
                    if (strict_ok) strict_matches++;
                    if (effective_k > 0) sum_recall += (100.0 * matches) / effective_k;
                    else sum_recall += 100.0;

                    if (q_max_err > max_err_total) max_err_total = q_max_err;

                    csv_raw << sel << "," << k << "," << cache_mode << "," << q << "," << order_str << ","
                            << t_alg << "," << t_faiss << ","
                            << met_alg.dists_evaluated << "," << pred.eligible_count << ","
                            << met_alg.partitions_visited << "," << met_alg.bytes_read << ","
                            << (strict_ok ? 1 : 0) << "," << ((effective_k > 0) ? (100.0 * matches) / effective_k : 100.0) << ","
                            << q_max_err << "\n";
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
                double m_faiss, p50_faiss, p95_faiss, p99_faiss;

                get_stats(times_alg, m_alg, p50_alg, p95_alg, p99_alg);
                get_stats(times_faiss, m_faiss, p50_faiss, p95_faiss, p99_faiss);

                double speedup = m_faiss / m_alg;
                double qps_alg = 1000000.0 / m_alg;
                double qps_faiss = 1000000.0 / m_faiss;
                double avg_dists = static_cast<double>(sum_dists_eval) / q_count;
                double avg_parts = static_cast<double>(sum_parts_visited) / q_count;
                double pruning_ratio = (pred.eligible_count > 0) ? (1.0 - (avg_dists / pred.eligible_count)) * 100.0 : 100.0;
                double pct_strict = (100.0 * strict_matches) / q_count;
                double pct_recall = sum_recall / q_count;

                csv_sum << sel << "," << k << "," << cache_mode << "," << speedup << ","
                        << m_alg << "," << p50_alg << "," << p95_alg << "," << p99_alg << ","
                        << m_faiss << "," << p50_faiss << "," << p95_faiss << "," << p99_faiss << ","
                        << qps_alg << "," << qps_faiss << ","
                        << avg_dists << "," << pred.eligible_count << "," << pruning_ratio << ","
                        << avg_parts << "," << pct_strict << "," << pct_recall << "\n";

                std::cout << "[Sel=" << std::setw(5) << sel << "% | " << cache_mode << " | Top-" << k << "] "
                          << "ALG=" << std::fixed << std::setprecision(1) << m_alg << " us (p95=" << p95_alg << ") | "
                          << "FAISS=" << m_faiss << " us | "
                          << "Speedup=" << std::setprecision(2) << speedup << "x | "
                          << "Pruned=" << std::setprecision(1) << pruning_ratio << "% ("
                          << static_cast<uint32_t>(avg_dists) << "/" << pred.eligible_count << " evals) | "
                          << "Recall=" << pct_recall << "%"
                          << std::endl;
            }
        }

        faiss_shim_free_params(params_flat);
        faiss_shim_free_selector(sel_handle);
    }

    faiss_shim_free_flat(faiss_flat);
    csv_raw.close();
    csv_sum.close();

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "CURVA DE BREAK-EVEN COMPLETADA CON EXITO" << std::endl;
    std::cout << "Raw: " << out_raw_csv << std::endl;
    std::cout << "Summary: " << out_summary_csv << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
