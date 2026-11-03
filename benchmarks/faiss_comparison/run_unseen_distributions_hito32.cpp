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
#include <random>
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

/*
 * Generadores de Distribuciones Desconocidas (No Vistas)
 */
enum UnseenDistributionType {
    DIST_UNIFORM = 0,
    DIST_ANISOTROPIC,
    DIST_HEAVY_TAIL_STUDENT_T,
    DIST_CORRELATED_AUTOREGRESSIVE,
    DIST_DEGENERATE_LOW_RANK,
    DIST_MULTIMODAL_100_CLUSTERS
};

static void generate_unseen_dataset(
    UnseenDistributionType dist_type,
    uint32_t N,
    uint32_t dim,
    uint32_t n_queries,
    uint64_t seed,
    BenchmarkDataset& out_ds) {

    memset(&out_ds.header, 0, sizeof(DatasetBinaryHeader));
    out_ds.header.num_vectors = N;
    out_ds.header.dimension = dim;
    out_ds.header.num_queries = n_queries;
    out_ds.header.seed = seed;

    out_ds.vectors = static_cast<float*>(_aligned_malloc(static_cast<size_t>(N) * dim * sizeof(float), 64));
    out_ds.categories = static_cast<uint32_t*>(_aligned_malloc(static_cast<size_t>(N) * sizeof(uint32_t), 64));
    out_ds.queries.resize(n_queries);

    std::mt19937_64 rng(seed);

    std::uniform_real_distribution<float> uniform_dist(-1.0f, 1.0f);
    std::normal_distribution<float> normal_dist(0.0f, 1.0f);
    std::student_t_distribution<float> student_t_dist(3.0f); // 3 grados de libertad (colas pesadas)
    std::uniform_int_distribution<uint32_t> cat_dist(0, 9999);

    std::vector<float> cluster_centers(100 * dim);
    if (dist_type == DIST_MULTIMODAL_100_CLUSTERS) {
        for (size_t i = 0; i < cluster_centers.size(); ++i) {
            cluster_centers[i] = uniform_dist(rng) * 5.0f;
        }
    }

    for (uint32_t i = 0; i < N; ++i) {
        float* v = out_ds.vectors + static_cast<size_t>(i) * dim;
        out_ds.categories[i] = cat_dist(rng);

        switch (dist_type) {
            case DIST_UNIFORM:
                for (uint32_t d = 0; d < dim; ++d) v[d] = uniform_dist(rng);
                break;
            case DIST_ANISOTROPIC:
                for (uint32_t d = 0; d < dim; ++d) {
                    float sigma_d = 0.02f * static_cast<float>(d + 1);
                    v[d] = normal_dist(rng) * sigma_d;
                }
                break;
            case DIST_HEAVY_TAIL_STUDENT_T:
                for (uint32_t d = 0; d < dim; ++d) {
                    v[d] = student_t_dist(rng) * 0.1f;
                }
                break;
            case DIST_CORRELATED_AUTOREGRESSIVE: {
                float val = 0.0f;
                for (uint32_t d = 0; d < dim; ++d) {
                    val = 0.85f * val + 0.15f * normal_dist(rng);
                    v[d] = val;
                }
                break;
            }
            case DIST_DEGENERATE_LOW_RANK:
                for (uint32_t d = 0; d < dim; ++d) {
                    if (d < 8) v[d] = normal_dist(rng);
                    else v[d] = normal_dist(rng) * 1.0e-5f; // Ruido residual
                }
                break;
            case DIST_MULTIMODAL_100_CLUSTERS: {
                uint32_t c = rng() % 100;
                const float* center = cluster_centers.data() + c * dim;
                for (uint32_t d = 0; d < dim; ++d) {
                    v[d] = center[d] + normal_dist(rng) * 0.05f;
                }
                break;
            }
        }
    }

    // Generar consultas muestreadas del mismo proceso estocastico
    for (uint32_t q = 0; q < n_queries; ++q) {
        float* qv = out_ds.queries[q].values;
        uint32_t ref_id = rng() % N;
        const float* ref_v = out_ds.vectors + static_cast<size_t>(ref_id) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            qv[d] = ref_v[d] + normal_dist(rng) * 0.02f;
        }
    }
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "HITO 3.2: EVALUACION EN 6 DISTRIBUCIONES DESCONOCIDAS (UNSEEN DISTRIBUTIONS)" << std::endl;
    std::cout << "  - Distribuciones no vistas: Uniforme, Anisotropica, Colas Pesadas (Student-t)," << std::endl;
    std::cout << "    Correlacionada AR(1), Rango Degenerado (8D/64D), Multimodal (100 Clusters)" << std::endl;
    std::cout << "  - Exactitud: Auditoria formal 100% IDSetRecall vs Double Ground Truth" << std::endl;
    std::cout << "  - Protocolo: P-Core 0, HIGH priority, 1 hilo OpenMP, LLC 36MB flush" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    ColdCacheEvictor evictor(36);

    std::string out_csv = "results/hito3/unseen_distributions_summary.csv";
    std::ofstream csv(out_csv);
    if (!csv) {
        std::cerr << "Error creando archivo " << out_csv << std::endl;
        return 1;
    }

    csv << "Distribution,Selectivity,TopK,CacheMode,Speedup_vs_Flat,"
        << "Mean_ALG_us,p50_ALG_us,p95_ALG_us,p99_ALG_us,"
        << "Mean_FAISS_us,p50_FAISS_us,p95_FAISS_us,p99_FAISS_us,"
        << "QPS_ALG,QPS_FAISS,Avg_Dists_Eval,Eligible_Total,Pruning_Ratio,StrictRankMatch_Pct,IDSetRecall_Pct\n";

    const uint32_t N = 100000U;
    const uint32_t dim = 64U;
    const uint32_t num_queries_warm = 80U;
    const uint32_t num_queries_cold = 40U;

    struct UnseenDistConfig {
        std::string name;
        UnseenDistributionType type;
    };

    std::vector<UnseenDistConfig> dist_configs = {
        {"Uniform_Hypercube",         DIST_UNIFORM},
        {"Anisotropic_Gaussian",      DIST_ANISOTROPIC},
        {"Heavy_Tailed_StudentT",     DIST_HEAVY_TAIL_STUDENT_T},
        {"Correlated_AR1",            DIST_CORRELATED_AUTOREGRESSIVE},
        {"Degenerate_LowRank_8D",     DIST_DEGENERATE_LOW_RANK},
        {"Multimodal_100_Clusters",   DIST_MULTIMODAL_100_CLUSTERS}
    };

    std::vector<float> selectivities = {1.0f, 5.0f, 10.0f, 25.0f};
    std::vector<uint32_t> top_k_list = {10U, 100U};

    DoubleNeighbor gt_topk[512];
    Neighbor res_alg[512];
    float dists_faiss[512];
    int64_t labels_faiss[512];
    uint32_t cnt_alg = 0;
    QueryExecutionMetrics met_alg;

    for (const auto& dist : dist_configs) {
        std::cout << "\n================================================================================" << std::endl;
        std::cout << ">>> Evaluando Distribucion Desconocida: " << dist.name << "..." << std::endl;
        std::cout << "================================================================================" << std::endl;

        BenchmarkDataset ds;
        generate_unseen_dataset(dist.type, N, dim, num_queries_warm + num_queries_cold, 42ULL, ds);

        IVFPartitionedIndex idx_alg;
        if (!build_ivf_partitioned_index(ds, 128U, 15U, idx_alg)) {
            std::cerr << "Error construyendo IVFPartitionedIndex." << std::endl;
            return 1;
        }

        FaissFlatHandle faiss_flat = faiss_shim_create_flat_l2(dim);
        faiss_shim_flat_add(faiss_flat, N, ds.vectors);

        for (size_t s_idx = 0; s_idx < selectivities.size(); ++s_idx) {
            float sel = selectivities[s_idx];
            uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);
            set_dynamic_selectivity_slice(idx_alg, static_cast<uint32_t>(s_idx), target_cat);
        }

        size_t bmp_bytes = (N + 7) / 8;
        std::vector<uint8_t> bitmap(bmp_bytes, 0);

        for (size_t s_idx = 0; s_idx < selectivities.size(); ++s_idx) {
            float sel = selectivities[s_idx];
            uint32_t target_cat = static_cast<uint32_t>(sel * 100.0f + 0.5f);

            std::vector<uint32_t> eligible_ids;
            eligible_ids.reserve(N / 3);
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
                    uint32_t strict_matches = 0;
                    double sum_recall = 0.0;

                    for (uint32_t q = q_start; q < q_start + q_count; ++q) {
                        const float* q_vec = ds.queries[q].values;

                        compute_ground_truth_double(ds.vectors, N, dim, q_vec, eligible_ids.data(), pred.eligible_count, k, gt_topk);

                        uint32_t order_mode = q % 2U;
                        double t_alg = 0.0, t_faiss = 0.0;

                        if (cache_mode == "WARM_CACHE") {
                            if (order_mode == 0) {
                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;

                                uint64_t t0 = get_highres_time_ns();
                                faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                                uint64_t t1 = get_highres_time_ns();
                                t_faiss = static_cast<double>(t1 - t0) / 1000.0;
                            } else {
                                uint64_t t0 = get_highres_time_ns();
                                faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                                uint64_t t1 = get_highres_time_ns();
                                t_faiss = static_cast<double>(t1 - t0) / 1000.0;

                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;
                            }
                        } else {
                            if (order_mode == 0) {
                                evictor.flush_llc();
                                execute_plan5a_ivf_part_prune_avx2(idx_alg, q_vec, pred, k, res_alg, &cnt_alg, met_alg);
                                t_alg = met_alg.total_us;

                                evictor.flush_llc();
                                uint64_t t0 = get_highres_time_ns();
                                faiss_shim_search_flat(faiss_flat, 1, q_vec, k, dists_faiss, labels_faiss, params_flat);
                                uint64_t t1 = get_highres_time_ns();
                                t_faiss = static_cast<double>(t1 - t0) / 1000.0;
                            } else {
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

                        bool strict_ok = true;
                        uint32_t matches = 0;
                        for (uint32_t i = 0; i < k; ++i) {
                            if (res_alg[i].id != gt_topk[i].id) strict_ok = false;
                            for (uint32_t j = 0; j < k; ++j) {
                                if (res_alg[i].id == gt_topk[j].id) { matches++; break; }
                            }
                        }
                        if (strict_ok) strict_matches++;
                        sum_recall += (100.0 * matches) / k;
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
                    double pruning_ratio = (pred.eligible_count > 0) ? (1.0 - (avg_dists / pred.eligible_count)) * 100.0 : 100.0;
                    double pct_strict = (100.0 * strict_matches) / q_count;
                    double pct_recall = sum_recall / q_count;

                    csv << dist.name << "," << sel << "," << k << "," << cache_mode << ","
                        << speedup << "," << m_alg << "," << p50_alg << "," << p95_alg << "," << p99_alg << ","
                        << m_faiss << "," << p50_faiss << "," << p95_faiss << "," << p99_faiss << ","
                        << qps_alg << "," << qps_faiss << ","
                        << avg_dists << "," << pred.eligible_count << "," << pruning_ratio << ","
                        << pct_strict << "," << pct_recall << "\n";

                    std::cout << "[" << dist.name << " | Sel=" << std::setw(4) << sel << "% | " << cache_mode << " | Top-" << k << "] "
                              << "ALG=" << std::fixed << std::setprecision(1) << m_alg << " us (p95=" << p95_alg << ") | "
                              << "FAISS=" << m_faiss << " us | "
                              << "Speedup=" << std::setprecision(2) << speedup << "x | "
                              << "Recall=" << pct_recall << "%"
                              << std::endl;
                }
            }

            faiss_shim_free_params(params_flat);
            faiss_shim_free_selector(sel_handle);
        }

        faiss_shim_free_flat(faiss_flat);
    }

    csv.close();
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "EVALUACION EN DISTRIBUCIONES DESCONOCIDAS COMPLETADA EXITOSAMENTE" << std::endl;
    std::cout << "Resumen: " << out_csv << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
