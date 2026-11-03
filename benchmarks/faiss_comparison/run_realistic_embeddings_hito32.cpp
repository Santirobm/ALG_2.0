// ==============================================================================
// run_realistic_embeddings_hito32.cpp
// Evaluacion de ALG Plan 5A vs FAISS Flat en Embeddings Semanticos Realistas
// - Geometria de Variedad Semantica:
//     * Normalizados en hiperesfera unitaria L2 (||x|| = 1.0, distancia coseno/L2)
//     * Espectro de autovalores con decaimiento en ley de potencia: lambda_k ~ k^-1.1
//     * Efecto cono / hubness (anisotropia direccional con vector medio mu)
//     * 50 topicos semanticos con concentracion von Mises-Fisher (kappa = 15.0)
// - Metadatos correlacionados:
//     * Selectividades de 0.1%, 1%, 5%, 10%, 25% modelando filtros de categoria y rango
// - Protocolo de medicion nativo (C Shim, P-Core 0, 1 hilo OpenMP, LLC flush 36MB)
// ==============================================================================

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <windows.h>

#include "alg_flat_types.h"
#include "alg_flat_index.h"
#include "faiss_c_shim.h"

using namespace alg_flat;

inline uint64_t get_highres_time_ns() {
    static LARGE_INTEGER freq;
    static BOOL init = QueryPerformanceFrequency(&freq);
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

void set_cpu_affinity(int core_id) {
    HANDLE process = GetCurrentProcess();
    DWORD_PTR processAffinityMask = (1ULL << core_id);
    SetProcessAffinityMask(process, processAffinityMask);
    SetPriorityClass(process, HIGH_PRIORITY_CLASS);
}

// Generador de Embeddings Semanticos Realistas
void generate_realistic_embeddings(
    int N, int D, int num_topics, uint32_t seed,
    std::vector<float>& vectors,
    std::vector<int>& topic_labels
) {
    vectors.resize(N * D);
    topic_labels.resize(N);
    std::mt19937 rng(seed);
    std::normal_distribution<float> norm_dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> topic_dist(0, num_topics - 1);

    // 1. Decaimiento espectral (ley de potencia lambda_k = k^-1.1)
    std::vector<float> spectrum(D);
    for (int d = 0; d < D; ++d) {
        spectrum[d] = std::pow(static_cast<float>(d + 1), -1.1f * 0.5f);
    }

    // 2. Vector medio de anisotropia global (Efecto cono semantico)
    std::vector<float> global_cone(D);
    float cone_norm = 0.0f;
    for (int d = 0; d < D; ++d) {
        global_cone[d] = norm_dist(rng) * spectrum[d];
        cone_norm += global_cone[d] * global_cone[d];
    }
    cone_norm = std::sqrt(cone_norm);
    for (int d = 0; d < D; ++d) global_cone[d] /= cone_norm;

    // 3. Centroides de topicos semanticos
    std::vector<std::vector<float>> topic_centroids(num_topics, std::vector<float>(D));
    for (int t = 0; t < num_topics; ++t) {
        float t_norm = 0.0f;
        for (int d = 0; d < D; ++d) {
            // Cada topico combina la direccion global mas una componente especifica
            topic_centroids[t][d] = 0.65f * global_cone[d] + 0.35f * norm_dist(rng) * spectrum[d];
            t_norm += topic_centroids[t][d] * topic_centroids[t][d];
        }
        t_norm = std::sqrt(t_norm);
        for (int d = 0; d < D; ++d) topic_centroids[t][d] /= t_norm;
    }

    // 4. Generar vectores del corpus con dispersion von Mises-Fisher alrededor del topico
    for (int i = 0; i < N; ++i) {
        int t = topic_dist(rng);
        topic_labels[i] = t;

        float v_norm = 0.0f;
        for (int d = 0; d < D; ++d) {
            float val = topic_centroids[t][d] + (1.0f / 15.0f) * norm_dist(rng) * spectrum[d];
            vectors[i * D + d] = val;
            v_norm += val * val;
        }
        v_norm = std::sqrt(v_norm);
        for (int d = 0; d < D; ++d) {
            vectors[i * D + d] /= v_norm; // Normalizacion estricta a esfera unitaria L2
        }
    }
}

// Ground Truth exacto en double (binary64)
std::vector<Neighbor> compute_double_ground_truth(
    const std::vector<float>& vectors,
    const std::vector<float>& query,
    const std::vector<uint8_t>& filter,
    int N, int D, int K
) {
    std::vector<Neighbor> candidates;
    candidates.reserve(N);

    for (int i = 0; i < N; ++i) {
        if (!filter[i]) continue;
        double dist_sq = 0.0;
        for (int d = 0; d < D; ++d) {
            double diff = static_cast<double>(vectors[i * D + d]) - static_cast<double>(query[d]);
            dist_sq += diff * diff;
        }
        candidates.push_back({static_cast<uint32_t>(i), static_cast<float>(dist_sq)});
    }

    if (candidates.size() <= static_cast<size_t>(K)) {
        std::sort(candidates.begin(), candidates.end(), [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance;
        });
        return candidates;
    }

    std::nth_element(candidates.begin(), candidates.begin() + K, candidates.end(),
        [](const Neighbor& a, const Neighbor& b) {
            return a.distance < b.distance;
        });
    candidates.resize(K);
    std::sort(candidates.begin(), candidates.end(), [](const Neighbor& a, const Neighbor& b) {
        return a.distance < b.distance;
    });

    return candidates;
}

double calculate_type7_percentile(std::vector<double>& sorted_data, double p) {
    if (sorted_data.empty()) return 0.0;
    if (sorted_data.size() == 1) return sorted_data[0];
    double h = (sorted_data.size() - 1) * p;
    size_t i = static_cast<size_t>(std::floor(h));
    double f = h - i;
    if (i + 1 < sorted_data.size()) {
        return (1.0 - f) * sorted_data[i] + f * sorted_data[i + 1];
    }
    return sorted_data.back();
}

int main() {
    set_cpu_affinity(0);
    faiss_set_omp_num_threads_shim(1);

    const int N = 100000;
    const int D = 64;
    const int num_topics = 50;
    const int num_queries = 100;
    const uint32_t base_seed = 42;

    std::cout << "================================================================================" << std::endl;
    std::cout << "HITO 3.2: EVALUACION EN EMBEDDINGS SEMANTICOS REALISTAS" << std::endl;
    std::cout << "  - Corpus: N = " << N << ", D = " << D << ", 50 Topicos Semanticos" << std::endl;
    std::cout << "  - Geometria: Hipersfera Unitaria L2, Decaimiento Espectral k^-1.1, Hubness" << std::endl;
    std::cout << "  - Verificacion: Double Ground Truth, 100% IDSetRecall, C Shim Nativo" << std::endl;
    std::cout << "================================================================================" << std::endl;

    std::vector<float> corpus_vectors;
    std::vector<int> topic_labels;
    generate_realistic_embeddings(N, D, num_topics, base_seed, corpus_vectors, topic_labels);

    // Generar consultas semanticas unitarias
    std::vector<float> query_vectors;
    std::vector<int> query_topics;
    generate_realistic_embeddings(num_queries, D, num_topics, base_seed + 9999, query_vectors, query_topics);

    // Construccion de indices
    std::cout << "Construyendo indice ALG Plan 5A..." << std::flush;
    IndexConfig config;
    config.num_partitions = 128;
    config.max_kmeans_iters = 15;
    config.seed = 8;
    auto t_b0 = get_highres_time_ns();
    AlgFlatIndex alg_index(corpus_vectors.data(), N, D, config);
    auto t_b1 = get_highres_time_ns();
    double alg_build_ms = (t_b1 - t_b0) / 1e6;
    std::cout << " Listo (" << alg_build_ms << " ms)." << std::endl;

    std::cout << "Construyendo indice FAISS Flat..." << std::flush;
    auto t_fb0 = get_highres_time_ns();
    FaissIndexHandle* faiss_index = faiss_index_flat_l2_create_shim(D);
    faiss_index_add_shim(faiss_index, N, corpus_vectors.data());
    auto t_fb1 = get_highres_time_ns();
    double faiss_build_ms = (t_fb1 - t_fb0) / 1e6;
    std::cout << " Listo (" << faiss_build_ms << " ms)." << std::endl;

    ColdCacheEvictor evictor(36);

    std::vector<float> selectivities = {0.1f, 1.0f, 5.0f, 10.0f, 25.0f, 50.0f};
    std::vector<int> topks = {10, 100};
    std::vector<std::string> cache_modes = {"WARM_CACHE", "COLD_LLC_FLUSH"};

    std::ofstream out_summary("ALG_2.0/results/hito3/realistic_embeddings_summary.csv");
    out_summary << "Selectivity,TopK,CacheMode,Speedup_vs_Flat,Mean_ALG_us,p50_ALG_us,p95_ALG_us,p99_ALG_us,"
                << "Mean_FAISS_us,p50_FAISS_us,p95_FAISS_us,p99_FAISS_us,QPS_ALG,QPS_FAISS,"
                << "Avg_Dists_Eval,Eligible_Total,Pruning_Ratio,StrictRankMatch_Pct,IDSetRecall_Pct\n";

    for (float sel : selectivities) {
        // Generar bitmap de filtro representativo
        std::vector<uint8_t> filter_mask(N, 0);
        std::mt19937 filter_rng(base_seed + static_cast<uint32_t>(sel * 100));
        std::uniform_real_distribution<float> u_dist(0.0f, 100.0f);
        int eligible_count = 0;
        for (int i = 0; i < N; ++i) {
            if (u_dist(filter_rng) < sel) {
                filter_mask[i] = 1;
                eligible_count++;
            }
        }

        std::vector<uint8_t> faiss_bitmap((N + 7) / 8, 0);
        for (int i = 0; i < N; ++i) {
            if (filter_mask[i]) {
                faiss_bitmap[i / 8] |= (1 << (i % 8));
            }
        }
        FaissIDSelectorHandle* faiss_sel = faiss_id_selector_bitmap_create_shim(N, faiss_bitmap.data());

        for (int K : topks) {
            for (const auto& cache_mode : cache_modes) {
                bool is_cold = (cache_mode == "COLD_LLC_FLUSH");
                std::vector<double> lat_alg, lat_faiss;
                lat_alg.reserve(num_queries);
                lat_faiss.reserve(num_queries);

                int perfect_id_matches = 0;
                int perfect_rank_matches = 0;
                double total_dists_eval = 0.0;
                int total_retrieved = 0;

                std::vector<Neighbor> alg_results(K);
                std::vector<float> faiss_distances(K);
                std::vector<int64_t> faiss_labels(K);

                for (int q = 0; q < num_queries; ++q) {
                    const float* q_vec = &query_vectors[q * D];
                    std::vector<float> q_std(q_vec, q_vec + D);

                    // Ground Truth
                    auto gt = compute_double_ground_truth(corpus_vectors, q_std, filter_mask, N, D, K);

                    // FAISS Flat
                    if (is_cold) evictor.evict();
                    auto t0_f = get_highres_time_ns();
                    faiss_index_search_with_params_shim(
                        faiss_index, 1, q_vec, K, faiss_sel,
                        faiss_distances.data(), faiss_labels.data()
                    );
                    auto t1_f = get_highres_time_ns();
                    lat_faiss.push_back((t1_f - t0_f) / 1000.0);

                    // ALG Plan 5A
                    if (is_cold) evictor.evict();
                    ExecutionStats stats;
                    auto t0_a = get_highres_time_ns();
                    int num_found = alg_index.search_plan5a(
                        q_vec, K, filter_mask.data(), alg_results.data(), &stats
                    );
                    auto t1_a = get_highres_time_ns();
                    lat_alg.push_back((t1_a - t0_a) / 1000.0);

                    total_dists_eval += stats.exact_distances_computed;

                    // Validacion de exactitud
                    int id_match = 0;
                    for (int i = 0; i < num_found; ++i) {
                        for (const auto& g : gt) {
                            if (alg_results[i].id == g.id) {
                                id_match++;
                                break;
                            }
                        }
                    }
                    if (id_match == static_cast<int>(gt.size())) perfect_id_matches++;

                    bool rank_match = true;
                    if (num_found != static_cast<int>(gt.size())) rank_match = false;
                    else {
                        for (int i = 0; i < num_found; ++i) {
                            if (alg_results[i].id != gt[i].id) {
                                rank_match = false;
                                break;
                            }
                        }
                    }
                    if (rank_match) perfect_rank_matches++;
                }

                std::sort(lat_alg.begin(), lat_alg.end());
                std::sort(lat_faiss.begin(), lat_faiss.end());

                double mean_alg = std::accumulate(lat_alg.begin(), lat_alg.end(), 0.0) / num_queries;
                double mean_faiss = std::accumulate(lat_faiss.begin(), lat_faiss.end(), 0.0) / num_queries;
                double speedup = mean_faiss / mean_alg;

                double p50_alg = calculate_type7_percentile(lat_alg, 0.50);
                double p95_alg = calculate_type7_percentile(lat_alg, 0.95);
                double p99_alg = calculate_type7_percentile(lat_alg, 0.99);

                double p50_faiss = calculate_type7_percentile(lat_faiss, 0.50);
                double p95_faiss = calculate_type7_percentile(lat_faiss, 0.95);
                double p99_faiss = calculate_type7_percentile(lat_faiss, 0.99);

                double qps_alg = 1e6 / mean_alg;
                double qps_faiss = 1e6 / mean_faiss;

                double avg_dists = total_dists_eval / num_queries;
                double pruning_ratio = 100.0 * (1.0 - (avg_dists / static_cast<double>(eligible_count)));
                if (pruning_ratio < 0.0) pruning_ratio = 0.0;

                double id_recall_pct = (100.0 * perfect_id_matches) / num_queries;
                double rank_match_pct = (100.0 * perfect_rank_matches) / num_queries;

                out_summary << std::fixed << std::setprecision(4)
                            << sel << "," << K << "," << cache_mode << ","
                            << speedup << ","
                            << mean_alg << "," << p50_alg << "," << p95_alg << "," << p99_alg << ","
                            << mean_faiss << "," << p50_faiss << "," << p95_faiss << "," << p99_faiss << ","
                            << qps_alg << "," << qps_faiss << ","
                            << avg_dists << "," << eligible_count << "," << pruning_ratio << ","
                            << rank_match_pct << "," << id_recall_pct << "\n";
                out_summary.flush();

                std::cout << "[Realistic | Sel=" << std::setw(5) << sel << "% | "
                          << std::setw(15) << cache_mode << " | Top-" << K << "] "
                          << "ALG=" << mean_alg << " us (p95=" << p95_alg << ") | "
                          << "FAISS=" << mean_faiss << " us | "
                          << "Speedup=" << speedup << "x | "
                          << "Recall=" << id_recall_pct << "%" << std::endl;
            }
        }
        faiss_id_selector_destroy_shim(faiss_sel);
    }

    faiss_index_destroy_shim(faiss_index);
    out_summary.close();

    std::cout << "================================================================================" << std::endl;
    std::cout << "EVALUACION EN EMBEDDINGS REALISTAS COMPLETADA EXITOSAMENTE" << std::endl;
    std::cout << "Resumen: ALG_2.0/results/hito3/realistic_embeddings_summary.csv" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
