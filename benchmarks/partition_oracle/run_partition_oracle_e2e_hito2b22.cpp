#include "../../include/alg_flat_types.h"
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

static inline uint64_t get_perf_counter_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

static bool compare_topk_exact(const Neighbor* a, const Neighbor* b, uint32_t k) {
    for (uint32_t i = 0; i < k; ++i) {
        if (a[i].id != b[i].id) {
            if (fabsf(a[i].dist_sq - b[i].dist_sq) > 1.0e-5f) return false;
        }
    }
    return true;
}

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "ALG-Flat 2.0 (Hito 2B.2.2): Auditoria Rigurosa End-to-End del Oracle" << std::endl;
    std::cout << "  - Medicion directa End-to-End continua sin cronometros en bucle de particion" << std::endl;
    std::cout << "  - Rondas pareadas e intercaladas (Round 1: 5A->Oracle, Round 2: Oracle->5A)" << std::endl;
    std::cout << "  - Separacion estricta WARM_CACHE y COLD_LLC_FLUSH (eviccion de 36 MB)" << std::endl;
    std::cout << "  - Certificacion estricta RankMatch@K (100% verificado contra 5A)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    ColdCacheEvictor evictor(36);

    std::string out_csv_path = "results/partition_oracle_end_to_end_audit_hito2b22.csv";
    std::ofstream csv(out_csv_path);
    if (!csv) {
        std::cerr << "Error abriendo " << out_csv_path << std::endl;
        return 1;
    }

    csv << "seed,scenario,sigma,selectivity,top_k,cache_mode,query_id,execution_order,"
        << "time_5a_us,time_oracle_e2e_us,oracle_advantage_vs_5a_pct,"
        << "oracle_used_pca_any,parts_5a_count,parts_5b_count,parts_pruned_count,exactness_status\n";

    struct ScenarioConfig {
        std::string name;
        float sigma;
        MixMode mix;
    };

    std::vector<uint64_t> seeds = {9ULL, 10ULL};
    std::vector<ScenarioConfig> scenarios = {
        {"Concentrated_s0015", 0.015f, MIX_R},
        {"Medium_s006",        0.060f, MIX_R},
        {"Diffuse_s015",       0.150f, MIX_R},
        {"Bimodal_5050",       0.060f, MIX_50_50},
        {"Adversarial_OOD",    0.150f, MIX_D}
    };

    std::vector<uint32_t> top_k_list = {10U, 100U};
    std::vector<uint32_t> sel_indices = {0U, 2U, 6U}; /* 10%, 15%, 25% */
    const uint32_t N = 100000U;
    const uint32_t num_queries_warm = 50U;
    const uint32_t num_queries_cold = 10U;

    Neighbor res_5a[512], res_oracle[512];
    uint32_t count_5a = 0, count_oracle = 0;
    QueryExecutionMetrics met_5a;

    double sum_5a_warm = 0.0, sum_oracle_warm = 0.0;
    double sum_5a_cold = 0.0, sum_oracle_cold = 0.0;
    uint32_t total_warm_queries = 0, total_cold_queries = 0;

    for (uint64_t seed : seeds) {
        for (const auto& sc : scenarios) {
            std::cout << "\n>>> Dataset: Seed=" << seed << " | Escenario=" << sc.name << " | sigma=" << sc.sigma << "..." << std::endl;

            BenchmarkDataset ds;
            if (!generate_dataset_deterministic(seed, N, num_queries_warm + num_queries_cold, 64U, sc.sigma, sc.mix, ds)) {
                std::cerr << "Error generando dataset." << std::endl;
                return 1;
            }

            IVFPartitionedIndex idx;
            if (!build_ivf_partitioned_index(ds, 128U, 15U, idx)) {
                std::cerr << "Error construyendo indice IVF." << std::endl;
                return 1;
            }

            const uint32_t n_parts = idx.num_partitions;
            const uint32_t dim = idx.dim;

            for (uint32_t s_idx : sel_indices) {
                float sel_pct = static_cast<float>(ds.header.selectivities_deci_percent[s_idx]) / 10.0f;
                uint32_t el_count = static_cast<uint32_t>(ds.eligible_per_sel[s_idx].size());

                QueryPredicate pred;
                pred.target_category = ds.header.selectivities_deci_percent[s_idx] * 10U;
                pred.eligible_ids = ds.eligible_per_sel[s_idx].data();
                pred.eligible_count = el_count;
                pred.sel_index = s_idx;

                for (uint32_t k : top_k_list) {
                    /* ==========================================================
                     * BLOQUE WARM_CACHE (Intercalado balanceado)
                     * ========================================================== */
                    for (uint32_t q = 0; q < num_queries_warm; ++q) {
                        const float* q_vec = ds.queries[q].values;

                        /* 1. Paso Contrafactual Independiente: Determinar decision del Oracle */
                        struct LocalOrder {
                            uint32_t part_id;
                            float dc;
                            float dc_low;
                            float dc_high;
                            float lb_part_sq;
                        };
                        LocalOrder p_order[512];
                        for (uint32_t p = 0; p < n_parts; ++p) {
                            float d2 = 0.0f;
                            const float* c = idx.centroids + static_cast<size_t>(p) * dim;
                            for (uint32_t d = 0; d < dim; ++d) {
                                float diff = q_vec[d] - c[d];
                                d2 += diff * diff;
                            }
                            const GeometricPartition& gp = idx.partitions[p];
                            PartitionDistanceBounds pdb = compute_partition_distance_bounds(d2, gp.min_radius, gp.max_radius);
                            p_order[p] = {p, pdb.dc, pdb.dc_low, pdb.dc_high, pdb.lb_part_sq};
                        }
                        std::sort(p_order, p_order + n_parts, [](const LocalOrder& a, const LocalOrder& b) {
                            return a.dc < b.dc;
                        });

                        alignas(64) float q_rot_setup[64];
                        float q_mean_norm_setup = 0.0f;
                        {
                            alignas(64) float diff[64];
                            double norm_sq = 0.0;
                            for (uint32_t d = 0; d < dim; ++d) {
                                diff[d] = q_vec[d] - idx.global_mean[d];
                                norm_sq += static_cast<double>(diff[d]) * diff[d];
                            }
                            q_mean_norm_setup = round_up_f32(sqrt_up_f64(norm_sq));
                            for (uint32_t r = 0; r < dim; ++r) {
                                const float* row = idx.ortho_basis + static_cast<size_t>(r) * dim;
                                float sum = 0.0f;
                                for (uint32_t c = 0; c < dim; ++c) sum += row[c] * diff[c];
                                q_rot_setup[r] = sum;
                            }
                        }

                        uint32_t sorted_pids[512];
                        float sorted_lb_part_sq[512];
                        float sorted_dc_low[512];
                        float sorted_dc_high[512];
                        uint8_t decision_5b_mask[512] = {0};

                        for (uint32_t pi = 0; pi < n_parts; ++pi) {
                            sorted_pids[pi] = p_order[pi].part_id;
                            sorted_lb_part_sq[pi] = p_order[pi].lb_part_sq;
                            sorted_dc_low[pi] = p_order[pi].dc_low;
                            sorted_dc_high[pi] = p_order[pi].dc_high;
                        }

                        TopKHeap setup_heap;
                        setup_heap.init(k);
                        uint32_t parts_5a_cnt = 0, parts_5b_cnt = 0, parts_pruned_cnt = 0;
                        bool oracle_needs_pca = false;

                        for (uint32_t pi = 0; pi < n_parts; ++pi) {
                            uint32_t pid = sorted_pids[pi];
                            float lb_p = sorted_lb_part_sq[pi];

                            TopKHeap h_5a = setup_heap;
                            PartitionStepMetrics m_5a;
                            evaluate_single_partition_5a(idx, q_vec, pid, lb_p, pred, k, h_5a, m_5a);

                            TopKHeap h_5b = setup_heap;
                            PartitionStepMetrics m_5b;
                            evaluate_single_partition_5b(idx, q_vec, q_rot_setup, q_mean_norm_setup, pid,
                                                         sorted_dc_low[pi], sorted_dc_high[pi], lb_p,
                                                         pred, k, h_5b, m_5b);

                            if (m_5a.pruned_by_part && m_5b.pruned_by_part) {
                                parts_pruned_cnt++;
                                continue;
                            }

                            if (m_5b.time_ns < m_5a.time_ns) {
                                decision_5b_mask[pi] = 1;
                                setup_heap = h_5b;
                                parts_5b_cnt++;
                                oracle_needs_pca = true;
                            } else {
                                decision_5b_mask[pi] = 0;
                                setup_heap = h_5a;
                                parts_5a_cnt++;
                            }
                        }

                        /* 2. Medicion Pareada e Intercalada End-to-End */
                        uint32_t order_mode = q % 2U;
                        std::string order_str;
                        double t_5a_us = 0.0, t_oracle_us = 0.0;

                        if (order_mode == 0) {
                            order_str = "5A->Oracle";
                            uint64_t t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            uint64_t t1 = get_perf_counter_ns();
                            t_5a_us = static_cast<double>(t1 - t0) / 1000.0;

                            execute_oracle_plan_end_to_end(idx, q_vec, sorted_pids, sorted_lb_part_sq,
                                                           sorted_dc_low, sorted_dc_high, decision_5b_mask,
                                                           oracle_needs_pca, pred, k, res_oracle, &count_oracle, t_oracle_us);
                        } else {
                            order_str = "Oracle->5A";
                            execute_oracle_plan_end_to_end(idx, q_vec, sorted_pids, sorted_lb_part_sq,
                                                           sorted_dc_low, sorted_dc_high, decision_5b_mask,
                                                           oracle_needs_pca, pred, k, res_oracle, &count_oracle, t_oracle_us);

                            uint64_t t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            uint64_t t1 = get_perf_counter_ns();
                            t_5a_us = static_cast<double>(t1 - t0) / 1000.0;
                        }

                        bool match = compare_topk_exact(res_5a, res_oracle, k);
                        std::string exactness_status = (match && count_5a == k) ? "VERIFIED_100_MATCH" : "MISMATCH";

                        double adv_vs_5a = (t_5a_us - t_oracle_us) / (t_5a_us > 0.0 ? t_5a_us : 1.0) * 100.0;

                        sum_5a_warm += t_5a_us;
                        sum_oracle_warm += t_oracle_us;
                        total_warm_queries++;

                        csv << seed << "," << sc.name << "," << sc.sigma << "," << sel_pct << "," << k << ","
                            << "WARM_CACHE," << q << "," << order_str << "," << t_5a_us << "," << t_oracle_us << ","
                            << adv_vs_5a << "," << (oracle_needs_pca ? 1 : 0) << ","
                            << parts_5a_cnt << "," << parts_5b_cnt << "," << parts_pruned_cnt << "," << exactness_status << "\n";
                    }

                    /* ==========================================================
                     * BLOQUE COLD_LLC_FLUSH (Eviccion de 36 MB)
                     * ========================================================== */
                    for (uint32_t q = num_queries_warm; q < num_queries_warm + num_queries_cold; ++q) {
                        const float* q_vec = ds.queries[q].values;

                        struct LocalOrder {
                            uint32_t part_id;
                            float dc;
                            float dc_low;
                            float dc_high;
                            float lb_part_sq;
                        };
                        LocalOrder p_order[512];
                        for (uint32_t p = 0; p < n_parts; ++p) {
                            float d2 = 0.0f;
                            const float* c = idx.centroids + static_cast<size_t>(p) * dim;
                            for (uint32_t d = 0; d < dim; ++d) {
                                float diff = q_vec[d] - c[d];
                                d2 += diff * diff;
                            }
                            const GeometricPartition& gp = idx.partitions[p];
                            PartitionDistanceBounds pdb = compute_partition_distance_bounds(d2, gp.min_radius, gp.max_radius);
                            p_order[p] = {p, pdb.dc, pdb.dc_low, pdb.dc_high, pdb.lb_part_sq};
                        }
                        std::sort(p_order, p_order + n_parts, [](const LocalOrder& a, const LocalOrder& b) {
                            return a.dc < b.dc;
                        });

                        alignas(64) float q_rot_setup[64];
                        float q_mean_norm_setup = 0.0f;
                        {
                            alignas(64) float diff[64];
                            double norm_sq = 0.0;
                            for (uint32_t d = 0; d < dim; ++d) {
                                diff[d] = q_vec[d] - idx.global_mean[d];
                                norm_sq += static_cast<double>(diff[d]) * diff[d];
                            }
                            q_mean_norm_setup = round_up_f32(sqrt_up_f64(norm_sq));
                            for (uint32_t r = 0; r < dim; ++r) {
                                const float* row = idx.ortho_basis + static_cast<size_t>(r) * dim;
                                float sum = 0.0f;
                                for (uint32_t c = 0; c < dim; ++c) sum += row[c] * diff[c];
                                q_rot_setup[r] = sum;
                            }
                        }

                        uint32_t sorted_pids[512];
                        float sorted_lb_part_sq[512];
                        float sorted_dc_low[512];
                        float sorted_dc_high[512];
                        uint8_t decision_5b_mask[512] = {0};

                        for (uint32_t pi = 0; pi < n_parts; ++pi) {
                            sorted_pids[pi] = p_order[pi].part_id;
                            sorted_lb_part_sq[pi] = p_order[pi].lb_part_sq;
                            sorted_dc_low[pi] = p_order[pi].dc_low;
                            sorted_dc_high[pi] = p_order[pi].dc_high;
                        }

                        TopKHeap setup_heap;
                        setup_heap.init(k);
                        uint32_t parts_5a_cnt = 0, parts_5b_cnt = 0, parts_pruned_cnt = 0;
                        bool oracle_needs_pca = false;

                        for (uint32_t pi = 0; pi < n_parts; ++pi) {
                            uint32_t pid = sorted_pids[pi];
                            float lb_p = sorted_lb_part_sq[pi];

                            TopKHeap h_5a = setup_heap;
                            PartitionStepMetrics m_5a;
                            evaluate_single_partition_5a(idx, q_vec, pid, lb_p, pred, k, h_5a, m_5a);

                            TopKHeap h_5b = setup_heap;
                            PartitionStepMetrics m_5b;
                            evaluate_single_partition_5b(idx, q_vec, q_rot_setup, q_mean_norm_setup, pid,
                                                         sorted_dc_low[pi], sorted_dc_high[pi], lb_p,
                                                         pred, k, h_5b, m_5b);

                            if (m_5a.pruned_by_part && m_5b.pruned_by_part) {
                                parts_pruned_cnt++;
                                continue;
                            }

                            if (m_5b.time_ns < m_5a.time_ns) {
                                decision_5b_mask[pi] = 1;
                                setup_heap = h_5b;
                                parts_5b_cnt++;
                                oracle_needs_pca = true;
                            } else {
                                decision_5b_mask[pi] = 0;
                                setup_heap = h_5a;
                                parts_5a_cnt++;
                            }
                        }

                        /* 2. Medicion Pareada e Intercalada con Flush Forzado de 36 MB */
                        uint32_t order_mode = q % 2U;
                        std::string order_str;
                        double t_5a_us = 0.0, t_oracle_us = 0.0;

                        if (order_mode == 0) {
                            order_str = "5A->Oracle";
                            evictor.flush_llc();
                            uint64_t t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            uint64_t t1 = get_perf_counter_ns();
                            t_5a_us = static_cast<double>(t1 - t0) / 1000.0;

                            evictor.flush_llc();
                            execute_oracle_plan_end_to_end(idx, q_vec, sorted_pids, sorted_lb_part_sq,
                                                           sorted_dc_low, sorted_dc_high, decision_5b_mask,
                                                           oracle_needs_pca, pred, k, res_oracle, &count_oracle, t_oracle_us);
                        } else {
                            order_str = "Oracle->5A";
                            evictor.flush_llc();
                            execute_oracle_plan_end_to_end(idx, q_vec, sorted_pids, sorted_lb_part_sq,
                                                           sorted_dc_low, sorted_dc_high, decision_5b_mask,
                                                           oracle_needs_pca, pred, k, res_oracle, &count_oracle, t_oracle_us);

                            evictor.flush_llc();
                            uint64_t t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            uint64_t t1 = get_perf_counter_ns();
                            t_5a_us = static_cast<double>(t1 - t0) / 1000.0;
                        }

                        bool match = compare_topk_exact(res_5a, res_oracle, k);
                        std::string exactness_status = (match && count_5a == k) ? "VERIFIED_100_MATCH" : "MISMATCH";

                        double adv_vs_5a = (t_5a_us - t_oracle_us) / (t_5a_us > 0.0 ? t_5a_us : 1.0) * 100.0;

                        sum_5a_cold += t_5a_us;
                        sum_oracle_cold += t_oracle_us;
                        total_cold_queries++;

                        csv << seed << "," << sc.name << "," << sc.sigma << "," << sel_pct << "," << k << ","
                            << "COLD_LLC_FLUSH," << q << "," << order_str << "," << t_5a_us << "," << t_oracle_us << ","
                            << adv_vs_5a << "," << (oracle_needs_pca ? 1 : 0) << ","
                            << parts_5a_cnt << "," << parts_5b_cnt << "," << parts_pruned_cnt << "," << exactness_status << "\n";
                    }
                }
            }
        }
    }

    csv.close();

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "AUDITORIA END-TO-END DEL ORACLE COMPLETADA (HITO 2B.2.2)" << std::endl;
    std::cout << "Total consultas WARM medidas : " << total_warm_queries << std::endl;
    std::cout << "Total consultas COLD medidas : " << total_cold_queries << std::endl;
    std::cout << "CSV canónico generado        : " << out_csv_path << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "WARM_CACHE Acumulado (ms):" << std::endl;
    std::cout << "  Plan 5A Global       : " << std::fixed << std::setprecision(2) << (sum_5a_warm / 1000.0) << " ms" << std::endl;
    std::cout << "  Oracle End-to-End    : " << (sum_oracle_warm / 1000.0) << " ms" << std::endl;
    std::cout << "  Ganancia Teórica E2E : " << ((sum_5a_warm - sum_oracle_warm) / sum_5a_warm * 100.0) << "%" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "COLD_LLC_FLUSH Acumulado (ms):" << std::endl;
    std::cout << "  Plan 5A Global       : " << (sum_5a_cold / 1000.0) << " ms" << std::endl;
    std::cout << "  Oracle End-to-End    : " << (sum_oracle_cold / 1000.0) << " ms" << std::endl;
    std::cout << "  Ganancia Teórica E2E : " << ((sum_5a_cold - sum_oracle_cold) / sum_5a_cold * 100.0) << "%" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
