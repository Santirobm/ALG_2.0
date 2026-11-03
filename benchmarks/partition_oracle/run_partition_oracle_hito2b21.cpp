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

int main(int argc, char** argv) {
    std::cout << "================================================================================" << std::endl;
    std::cout << "ALG-Flat 2.0 (Hito 2B.2.1): Oracle de Despacho por Particion" << std::endl;
    std::cout << "  - Limite teorico de oportunidad: min(Cost_5A(p), Cost_5B(p)) por particion" << std::endl;
    std::cout << "  - Comparacion pareada con 5A global, 5B global y costo de preparacion" << std::endl;
    std::cout << "================================================================================" << std::endl;

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    std::string out_csv_path = "results/partition_oracle_opportunity_hito2b21.csv";
    std::ofstream csv(out_csv_path);
    if (!csv) {
        std::cerr << "Error al abrir " << out_csv_path << std::endl;
        return 1;
    }

    csv << "seed,scenario,sigma,selectivity,top_k,query_id,time_5a_global_us,time_5b_global_us,"
        << "time_oracle_part_us,oracle_advantage_vs_5a_pct,oracle_advantage_vs_5b_pct,"
        << "parts_total,parts_5a_won,parts_5b_won,parts_both_pruned\n";

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
    const uint32_t num_queries = 50U;

    Neighbor res_5a[512], res_5b[512];
    uint32_t count_5a = 0, count_5b = 0;
    QueryExecutionMetrics met_5a, met_5b;

    double sum_5a_us = 0.0, sum_5b_us = 0.0, sum_oracle_us = 0.0;
    uint64_t total_parts_5a_won = 0, total_parts_5b_won = 0, total_parts_both_pruned = 0;
    uint32_t total_queries_evaluated = 0;

    for (uint64_t seed : seeds) {
        for (const auto& sc : scenarios) {
            std::cout << "\n>>> Dataset: Seed=" << seed << " | Escenario=" << sc.name << " | sigma=" << sc.sigma << "..." << std::endl;

            BenchmarkDataset ds;
            if (!generate_dataset_deterministic(seed, N, num_queries, 64U, sc.sigma, sc.mix, ds)) {
                std::cerr << "Error generando dataset." << std::endl;
                return 1;
            }

            IVFPartitionedIndex idx;
            if (!build_ivf_partitioned_index(ds, 128U, 15U, idx)) {
                std::cerr << "Error construyendo indice IVF." << std::endl;
                return 1;
            }

            for (uint32_t s_idx : sel_indices) {
                float sel_pct = static_cast<float>(ds.header.selectivities_deci_percent[s_idx]) / 10.0f;
                uint32_t el_count = static_cast<uint32_t>(ds.eligible_per_sel[s_idx].size());

                QueryPredicate pred;
                pred.target_category = ds.header.selectivities_deci_percent[s_idx] * 10U;
                pred.eligible_ids = ds.eligible_per_sel[s_idx].data();
                pred.eligible_count = el_count;
                pred.sel_index = s_idx;

                for (uint32_t k : top_k_list) {
                    for (uint32_t q = 0; q < num_queries; ++q) {
                        const float* q_vec = ds.queries[q].values;

                        /* 1. Medir 5A global con prep y calc */
                        uint64_t t0 = get_perf_counter_ns();
                        execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                        uint64_t t1 = get_perf_counter_ns();
                        double t_5a_global_us = static_cast<double>(t1 - t0) / 1000.0;

                        /* 2. Medir 5B global con prep y calc */
                        t0 = get_perf_counter_ns();
                        execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                        t1 = get_perf_counter_ns();
                        double t_5b_global_us = static_cast<double>(t1 - t0) / 1000.0;

                        /* 3. Oracle por particion: ordenamiento de particiones */
                        const uint32_t n_parts = idx.num_partitions;
                        const uint32_t dim = idx.dim;

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

                        alignas(64) float q_rot[64];
                        float q_mean_norm = 0.0f;
                        {
                            alignas(64) float diff[64];
                            double norm_sq = 0.0;
                            for (uint32_t d = 0; d < dim; ++d) {
                                diff[d] = q_vec[d] - idx.global_mean[d];
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

                        TopKHeap oracle_heap;
                        oracle_heap.init(k);

                        double oracle_calc_ns = 0.0;
                        uint32_t parts_5a_won = 0, parts_5b_won = 0, parts_both_pruned = 0;
                        bool oracle_used_pca_any = false;

                        for (uint32_t pi = 0; pi < n_parts; ++pi) {
                            uint32_t pid = p_order[pi].part_id;
                            float lb_p = p_order[pi].lb_part_sq;

                            /* Evaluar particion bajo 5A */
                            TopKHeap heap_copy_5a = oracle_heap;
                            PartitionStepMetrics met_step_5a;
                            evaluate_single_partition_5a(idx, q_vec, pid, lb_p, pred, k, heap_copy_5a, met_step_5a);

                            /* Evaluar particion bajo 5B */
                            TopKHeap heap_copy_5b = oracle_heap;
                            PartitionStepMetrics met_step_5b;
                            evaluate_single_partition_5b(idx, q_vec, q_rot, q_mean_norm, pid,
                                                         p_order[pi].dc_low, p_order[pi].dc_high, lb_p,
                                                         pred, k, heap_copy_5b, met_step_5b);

                            if (met_step_5a.pruned_by_part && met_step_5b.pruned_by_part) {
                                parts_both_pruned++;
                                continue;
                            }

                            if (met_step_5b.time_ns < met_step_5a.time_ns) {
                                oracle_calc_ns += met_step_5b.time_ns;
                                oracle_heap = heap_copy_5b;
                                parts_5b_won++;
                                oracle_used_pca_any = true;
                            } else {
                                oracle_calc_ns += met_step_5a.time_ns;
                                oracle_heap = heap_copy_5a;
                                parts_5a_won++;
                            }
                        }

                        double prep_oracle_us = met_5a.prep_us + (oracle_used_pca_any ? 1.0 : 0.0);
                        double t_oracle_us = prep_oracle_us + (oracle_calc_ns / 1000.0);

                        double adv_vs_5a = (t_5a_global_us - t_oracle_us) / (t_5a_global_us > 0.0 ? t_5a_global_us : 1.0) * 100.0;
                        double adv_vs_5b = (t_5b_global_us - t_oracle_us) / (t_5b_global_us > 0.0 ? t_5b_global_us : 1.0) * 100.0;

                        sum_5a_us += t_5a_global_us;
                        sum_5b_us += t_5b_global_us;
                        sum_oracle_us += t_oracle_us;
                        total_parts_5a_won += parts_5a_won;
                        total_parts_5b_won += parts_5b_won;
                        total_parts_both_pruned += parts_both_pruned;
                        total_queries_evaluated++;

                        csv << seed << "," << sc.name << "," << sc.sigma << "," << sel_pct << "," << k << "," << q << ","
                            << t_5a_global_us << "," << t_5b_global_us << "," << t_oracle_us << ","
                            << adv_vs_5a << "," << adv_vs_5b << ","
                            << n_parts << "," << parts_5a_won << "," << parts_5b_won << "," << parts_both_pruned << "\n";
                    }
                }
            }
        }
    }

    csv.close();

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "EVALUACION DEL ORACLE POR PARTICION COMPLETADA (HITO 2B.2.1)" << std::endl;
    std::cout << "Total consultas evaluadas : " << total_queries_evaluated << std::endl;
    std::cout << "CSV de oportunidad       : " << out_csv_path << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Tiempo Acumulado Global (ms):" << std::endl;
    std::cout << "  Plan 5A Global       : " << std::fixed << std::setprecision(2) << (sum_5a_us / 1000.0) << " ms" << std::endl;
    std::cout << "  Plan 5B Global       : " << (sum_5b_us / 1000.0) << " ms" << std::endl;
    std::cout << "  Oracle por Particion : " << (sum_oracle_us / 1000.0) << " ms" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Ganancia Teórica Máxima del Oracle:" << std::endl;
    std::cout << "  vs Plan 5A Global    : " << ((sum_5a_us - sum_oracle_us) / sum_5a_us * 100.0) << "%" << std::endl;
    std::cout << "  vs Plan 5B Global    : " << ((sum_5b_us - sum_oracle_us) / sum_5b_us * 100.0) << "%" << std::endl;
    std::cout << "Distribución de Decisiones de Partición en el Oracle:" << std::endl;
    uint64_t total_p_decisions = total_parts_5a_won + total_parts_5b_won + total_parts_both_pruned;
    std::cout << "  Particiones donde 5A fue mejor : " << total_parts_5a_won << " (" << (100.0 * total_parts_5a_won / total_p_decisions) << "%)" << std::endl;
    std::cout << "  Particiones donde 5B fue mejor : " << total_parts_5b_won << " (" << (100.0 * total_parts_5b_won / total_p_decisions) << "%)" << std::endl;
    std::cout << "  Particiones podadas por ambas  : " << total_parts_both_pruned << " (" << (100.0 * total_parts_both_pruned / total_p_decisions) << "%)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
