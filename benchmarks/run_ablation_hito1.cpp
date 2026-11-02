#include "../include/alg_flat_types.h"
#include "../include/dataset_loader.h"
#include "../include/physical_layouts.h"
#include "../include/execution_plans.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cmath>

using namespace alg_flat;

static double compute_recall_k(const Neighbor* pred, uint32_t pred_cnt,
                               const Neighbor* gt, uint32_t gt_cnt, uint32_t k) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t n_p = (pred_cnt < k) ? pred_cnt : k;
    float kth_gt_dist = gt[target - 1].dist_sq * 1.00001f;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_p; ++i) {
        bool found = false;
        for (uint32_t j = 0; j < target; ++j) {
            if (pred[i].id == gt[j].id) {
                found = true;
                break;
            }
        }
        if (!found && pred[i].dist_sq <= kth_gt_dist) {
            found = true;
        }
        if (found) hits++;
    }
    return static_cast<double>(hits) / static_cast<double>(target);
}

struct PlanSummary {
    std::string scenario;
    std::string cache_mode;
    uint32_t top_k;
    double selectivity_pct;
    std::string plan_name;
    double build_ms;
    double prep_mean_us;
    double calc_mean_us;
    double total_mean_us;
    double total_p50_us;
    double total_p95_us;
    double total_p99_us;
    double mean_cycles;
    double cycles_per_el_vec;
    double dists_eval_mean;
    double pruned_pct;
    double parts_visited_pct;
    double bandwidth_gbs;
    double recall_at_10;
    double recall_at_k;
};

int main() {
#ifdef _WIN32
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

    std::cout << "========================================================================================\n";
    std::cout << "ALG-Flat 2.0 (Hito 1): Ablacion Incremental de Layouts Fisicos, SIMD y Poda Exacta\n";
    std::cout << "CPU Pinning: P-Core 0 | Vectores: 100,000 | Dim: 64 (float32 = 25.6 MB)\n";
    std::cout << "========================================================================================\n";

    struct ScenarioConfig {
        std::string name;
        uint64_t seed;
        uint32_t basins;
        float sigma;
        MixMode mix;
    };

    std::vector<ScenarioConfig> scenarios = {
        {"R_k64_s003_Tight",  8, 64, 0.03f, MIX_R},
        {"R_k64_s009_Base",   8, 64, 0.09f, MIX_R},
        {"M5050_k64_s009",    8, 64, 0.09f, MIX_50_50},
        {"D_k64_s009_Adv",    8, 64, 0.09f, MIX_D}
    };

    ColdCacheEvictor evictor(36); /* 36 MB > 24 MB L3 Smart Cache */
    std::ofstream csv("results/ablation_hito1_summary.csv");
    csv << "Scenario,CacheMode,TopK,SelectivityPct,Plan,BuildTime_ms,PrepMean_us,CalcMean_us,"
        << "TotalMean_us,p50_us,p95_us,p99_us,MeanCycles,CyclesPerEligibleVec,DistsEvaluated,"
        << "VectorsPrunedPct,PartitionsVisitedPct,EffectiveBW_GBs,Recall10,RecallK\n";

    for (const auto& sc : scenarios) {
        std::cout << "\n########################################################################################\n";
        std::cout << "ESCENARIO: " << sc.name << " (Seed=" << sc.seed << ", K=" << sc.basins
                  << ", sigma=" << sc.sigma << ")\n";
        std::cout << "########################################################################################\n";

        BenchmarkDataset ds;
        generate_dataset_deterministic(sc.seed, 100000, 1000, sc.basins, sc.sigma, sc.mix, ds);

        FlatAoSIndex aos_idx;
        FlatSoAIndex soa_idx;
        FlatAoSoAIndex aosoa_idx;
        IVFPartitionedIndex ivf_idx;

        build_flat_aos_index(ds, aos_idx);
        build_flat_soa_index(ds, soa_idx);
        build_flat_aosoa_index(ds, aosoa_idx);
        build_ivf_partitioned_index(ds, 128, 15, ivf_idx);

        printf("Tiempos de Construccion (ms): AoS=%.2f | SoA=%.2f | AoSoA=%.2f | IVF(P=128)=%.2f (KMeans=%.2f + Layout=%.2f)\n",
               aos_idx.build_time_ms, soa_idx.build_time_ms, aosoa_idx.build_time_ms,
               ivf_idx.total_build_time_ms, ivf_idx.kmeans_time_ms, ivf_idx.layout_time_ms);

        /* Evaluar en selectividad 15.0% (seli=2) y 25.0% (seli=6), para TopK=10 y TopK=100, en WARM y COLD */
        struct EvalCondition {
            uint32_t sel_idx;
            uint32_t top_k;
            bool cold_cache;
            uint32_t num_q;
        };

        std::vector<EvalCondition> conditions = {
            {2, 100, false, 500},  /* 15% Sel, K=100, WARM (Comparacion directa con 184 us de Hito 9) */
            {2, 100, true,  40},   /* 15% Sel, K=100, COLD (Con flush completo de 36 MB LLC) */
            {2, 10,  false, 500},  /* 15% Sel, K=10,  WARM */
            {0, 100, false, 500},  /* 10% Sel, K=100, WARM */
            {6, 100, false, 500}   /* 25% Sel, K=100, WARM */
        };

        for (const auto& cond : conditions) {
            /* En escenarios secundarios ejecutamos la condicion principal 15% Warm K=100 y K=10 */
            if (sc.name != "R_k64_s009_Base" && (cond.sel_idx != 2)) continue;

            uint32_t sel_deci = ds.header.selectivities_deci_percent[cond.sel_idx];
            double sel_pct = static_cast<double>(sel_deci) / 10.0;
            uint32_t target_cat = sel_deci * 10U;
            const auto& el_ids = ds.eligible_per_sel[cond.sel_idx];
            QueryPredicate pred{target_cat, el_ids.data(), static_cast<uint32_t>(el_ids.size()), cond.sel_idx};
            uint32_t nq = cond.num_q;
            uint32_t k = cond.top_k;
            std::string cmode = cond.cold_cache ? "COLD_LLC_FLUSH" : "WARM_CACHE";

            printf("\n--- [%s | %s | Sel=%.1f%% (%u elegibles) | Top-%u | Queries=%u] ---\n",
                   sc.name.c_str(), cmode.c_str(), sel_pct, pred.eligible_count, k, nq);
            printf("%-30s | %8s | %8s | %8s | %8s | %8s | %7s | %7s | %7s | %6s\n",
                   "Plan de Ejecucion", "Prep(us)", "Calc(us)", "p50(us)", "p95(us)", "Mean(us)",
                   "Cyc/Vec", "Pruned%", "BW(GBs)", "R@K(%)");
            printf("----------------------------------------------------------------------------------------------------------------------------\n");

            /* Precalcular Ground Truth Exacto con Plan 0 */
            std::vector<std::vector<Neighbor>> gt_all(nq, std::vector<Neighbor>(k));
            std::vector<uint32_t> gt_cnts(nq, 0);
            for (uint32_t q = 0; q < nq; ++q) {
                QueryExecutionMetrics m_gt{};
                execute_plan0_ref_hito9(aos_idx, ds.queries[q].values, pred, k, gt_all[q].data(), &gt_cnts[q], m_gt);
            }

            const int num_plans = 9;
            const char* plan_names[num_plans] = {
                "0_Ref_Hito9_ExactScan",
                "1_Flat_Scalar_AoS",
                "2A_Flat_AVX2_AoS_1x",
                "2B_Flat_AVX2_AoS_4x",
                "3A_Flat_AVX2_SoA",
                "3B_Flat_AVX2_AoSoA",
                "4_IVF_Flat_AVX2",
                "5A_IVF_PartPrune_AVX2",
                "5B_IVF_FullExactPrune_AVX2"
            };

            double build_times[num_plans] = {
                aos_idx.build_time_ms,
                aos_idx.build_time_ms,
                aos_idx.build_time_ms,
                aos_idx.build_time_ms,
                soa_idx.build_time_ms,
                aosoa_idx.build_time_ms,
                ivf_idx.total_build_time_ms,
                ivf_idx.total_build_time_ms,
                ivf_idx.total_build_time_ms
            };

            for (int pid = 0; pid < num_plans; ++pid) {
                /* Warmup (si es modo WARM) */
                if (!cond.cold_cache) {
                    Neighbor w_out[512];
                    uint32_t w_cnt = 0;
                    QueryExecutionMetrics w_m{};
                    for (uint32_t w = 0; w < 10; ++w) {
                        const float* qv = ds.queries[w].values;
                        if (pid == 0) execute_plan0_ref_hito9(aos_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 1) execute_plan1_flat_scalar_aos(aos_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 2) execute_plan2a_flat_avx2_aos_1x(aos_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 3) execute_plan2b_flat_avx2_aos_4x(aos_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 4) execute_plan3a_flat_avx2_soa(soa_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 5) execute_plan3b_flat_avx2_aosoa(aosoa_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 6) execute_plan4_ivf_flat_avx2(ivf_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 7) execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, w_out, &w_cnt, w_m);
                        else if (pid == 8) execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, w_out, &w_cnt, w_m);
                    }
                }

                std::vector<double> tot_lats(nq);
                double sum_prep = 0.0, sum_calc = 0.0, sum_tot = 0.0;
                double sum_cycles = 0.0, sum_dists = 0.0, sum_parts_pct = 0.0;
                double sum_bytes = 0.0, sum_r10 = 0.0, sum_rk = 0.0;

                Neighbor q_out[512];
                uint32_t q_cnt = 0;

                for (uint32_t q = 0; q < nq; ++q) {
                    if (cond.cold_cache) {
                        evictor.flush_llc();
                    }

                    const float* qv = ds.queries[q].values;
                    QueryExecutionMetrics m{};
                    if (pid == 0) execute_plan0_ref_hito9(aos_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 1) execute_plan1_flat_scalar_aos(aos_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 2) execute_plan2a_flat_avx2_aos_1x(aos_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 3) execute_plan2b_flat_avx2_aos_4x(aos_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 4) execute_plan3a_flat_avx2_soa(soa_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 5) execute_plan3b_flat_avx2_aosoa(aosoa_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 6) execute_plan4_ivf_flat_avx2(ivf_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 7) execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, q_out, &q_cnt, m);
                    else if (pid == 8) execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, q_out, &q_cnt, m);

                    tot_lats[q] = m.total_us;
                    sum_prep += m.prep_us;
                    sum_calc += m.calc_us;
                    sum_tot += m.total_us;
                    sum_cycles += static_cast<double>(m.cpu_cycles);
                    sum_dists += static_cast<double>(m.dists_evaluated);
                    sum_parts_pct += 100.0 * static_cast<double>(m.partitions_visited) / static_cast<double>(m.partitions_total);
                    sum_bytes += static_cast<double>(m.bytes_read);

                    sum_r10 += compute_recall_k(q_out, q_cnt, gt_all[q].data(), gt_cnts[q], 10);
                    sum_rk += compute_recall_k(q_out, q_cnt, gt_all[q].data(), gt_cnts[q], k);
                }

                std::sort(tot_lats.begin(), tot_lats.end());
                double p50 = tot_lats[static_cast<size_t>(nq * 0.50)];
                double p95 = tot_lats[static_cast<size_t>(nq * 0.95)];
                double p99 = tot_lats[static_cast<size_t>(nq * 0.99)];

                double mean_prep = sum_prep / nq;
                double mean_calc = sum_calc / nq;
                double mean_tot = sum_tot / nq;
                double mean_cyc = sum_cycles / nq;
                double cyc_per_vec = mean_cyc / static_cast<double>(pred.eligible_count);
                double mean_dists = sum_dists / nq;
                double pruned_pct = 100.0 * (1.0 - (mean_dists / static_cast<double>(pred.eligible_count)));
                if (pruned_pct < 0.0) pruned_pct = 0.0; /* Para SoA/AoSoA global que evaluan carriles extra */
                double mean_parts_pct = sum_parts_pct / nq;
                double bw_gbs = (sum_bytes / nq) / (mean_tot * 1e-6) / 1e9;
                double r10 = (sum_r10 / nq) * 100.0;
                double rk = (sum_rk / nq) * 100.0;

                printf("%-30s | %8.1f | %8.1f | %8.1f | %8.1f | %8.1f | %7.1f | %6.1f%% | %7.2f | %6.2f\n",
                       plan_names[pid], mean_prep, mean_calc, p50, p95, mean_tot,
                       cyc_per_vec, pruned_pct, bw_gbs, rk);

                csv << sc.name << "," << cmode << "," << k << "," << sel_pct << ","
                    << plan_names[pid] << "," << build_times[pid] << ","
                    << mean_prep << "," << mean_calc << "," << mean_tot << ","
                    << p50 << "," << p95 << "," << p99 << ","
                    << mean_cyc << "," << cyc_per_vec << "," << mean_dists << ","
                    << pruned_pct << "," << mean_parts_pct << "," << bw_gbs << ","
                    << r10 << "," << rk << "\n";
            }
        }
    }

    csv.close();
    std::cout << "\nBenchmark completado con exito. Resultados exportados a results/ablation_hito1_summary.csv\n";
    return 0;
}
