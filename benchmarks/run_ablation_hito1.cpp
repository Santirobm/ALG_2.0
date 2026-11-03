#include "../include/alg_flat_types.h"
#include "../include/dataset_loader.h"
#include "../include/physical_layouts.h"
#include "../include/execution_plans.h"
#include "../include/ieee754_bounds.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace alg_flat;

#ifdef _WIN32
typedef LONG (WINAPI* RtlGetVersionPtr)(PRTL_OSVERSIONINFOW);
static std::string query_exact_windows_version() {
    HMODULE hMod = GetModuleHandleW(L"ntdll.dll");
    if (hMod) {
        RtlGetVersionPtr fxPtr = reinterpret_cast<RtlGetVersionPtr>(
            GetProcAddress(hMod, "RtlGetVersion")
        );
        if (fxPtr != nullptr) {
            RTL_OSVERSIONINFOW rovi = {0};
            rovi.dwOSVersionInfoSize = sizeof(rovi);
            if (fxPtr(&rovi) == 0) {
                std::ostringstream oss;
                const char* win_name = (rovi.dwBuildNumber >= 22000) ? "Windows 11" : "Windows 10";
                oss << win_name << " x86_64 (NT " << rovi.dwMajorVersion << "."
                    << rovi.dwMinorVersion << " Build " << rovi.dwBuildNumber << ")";
                return oss.str();
            }
        }
    }
    return "Windows x86_64";
}
#else
static std::string query_exact_windows_version() {
    return "POSIX x86_64";
}
#endif

/*
 * Nivel 1A: IDSetRecall@K (sin tolerancia de distancia en la frontera y con rechazo
 * explicito de IDs duplicados en la salida predicha).
 */
static double compute_id_set_recall_k(const Neighbor* pred, uint32_t pred_cnt,
                                      const Neighbor* gt, uint32_t gt_cnt, uint32_t k) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t n_p = (pred_cnt < k) ? pred_cnt : k;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_p; ++i) {
        bool is_dup = false;
        for (uint32_t prev = 0; prev < i; ++prev) {
            if (pred[prev].id == pred[i].id) {
                is_dup = true;
                break;
            }
        }
        if (is_dup) continue;
        for (uint32_t j = 0; j < target; ++j) {
            if (pred[i].id == gt[j].id) {
                hits++;
                break;
            }
        }
    }
    return static_cast<double>(hits) / static_cast<double>(target);
}

/*
 * Nivel 1B: RankMatch@K (identidad exacta posicion a posicion pred[r].id == gt[r].id
 * bajo el orden total determinista dist_sq_orig ASC, id ASC).
 */
static double compute_rank_match_k(const Neighbor* pred, uint32_t pred_cnt,
                                   const Neighbor* gt, uint32_t gt_cnt, uint32_t k) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t matches = 0;
    for (uint32_t r = 0; r < target; ++r) {
        if (r < pred_cnt && pred[r].id == gt[r].id) {
            matches++;
        }
    }
    return static_cast<double>(matches) / static_cast<double>(target);
}

int main() {
#ifdef _WIN32
    SetProcessAffinityMask(GetCurrentProcess(), 1ULL);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

    std::string os_version_str = query_exact_windows_version();

    std::cout << "========================================================================================\n";
    std::cout << "ALG-Flat 2.0 (Hito 1 + Certificacion Hito 2A.2): Benchmark Canonico de Ablacion\n";
    std::cout << "OS         : " << os_version_str << "\n";
    std::cout << "CPU Pinning: P-Core 0 (AffinityMask=0x1, HIGH_PRIORITY_CLASS)\n";
    std::cout << "Compilador : " << __VERSION__ << " | Flags: -O3 -march=native -mavx2 -mfma -std=c++17\n";
    std::cout << "Corpus     : N=100,000 | Dim=64 (float32 = 25.6 MB) | IDSetRecall + RankMatch (0 tol)\n";
    std::cout << "========================================================================================\n";

    /* Guardar metadatos canonicos de reproduccion */
    {
        std::ofstream meta("results/run_metadata.txt");
        meta << "Project: ALG-Flat 2.0 (Hito 1 + Hito 2A.2 Formal 5-Stage IEEE-754 Exactness Audit with Directed Rounding)\n";
        meta << "Compiler: " << __VERSION__ << "\n";
        meta << "Compile_Flags: -O3 -march=native -mavx2 -mfma -std=c++17\n";
        meta << "CPU: Intel Core i5-13500 (24 MB Intel Smart Cache L3)\n";
        meta << "OS: " << os_version_str << "\n";
        meta << "Affinity_Mask: 0x1 (P-Core 0)\n";
        meta << "Process_Priority: HIGH_PRIORITY_CLASS\n";
        meta << "Corpus_Vectors: 100000\n";
        meta << "Dimension: 64 (float32, 25.6 MB raw corpus)\n";
        meta << "Seed: 8\n";
        meta << "Warm_Queries: 500 (after 20 warmup queries per plan)\n";
        meta << "Cold_Queries: 40 (with 36 MB LLC eviction buffer + _mm_mfence before each query)\n";
        meta << "Plan_Execution_Order: 0_Ref_Hito9 -> 1_Scalar -> 2A_AVX2_1x -> 2B_AVX2_4x -> 3A_SoA -> 3B_AoSoA -> 4_IVF_Flat -> 5A_IVF_PartPrune -> 5B_IVF_FullExactPrune\n";
        meta << "IEEE754_Unit_Roundoff_u32: 5.9604644775390625e-8 (2^-24), u64: 1.1102230246251565e-16 (2^-53)\n";
        meta << "IEEE754_Directed_Rounding: round_up_f32 (+infty via nextafterf) for slacks/upper bounds, round_down_f32 (0 via nextafterf) for scale_safe/lower bounds\n";
        meta << "IEEE754_Gamma_Constants: gamma_17=" << GAMMA_17_F32 << ", gamma_33=" << GAMMA_33_F32
             << ", gamma_64=" << GAMMA_64_F32 << ", gamma_66=" << GAMMA_66_F32 << ", gamma_67=" << GAMMA_67_F32 << "\n";
        meta << "Recall_Metrics: IDSetRecall (duplicate-checked exact ID set membership, 0 distance tolerance) + RankMatch (position-by-position identity under dist_sq ASC, id ASC)\n";
    }

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
        << "VectorsPrunedPct,PartitionsVisitedPct,EffectiveBW_GBs,IDSetRecall10,IDSetRecallK,RankMatch10,RankMatchK\n";

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

        printf("Tiempos de Construccion (ms): AoS=%.2f | SoA=%.2f | AoSoA=%.2f | IVF(P=128)=%.2f (KMeans=%.2f + Layout=%.2f) | OrthoDefect=%.3e\n",
               aos_idx.build_time_ms, soa_idx.build_time_ms, aosoa_idx.build_time_ms,
               ivf_idx.total_build_time_ms, ivf_idx.kmeans_time_ms, ivf_idx.layout_time_ms, ivf_idx.ortho_defect);

        struct EvalCondition {
            uint32_t sel_idx;
            uint32_t top_k;
            bool cold_cache;
            uint32_t num_q;
        };

        std::vector<EvalCondition> conditions = {
            {2, 100, false, 500},  /* 15% Sel, K=100, WARM */
            {2, 100, true,  40},   /* 15% Sel, K=100, COLD (36 MB LLC Flush) */
            {2, 10,  false, 500},  /* 15% Sel, K=10,  WARM */
            {0, 100, false, 500},  /* 10% Sel, K=100, WARM */
            {6, 100, false, 500}   /* 25% Sel, K=100, WARM */
        };

        for (const auto& cond : conditions) {
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
            printf("%-30s | %8s | %8s | %8s | %8s | %8s | %7s | %7s | %7s | %8s | %8s\n",
                   "Plan de Ejecucion", "Prep(us)", "Calc(us)", "p50(us)", "p95(us)", "Mean(us)",
                   "Cyc/Vec", "Pruned%", "BW(GBs)", "IDSetR%", "RankM%");
            printf("-----------------------------------------------------------------------------------------------------------------------------------------\n");

            /* Precalcular Ground Truth Exacto en espacio original con Plan 0 */
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
                if (!cond.cold_cache) {
                    Neighbor w_out[512];
                    uint32_t w_cnt = 0;
                    QueryExecutionMetrics w_m{};
                    for (uint32_t w = 0; w < 20; ++w) {
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
                double sum_bytes = 0.0, sum_r10 = 0.0, sum_rk = 0.0, sum_rm10 = 0.0, sum_rmk = 0.0;

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

                    sum_r10 += compute_id_set_recall_k(q_out, q_cnt, gt_all[q].data(), gt_cnts[q], 10);
                    sum_rk  += compute_id_set_recall_k(q_out, q_cnt, gt_all[q].data(), gt_cnts[q], k);
                    sum_rm10 += compute_rank_match_k(q_out, q_cnt, gt_all[q].data(), gt_cnts[q], 10);
                    sum_rmk  += compute_rank_match_k(q_out, q_cnt, gt_all[q].data(), gt_cnts[q], k);
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
                if (pruned_pct < 0.0) pruned_pct = 0.0;
                double mean_parts_pct = sum_parts_pct / nq;
                double bw_gbs = (sum_bytes / nq) / (mean_tot * 1e-6) / 1e9;
                double r10 = (sum_r10 / nq) * 100.0;
                double rk  = (sum_rk / nq) * 100.0;
                double rm10 = (sum_rm10 / nq) * 100.0;
                double rmk  = (sum_rmk / nq) * 100.0;

                printf("%-30s | %8.1f | %8.1f | %8.1f | %8.1f | %8.1f | %7.1f | %6.1f%% | %7.2f | %7.3f%% | %7.3f%%\n",
                       plan_names[pid], mean_prep, mean_calc, p50, p95, mean_tot,
                       cyc_per_vec, pruned_pct, bw_gbs, rk, rmk);

                csv << sc.name << "," << cmode << "," << k << "," << sel_pct << ","
                    << plan_names[pid] << "," << build_times[pid] << ","
                    << mean_prep << "," << mean_calc << "," << mean_tot << ","
                    << p50 << "," << p95 << "," << p99 << ","
                    << mean_cyc << "," << cyc_per_vec << "," << mean_dists << ","
                    << pruned_pct << "," << mean_parts_pct << "," << bw_gbs << ","
                    << r10 << "," << rk << "," << rm10 << "," << rmk << "\n";
            }
        }
    }

    csv.close();
    std::cout << "\nBenchmark canonico completado. Resultados en results/ablation_hito1_summary.csv y results/run_metadata.txt\n";
    return 0;
}
