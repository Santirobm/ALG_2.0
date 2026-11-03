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

/*
 * ============================================================================
 * ESTRUCTURAS DE REGISTRO PAREADO POR CONSULTA (HITO 2B.1)
 * ============================================================================
 */
struct QueryRegretRecord {
    std::string dataset_id;
    uint64_t seed;
    std::string scenario;
    float sigma;
    float selectivity_pct;
    uint32_t eligible_count;
    uint32_t top_k;
    std::string cache_mode;
    uint32_t query_id;
    std::string execution_order;
    double time_5a_ns;
    double time_5b_ns;
    double time_6_ns;
    std::string best_fixed_plan;
    std::string selected_plan;
    double regret_ns;
    double relative_regret;
    uint8_t plan6_worse_than_both;
    uint8_t plan6_better_than_both;
    uint8_t plan6_between_both;
    uint8_t gate3_aborted;
    float gate3_pilot_work;
    float gate3_max_q_ratio;
    std::string gate3_reason;
    std::string exactness_status;
};

/*
 * Cronometro de alta resolucion pareado con serializacion de pipeline
 */
static inline uint64_t get_perf_counter_ns() {
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    _mm_lfence();
    QueryPerformanceCounter(&counter);
    _mm_lfence();
    return static_cast<uint64_t>((counter.QuadPart * 1000000000ULL) / freq.QuadPart);
}

/*
 * Verificacion estricta de RankMatch e IDSetRecall entre dos conjuntos de Top-K
 */
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
    std::cout << "ALG-Flat 2.0 (Hito 2B.1): Harness de Evaluacion Pareada de Regret de Plan 6" << std::endl;
    std::cout << "  - Protocolo balanceado de rotacion de orden: Round 1 (5A->5B->6), Round 2 (5B->6->5A), Round 3 (6->5A->5B)" << std::endl;
    std::cout << "  - Aislamiento WARM_CACHE y COLD_LLC_FLUSH (eviccion de 36 MB)" << std::endl;
    std::cout << "  - Verificacion estricta RankMatch@K pareada (5A == 5B == 6)" << std::endl;
    std::cout << "================================================================================" << std::endl;

    /* Pinning P-Core 0 y prioridad alta */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), 0x1);

    ColdCacheEvictor evictor(36);

    std::string out_csv_path = "results/regret_paired_eval_hito2b1.csv";
    std::ofstream csv(out_csv_path);
    if (!csv) {
        std::cerr << "Error: no se pudo abrir " << out_csv_path << " para escritura." << std::endl;
        return 1;
    }

    /* Encabezado canónico CSV exigido por la auditoría de Hito 2B.1 */
    csv << "dataset_id,seed,scenario,sigma,selectivity,eligible_count,top_k,cache_mode,query_id,"
        << "execution_order,time_5a_ns,time_5b_ns,time_6_ns,best_fixed_plan,selected_plan,"
        << "regret_ns,relative_regret,plan6_worse_than_both,plan6_better_than_both,plan6_between_both,"
        << "gate3_aborted,gate3_pilot_work,gate3_max_q_ratio,gate3_reason,exactness_status\n";

    std::vector<QueryRegretRecord> all_records;
    all_records.reserve(50000);

    /* Configuracion de la matriz experimental Hito 2B.1 */
    struct ScenarioConfig {
        std::string name;
        float sigma;
        MixMode mix;
    };

    /*
     * Batería fuera de muestra: Semillas 9 y 10
     * Dispersiones sigma: 0.015 (ultra-concentrado), 0.06 (moderado-concentrado), 0.15 (muy difuso)
     * Escenario adversarial puro
     */
    std::vector<uint64_t> seeds = {9ULL, 10ULL};
    std::vector<ScenarioConfig> scenarios = {
        {"Concentrated_s0015", 0.015f, MIX_R},
        {"Medium_s006",        0.060f, MIX_R},
        {"Diffuse_s015",       0.150f, MIX_R},
        {"Bimodal_5050",       0.060f, MIX_50_50},
        {"Adversarial_OOD",    0.150f, MIX_D}
    };

    std::vector<uint32_t> top_k_list = {10U, 100U};
    std::vector<uint32_t> sel_indices = {0U, 2U, 4U, 6U}; /* Corresponde a 10.0%, 15.0%, 20.0%, 25.0% */
    const uint32_t N = 100000U;
    const uint32_t num_queries_warm = 80U;
    const uint32_t num_queries_cold = 10U;

    Neighbor res_5a[512], res_5b[512], res_6[512];
    uint32_t count_5a = 0, count_5b = 0, count_6 = 0;
    QueryExecutionMetrics met_5a, met_5b, met_6;

    for (uint64_t seed : seeds) {
        for (const auto& sc : scenarios) {
            std::cout << "\n>>> Generando dataset independiente: Seed=" << seed 
                      << " | Escenario=" << sc.name << " | sigma=" << sc.sigma << "..." << std::endl;

            BenchmarkDataset ds;
            if (!generate_dataset_deterministic(seed, N, num_queries_warm + num_queries_cold, 64U, sc.sigma, sc.mix, ds)) {
                std::cerr << "Error al generar dataset determinista." << std::endl;
                return 1;
            }

            std::cout << ">>> Construyendo indice IVF particionado (P=128, K-Means)..." << std::endl;
            IVFPartitionedIndex idx;
            if (!build_ivf_partitioned_index(ds, 128U, 15U, idx)) {
                std::cerr << "Error al construir IVFPartitionedIndex." << std::endl;
                return 1;
            }

            std::string ds_id = "ds_s" + std::to_string(seed) + "_" + sc.name;

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
                     * BLOQUE WARM_CACHE (Orden balanceado)
                     * ========================================================== */
                    for (uint32_t q = 0; q < num_queries_warm; ++q) {
                        const float* q_vec = ds.queries[q].values;

                        /* Rotacion balanceada de orden de ejecucion */
                        uint32_t order_mode = q % 3U;
                        std::string order_str;
                        uint64_t t0, t1;
                        double ns_5a = 0.0, ns_5b = 0.0, ns_6 = 0.0;

                        if (order_mode == 0) {
                            order_str = "5A->5B->6";
                            t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            t1 = get_perf_counter_ns();
                            ns_5a = static_cast<double>(t1 - t0);

                            t0 = get_perf_counter_ns();
                            execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                            t1 = get_perf_counter_ns();
                            ns_5b = static_cast<double>(t1 - t0);

                            t0 = get_perf_counter_ns();
                            execute_plan6_ivf_adaptive_cost_model_avx2(idx, q_vec, pred, k, res_6, &count_6, met_6);
                            t1 = get_perf_counter_ns();
                            ns_6 = static_cast<double>(t1 - t0);
                        } else if (order_mode == 1) {
                            order_str = "5B->6->5A";
                            t0 = get_perf_counter_ns();
                            execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                            t1 = get_perf_counter_ns();
                            ns_5b = static_cast<double>(t1 - t0);

                            t0 = get_perf_counter_ns();
                            execute_plan6_ivf_adaptive_cost_model_avx2(idx, q_vec, pred, k, res_6, &count_6, met_6);
                            t1 = get_perf_counter_ns();
                            ns_6 = static_cast<double>(t1 - t0);

                            t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            t1 = get_perf_counter_ns();
                            ns_5a = static_cast<double>(t1 - t0);
                        } else {
                            order_str = "6->5A->5B";
                            t0 = get_perf_counter_ns();
                            execute_plan6_ivf_adaptive_cost_model_avx2(idx, q_vec, pred, k, res_6, &count_6, met_6);
                            t1 = get_perf_counter_ns();
                            ns_6 = static_cast<double>(t1 - t0);

                            t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            t1 = get_perf_counter_ns();
                            ns_5a = static_cast<double>(t1 - t0);

                            t0 = get_perf_counter_ns();
                            execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                            t1 = get_perf_counter_ns();
                            ns_5b = static_cast<double>(t1 - t0);
                        }

                        /* Verificacion de exactitud estricta */
                        bool match_5a_5b = compare_topk_exact(res_5a, res_5b, k);
                        bool match_5a_6 = compare_topk_exact(res_5a, res_6, k);
                        std::string exactness_status = (match_5a_5b && match_5a_6 && count_5a == k) ? "VERIFIED_100_MATCH" : "MISMATCH_DETECTED";

                        double best_fixed_ns = std::min(ns_5a, ns_5b);
                        std::string best_fixed_name = (ns_5a <= ns_5b) ? "5A" : "5B";
                        std::string sel_plan_name;
                        if (met_6.plan6_selected_cascade == 0) sel_plan_name = "5A_REMAIN";
                        else if (met_6.plan6_gate3_aborted) sel_plan_name = "5B_ABORTED_TO_5A";
                        else sel_plan_name = "5B_CASCADE";

                        double regret_ns = ns_6 - best_fixed_ns;
                        double rel_regret = regret_ns / (best_fixed_ns > 0.0 ? best_fixed_ns : 1.0);

                        uint8_t worse_both = (ns_6 > std::max(ns_5a, ns_5b) + 500.0) ? 1 : 0;
                        uint8_t better_both = (ns_6 < std::min(ns_5a, ns_5b) - 500.0) ? 1 : 0;
                        uint8_t between_both = (!worse_both && !better_both) ? 1 : 0;

                        std::string g3_reason = "NONE";
                        if (met_6.plan6_selected_cascade == 0) g3_reason = "GATE1_OR_GATE2_FILTER";
                        else if (met_6.plan6_gate3_aborted) g3_reason = "PILOT_EXCEEDED_MAX_RATIO";
                        else g3_reason = "CASCADE_COMPLETED";

                        QueryRegretRecord rec;
                        rec.dataset_id = ds_id;
                        rec.seed = seed;
                        rec.scenario = sc.name;
                        rec.sigma = sc.sigma;
                        rec.selectivity_pct = sel_pct;
                        rec.eligible_count = el_count;
                        rec.top_k = k;
                        rec.cache_mode = "WARM_CACHE";
                        rec.query_id = q;
                        rec.execution_order = order_str;
                        rec.time_5a_ns = ns_5a;
                        rec.time_5b_ns = ns_5b;
                        rec.time_6_ns = ns_6;
                        rec.best_fixed_plan = best_fixed_name;
                        rec.selected_plan = sel_plan_name;
                        rec.regret_ns = regret_ns;
                        rec.relative_regret = rel_regret;
                        rec.plan6_worse_than_both = worse_both;
                        rec.plan6_better_than_both = better_both;
                        rec.plan6_between_both = between_both;
                        rec.gate3_aborted = met_6.plan6_gate3_aborted;
                        rec.gate3_pilot_work = met_6.plan6_gate3_pilot_work;
                        rec.gate3_max_q_ratio = met_6.plan6_gate3_max_q_ratio;
                        rec.gate3_reason = g3_reason;
                        rec.exactness_status = exactness_status;

                        all_records.push_back(rec);

                        csv << rec.dataset_id << "," << rec.seed << "," << rec.scenario << ","
                            << rec.sigma << "," << rec.selectivity_pct << "," << rec.eligible_count << ","
                            << rec.top_k << "," << rec.cache_mode << "," << rec.query_id << ","
                            << rec.execution_order << "," << rec.time_5a_ns << "," << rec.time_5b_ns << ","
                            << rec.time_6_ns << "," << rec.best_fixed_plan << "," << rec.selected_plan << ","
                            << rec.regret_ns << "," << rec.relative_regret << ","
                            << static_cast<int>(rec.plan6_worse_than_both) << ","
                            << static_cast<int>(rec.plan6_better_than_both) << ","
                            << static_cast<int>(rec.plan6_between_both) << ","
                            << static_cast<int>(rec.gate3_aborted) << ","
                            << rec.gate3_pilot_work << "," << rec.gate3_max_q_ratio << ","
                            << rec.gate3_reason << "," << rec.exactness_status << "\n";
                    }

                    /* ==========================================================
                     * BLOQUE COLD_LLC_FLUSH (Eviccion forzada de 36 MB)
                     * ========================================================== */
                    for (uint32_t q = num_queries_warm; q < num_queries_warm + num_queries_cold; ++q) {
                        const float* q_vec = ds.queries[q].values;
                        uint32_t order_mode = q % 3U;
                        std::string order_str;
                        uint64_t t0, t1;
                        double ns_5a = 0.0, ns_5b = 0.0, ns_6 = 0.0;

                        if (order_mode == 0) {
                            order_str = "5A->5B->6";
                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            t1 = get_perf_counter_ns();
                            ns_5a = static_cast<double>(t1 - t0);

                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                            t1 = get_perf_counter_ns();
                            ns_5b = static_cast<double>(t1 - t0);

                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan6_ivf_adaptive_cost_model_avx2(idx, q_vec, pred, k, res_6, &count_6, met_6);
                            t1 = get_perf_counter_ns();
                            ns_6 = static_cast<double>(t1 - t0);
                        } else if (order_mode == 1) {
                            order_str = "5B->6->5A";
                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                            t1 = get_perf_counter_ns();
                            ns_5b = static_cast<double>(t1 - t0);

                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan6_ivf_adaptive_cost_model_avx2(idx, q_vec, pred, k, res_6, &count_6, met_6);
                            t1 = get_perf_counter_ns();
                            ns_6 = static_cast<double>(t1 - t0);

                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            t1 = get_perf_counter_ns();
                            ns_5a = static_cast<double>(t1 - t0);
                        } else {
                            order_str = "6->5A->5B";
                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan6_ivf_adaptive_cost_model_avx2(idx, q_vec, pred, k, res_6, &count_6, met_6);
                            t1 = get_perf_counter_ns();
                            ns_6 = static_cast<double>(t1 - t0);

                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan5a_ivf_part_prune_avx2(idx, q_vec, pred, k, res_5a, &count_5a, met_5a);
                            t1 = get_perf_counter_ns();
                            ns_5a = static_cast<double>(t1 - t0);

                            evictor.flush_llc();
                            t0 = get_perf_counter_ns();
                            execute_plan5b_ivf_full_exact_prune_avx2(idx, q_vec, pred, k, res_5b, &count_5b, met_5b);
                            t1 = get_perf_counter_ns();
                            ns_5b = static_cast<double>(t1 - t0);
                        }

                        bool match_5a_5b = compare_topk_exact(res_5a, res_5b, k);
                        bool match_5a_6 = compare_topk_exact(res_5a, res_6, k);
                        std::string exactness_status = (match_5a_5b && match_5a_6 && count_5a == k) ? "VERIFIED_100_MATCH" : "MISMATCH_DETECTED";

                        double best_fixed_ns = std::min(ns_5a, ns_5b);
                        std::string best_fixed_name = (ns_5a <= ns_5b) ? "5A" : "5B";
                        std::string sel_plan_name;
                        if (met_6.plan6_selected_cascade == 0) sel_plan_name = "5A_REMAIN";
                        else if (met_6.plan6_gate3_aborted) sel_plan_name = "5B_ABORTED_TO_5A";
                        else sel_plan_name = "5B_CASCADE";

                        double regret_ns = ns_6 - best_fixed_ns;
                        double rel_regret = regret_ns / (best_fixed_ns > 0.0 ? best_fixed_ns : 1.0);

                        uint8_t worse_both = (ns_6 > std::max(ns_5a, ns_5b) + 500.0) ? 1 : 0;
                        uint8_t better_both = (ns_6 < std::min(ns_5a, ns_5b) - 500.0) ? 1 : 0;
                        uint8_t between_both = (!worse_both && !better_both) ? 1 : 0;

                        std::string g3_reason = "NONE";
                        if (met_6.plan6_selected_cascade == 0) g3_reason = "GATE1_OR_GATE2_FILTER";
                        else if (met_6.plan6_gate3_aborted) g3_reason = "PILOT_EXCEEDED_MAX_RATIO";
                        else g3_reason = "CASCADE_COMPLETED";

                        QueryRegretRecord rec;
                        rec.dataset_id = ds_id;
                        rec.seed = seed;
                        rec.scenario = sc.name;
                        rec.sigma = sc.sigma;
                        rec.selectivity_pct = sel_pct;
                        rec.eligible_count = el_count;
                        rec.top_k = k;
                        rec.cache_mode = "COLD_LLC_FLUSH";
                        rec.query_id = q;
                        rec.execution_order = order_str;
                        rec.time_5a_ns = ns_5a;
                        rec.time_5b_ns = ns_5b;
                        rec.time_6_ns = ns_6;
                        rec.best_fixed_plan = best_fixed_name;
                        rec.selected_plan = sel_plan_name;
                        rec.regret_ns = regret_ns;
                        rec.relative_regret = rel_regret;
                        rec.plan6_worse_than_both = worse_both;
                        rec.plan6_better_than_both = better_both;
                        rec.plan6_between_both = between_both;
                        rec.gate3_aborted = met_6.plan6_gate3_aborted;
                        rec.gate3_pilot_work = met_6.plan6_gate3_pilot_work;
                        rec.gate3_max_q_ratio = met_6.plan6_gate3_max_q_ratio;
                        rec.gate3_reason = g3_reason;
                        rec.exactness_status = exactness_status;

                        all_records.push_back(rec);

                        csv << rec.dataset_id << "," << rec.seed << "," << rec.scenario << ","
                            << rec.sigma << "," << rec.selectivity_pct << "," << rec.eligible_count << ","
                            << rec.top_k << "," << rec.cache_mode << "," << rec.query_id << ","
                            << rec.execution_order << "," << rec.time_5a_ns << "," << rec.time_5b_ns << ","
                            << rec.time_6_ns << "," << rec.best_fixed_plan << "," << rec.selected_plan << ","
                            << rec.regret_ns << "," << rec.relative_regret << ","
                            << static_cast<int>(rec.plan6_worse_than_both) << ","
                            << static_cast<int>(rec.plan6_better_than_both) << ","
                            << static_cast<int>(rec.plan6_between_both) << ","
                            << static_cast<int>(rec.gate3_aborted) << ","
                            << rec.gate3_pilot_work << "," << rec.gate3_max_q_ratio << ","
                            << rec.gate3_reason << "," << rec.exactness_status << "\n";
                    }
                }
            }
        }
    }

    csv.close();
    std::cout << "\n================================================================================" << std::endl;
    std::cout << "Evaluacion completada. Total consultas medidas: " << all_records.size() << std::endl;
    std::cout << "CSV canónico generado: " << out_csv_path << std::endl;

    /* Analisis estadistico de Regret y Optimalidad */
    std::vector<double> rel_regrets_warm, rel_regrets_cold;
    uint32_t count_worse_both = 0, count_better_both = 0, count_between_both = 0;
    uint32_t correct_decision_count = 0;

    for (const auto& r : all_records) {
        if (r.cache_mode == "WARM_CACHE") rel_regrets_warm.push_back(r.relative_regret);
        else rel_regrets_cold.push_back(r.relative_regret);

        if (r.plan6_worse_than_both) count_worse_both++;
        if (r.plan6_better_than_both) count_better_both++;
        if (r.plan6_between_both) count_between_both++;

        bool chosen_5a = (r.selected_plan == "5A_REMAIN" || r.selected_plan == "5B_ABORTED_TO_5A");
        bool best_5a = (r.best_fixed_plan == "5A");
        if (chosen_5a == best_5a) correct_decision_count++;
    }

    auto compute_stats = [](std::vector<double>& v, double& mean, double& p50, double& p95, double& p99) {
        if (v.empty()) return;
        std::sort(v.begin(), v.end());
        double sum = std::accumulate(v.begin(), v.end(), 0.0);
        mean = sum / static_cast<double>(v.size());
        p50 = v[v.size() * 50 / 100];
        p95 = v[v.size() * 95 / 100];
        p99 = v[v.size() * 99 / 100];
    };

    double w_mean = 0, w_p50 = 0, w_p95 = 0, w_p99 = 0;
    double c_mean = 0, c_p50 = 0, c_p95 = 0, c_p99 = 0;
    compute_stats(rel_regrets_warm, w_mean, w_p50, w_p95, w_p99);
    compute_stats(rel_regrets_cold, c_mean, c_p50, c_p95, c_p99);

    double pct_worse = (100.0 * count_worse_both) / static_cast<double>(all_records.size());
    double pct_better = (100.0 * count_better_both) / static_cast<double>(all_records.size());
    double pct_between = (100.0 * count_between_both) / static_cast<double>(all_records.size());
    double sel_accuracy = (100.0 * correct_decision_count) / static_cast<double>(all_records.size());

    std::cout << "\n--------------------------------------------------------------------------------" << std::endl;
    std::cout << "RESUMEN ESTADISTICO GLOBAL DE REGRET Y SELECCION ADAPTATIVA" << std::endl;
    std::cout << "--------------------------------------------------------------------------------" << std::endl;
    std::cout << "Precision de Seleccion (Accuracy vs min(5A,5B)) : " << std::fixed << std::setprecision(2) << sel_accuracy << "%" << std::endl;
    std::cout << "Consultas donde Plan 6 es MEJOR que ambos       : " << pct_better << "%" << std::endl;
    std::cout << "Consultas donde Plan 6 queda ENTRE ambos        : " << pct_between << "%" << std::endl;
    std::cout << "Consultas donde Plan 6 es PEOR que ambos        : " << pct_worse << "% (Criterio cierre: <= 5.0%)" << std::endl;
    std::cout << "\nRegret Relativo WARM_CACHE: Mean=" << (w_mean * 100.0) << "% | p50=" << (w_p50 * 100.0) 
              << "% | p95=" << (w_p95 * 100.0) << "% | p99=" << (w_p99 * 100.0) << "%" << std::endl;
    std::cout << "Regret Relativo COLD_CACHE: Mean=" << (c_mean * 100.0) << "% | p50=" << (c_p50 * 100.0) 
              << "% | p95=" << (c_p95 * 100.0) << "% | p99=" << (c_p99 * 100.0) << "%" << std::endl;
    std::cout << "================================================================================" << std::endl;

    return 0;
}
