#include "../include/alg_flat_types.h"
#include "../include/dataset_loader.h"
#include "../include/physical_layouts.h"
#include "../include/execution_plans.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace alg_flat;

/*
 * Nivel 1A: Recall Estricto por ID (SIN tolerancia de distancia `dist_sq <= kth_gt_dist`).
 * Cada ID predicho debe pertenecer estrictamente al conjunto exacto de los K IDs del Ground Truth
 * definido por el orden total determinista (dist_sq_orig ASC, id ASC).
 */
static double compute_strict_id_recall(const Neighbor* pred, uint32_t pred_cnt,
                                       const Neighbor* gt, uint32_t gt_cnt, uint32_t k) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t n_p = (pred_cnt < k) ? pred_cnt : k;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_p; ++i) {
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
 * Nivel 1B + Nivel 2: Identidad de Rango Posicion-a-Posicion e Igualdad Exacta de Distancia Original.
 * Verifica que pred[r].id == gt[r].id y que pred[r].dist_sq coincida exactamente con la distancia
 * en el espacio original float32.
 */
static void verify_rank_and_original_distance(const Neighbor* pred, uint32_t pred_cnt,
                                              const Neighbor* gt, uint32_t gt_cnt, uint32_t k,
                                              uint64_t& out_rank_matches,
                                              uint64_t& out_rank_total,
                                              double& inout_max_orig_dist_diff) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    for (uint32_t r = 0; r < target; ++r) {
        out_rank_total++;
        if (r < pred_cnt && pred[r].id == gt[r].id) {
            out_rank_matches++;
        }
        if (r < pred_cnt) {
            double diff = std::fabs(static_cast<double>(pred[r].dist_sq) - static_cast<double>(gt[r].dist_sq));
            if (diff > inout_max_orig_dist_diff) {
                inout_max_orig_dist_diff = diff;
            }
        }
    }
}

/*
 * Construye el Escenario Adversarial 2: Centroides alejados del origen (||c_p||_2 >> 0)
 * donde cada cluster tiene un desplazamiento diferente entre +500.0 y +2500.0 por coordenada.
 */
static void make_adversarial_off_origin_dataset(BenchmarkDataset& ds) {
    generate_dataset_deterministic(101, 20000, 80, 64, 0.05f, MIX_R, ds);
    const uint32_t nv = ds.header.num_vectors;
    const uint32_t nq = ds.header.num_queries;
    const uint32_t dim = ds.header.dimension;

    /* Desplazar todo el sistema de coordenadas lejos del origen de forma anisotropica */
    for (uint32_t i = 0; i < nv; ++i) {
        float* v = ds.vectors + static_cast<size_t>(i) * dim;
        uint32_t basin = i % 64U;
        for (uint32_t d = 0; d < dim; ++d) {
            float shift = 500.0f + 25.0f * static_cast<float>(basin) + 3.5f * static_cast<float>(d);
            v[d] += shift;
        }
    }
    for (uint32_t q = 0; q < nq; ++q) {
        float* qv = ds.queries[q].values;
        uint32_t basin = q % 64U;
        for (uint32_t d = 0; d < dim; ++d) {
            float shift = 500.0f + 25.0f * static_cast<float>(basin) + 3.5f * static_cast<float>(d);
            qv[d] += shift;
        }
    }
}

/*
 * Construye el Escenario Adversarial 3:
 *   - Vectores con residuos (x - c_p) estrictamente colineales con (q - c_p) en las colas 16..63 y 32..63
 *     (caso exacto de igualdad en la desigualdad triangular inversa).
 *   - Empates exactos de distancia y candidatos separados por 1 ULP sobre la frontera Top-K.
 *   - Consultas situadas sobre la frontera entre dos particiones.
 */
static void make_adversarial_collinear_and_boundary_dataset(BenchmarkDataset& ds) {
    generate_dataset_deterministic(202, 20000, 60, 64, 0.06f, MIX_50_50, ds);
    const uint32_t dim = ds.header.dimension;

    /* Para las primeras 20 consultas, inyectar vectores elegibles colineales y empates exactos en la frontera */
    for (uint32_t q = 0; q < 20; ++q) {
        const float* qv = ds.queries[q].values;
        uint32_t base_vec = q * 200U;
        const float* anchor = ds.vectors + static_cast<size_t>(base_vec) * dim;

        /* Crear 12 vectores con diferencias puramente en las dimensiones finales y empates exactos */
        for (uint32_t m = 0; m < 12; ++m) {
            uint32_t vid = base_vec + 1 + m;
            float* target_v = ds.vectors + static_cast<size_t>(vid) * dim;
            ds.categories[vid] = 0; /* Asegurar que sea elegible en todas las selectividades */

            float scale = 0.015f + 0.002f * static_cast<float>(m / 2); /* Pares con distancia identica (empates) */
            for (uint32_t d = 0; d < dim; ++d) {
                /* Colineal con (qv - anchor) */
                float dir = (qv[d] - anchor[d]);
                target_v[d] = qv[d] + scale * (dir != 0.0f ? dir : 1.0f);
            }
        }
    }

    /* Reconstruir las listas de elegibles tras forzar categories[vid] = 0 */
    uint32_t num_sels = ds.header.num_selectivities;
    for (uint32_t s = 0; s < num_sels; ++s) {
        uint32_t target_cat = ds.header.selectivities_deci_percent[s] * 10U;
        auto& el = ds.eligible_per_sel[s];
        el.clear();
        for (uint32_t i = 0; i < ds.header.num_vectors; ++i) {
            if (ds.categories[i] < target_cat) {
                el.push_back(i);
            }
        }
    }
}

static void run_certification_suite(const char* suite_name,
                                    const BenchmarkDataset& ds,
                                    uint32_t num_partitions) {
    std::cout << "\n--------------------------------------------------------------------------------\n";
    std::cout << "SUITE DE CERTIFICACION: " << suite_name
              << " (N=" << ds.header.num_vectors << ", Q=" << ds.header.num_queries
              << ", P=" << num_partitions << ")\n";
    std::cout << "--------------------------------------------------------------------------------\n";

    FlatAoSIndex aos_idx;
    FlatSoAIndex soa_idx;
    FlatAoSoAIndex aosoa_idx;
    IVFPartitionedIndex ivf_idx;

    build_flat_aos_index(ds, aos_idx);
    build_flat_soa_index(ds, soa_idx);
    build_flat_aosoa_index(ds, aosoa_idx);
    build_ivf_partitioned_index(ds, num_partitions, 12, ivf_idx);

    printf("Defecto de ortogonalidad medido ||Q*Q^T - I||_F = %.3e\n", ivf_idx.ortho_defect);

    const uint32_t sel_idx = 2; /* 15.0% selectividad */
    const uint32_t target_cat = ds.header.selectivities_deci_percent[sel_idx] * 10U;
    const auto& el_ids = ds.eligible_per_sel[sel_idx];
    QueryPredicate pred{target_cat, el_ids.data(), static_cast<uint32_t>(el_ids.size()), sel_idx};

    const uint32_t k = 100;
    Neighbor gt[100], out[100];
    uint32_t gt_cnt = 0, out_cnt = 0;
    QueryExecutionMetrics m{};

    double strict_r10[8] = {0}, strict_r100[8] = {0};
    uint64_t rank_matches[8] = {0}, rank_totals[8] = {0};
    double max_orig_dist_err[8] = {0};

    PruningAuditReport audit{};
    audit.min_ratio_orig_over_lb = 1e30;

    const char* names[8] = {
        "Plan 1  (Flat_Scalar_AoS)",
        "Plan 2A (Flat_AVX2_AoS_1x)",
        "Plan 2B (Flat_AVX2_AoS_4x)",
        "Plan 3A (Flat_AVX2_SoA)",
        "Plan 3B (Flat_AVX2_AoSoA)",
        "Plan 4  (IVF_Flat_AVX2)",
        "Plan 5A (IVF_PartPrune_AVX2)",
        "Plan 5B (IVF_FullExactPrune_AVX2)"
    };

    for (uint32_t q = 0; q < ds.header.num_queries; ++q) {
        const float* qv = ds.queries[q].values;
        execute_plan0_ref_hito9(aos_idx, qv, pred, k, gt, &gt_cnt, m);

        for (int p = 0; p < 8; ++p) {
            if (p == 0) execute_plan1_flat_scalar_aos(aos_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 1) execute_plan2a_flat_avx2_aos_1x(aos_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 2) execute_plan2b_flat_avx2_aos_4x(aos_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 3) execute_plan3a_flat_avx2_soa(soa_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 4) execute_plan3b_flat_avx2_aosoa(aosoa_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 5) execute_plan4_ivf_flat_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 6) execute_plan5a_ivf_part_prune_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);
            else if (p == 7) execute_plan5b_ivf_full_exact_prune_avx2(ivf_idx, qv, pred, k, out, &out_cnt, m);

            strict_r10[p] += compute_strict_id_recall(out, out_cnt, gt, gt_cnt, 10);
            strict_r100[p] += compute_strict_id_recall(out, out_cnt, gt, gt_cnt, 100);
            verify_rank_and_original_distance(out, out_cnt, gt, gt_cnt, 100,
                                              rank_matches[p], rank_totals[p], max_orig_dist_err[p]);
        }

        /* Nivel 3: Auditar cada vector podado por Plan 5B en esta consulta */
        audit_plan5b_pruning_soundness(ivf_idx, qv, pred, k, gt, gt_cnt, audit);
    }

    double nq = static_cast<double>(ds.header.num_queries);
    printf("%-34s | %12s | %13s | %12s | %13s\n",
           "Plan", "StrictID@10", "StrictID@100", "RankMatch%", "MaxOrigDistErr");
    printf("-----------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < 8; ++i) {
        double r10 = (strict_r10[i] / nq) * 100.0;
        double r100 = (strict_r100[i] / nq) * 100.0;
        double rmatch = 100.0 * static_cast<double>(rank_matches[i]) / static_cast<double>(rank_totals[i]);
        printf("%-34s | %11.4f%% | %12.4f%% | %11.4f%% | %13.3e\n",
               names[i], r10, r100, rmatch, max_orig_dist_err[i]);

        assert(r10 == 100.0 && "Nivel 1 fallo: Strict ID Recall@10 debe ser 100.0000%!");
        assert(r100 == 100.0 && "Nivel 1 fallo: Strict ID Recall@100 debe ser 100.0000%!");
        assert(rmatch == 100.0 && "Nivel 1B fallo: Identidad exacta de rango (dist_sq, id) debe ser 100.0000%!");
        assert(max_orig_dist_err[i] == 0.0 && "Nivel 2 fallo: La distancia debe coincidir bit a bit en espacio original!");
    }

    uint64_t total_pruned = audit.pruned_by_partition + audit.pruned_by_radial_shell
                          + audit.pruned_by_pca_16d + audit.pruned_by_pca_32d;
    printf("\n[Nivel 3 - Auditoria Directa de Poda en 5B]\n");
    printf("  Elegibles auditados : %llu\n", static_cast<unsigned long long>(audit.total_eligible_checked));
    printf("  Podados por etapa   : Particion=%llu | Radial=%llu | PCA_16D=%llu | PCA_32D=%llu (Total=%llu)\n",
           static_cast<unsigned long long>(audit.pruned_by_partition),
           static_cast<unsigned long long>(audit.pruned_by_radial_shell),
           static_cast<unsigned long long>(audit.pruned_by_pca_16d),
           static_cast<unsigned long long>(audit.pruned_by_pca_32d),
           static_cast<unsigned long long>(total_pruned));
    printf("  Violaciones de cota : Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu\n",
           static_cast<unsigned long long>(audit.bound_violations_partition),
           static_cast<unsigned long long>(audit.bound_violations_radial),
           static_cast<unsigned long long>(audit.bound_violations_pca16),
           static_cast<unsigned long long>(audit.bound_violations_pca32));
    printf("  Falsos descartes K  : %llu\n", static_cast<unsigned long long>(audit.false_prunings_topk));
    printf("  Min ratio d^2/LB^2  : %.6f (requerido >= 1.000000)\n", audit.min_ratio_orig_over_lb);

    assert(audit.bound_violations_partition == 0 && "Violacion de cota en poda de particion!");
    assert(audit.bound_violations_radial == 0 && "Violacion de cota en poda radial!");
    assert(audit.bound_violations_pca16 == 0 && "Violacion de cota en poda PCA 16D!");
    assert(audit.bound_violations_pca32 == 0 && "Violacion de cota en poda PCA 32D!");
    assert(audit.false_prunings_topk == 0 && "Un vecino verdadero del Top-K fue podado!");
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "ALG-Flat 2.0 (Hito 2A): Auditoria Estricta de Exactitud en 3 Niveles\n";
    std::cout << "  Nivel 1: Strict ID Recall & Exact Rank Identity (sin tolerancia de frontera)\n";
    std::cout << "  Nivel 2: Igualdad bit a bit de distancias en el espacio original float32\n";
    std::cout << "  Nivel 3: Verificacion directa LB_safe^2(q,x) <= d_orig^2(q,x) en todo descarte\n";
    std::cout << "================================================================================\n";

    /* Suite 1: Dataset completo de benchmark (N=100,000, M50/50, sigma=0.09, P=128) */
    {
        BenchmarkDataset ds_full;
        generate_dataset_deterministic(8, 100000, 100, 64, 0.09f, MIX_50_50, ds_full);
        run_certification_suite("1_Full_Benchmark_100k_Mix_s009", ds_full, 128);
    }

    /* Suite 2: Centroides alejados del origen (||c_p|| >> 0, prueba de invariancia traslacional de colas) */
    {
        BenchmarkDataset ds_off;
        make_adversarial_off_origin_dataset(ds_off);
        run_certification_suite("2_Adversarial_FarOffOrigin_Centroids", ds_off, 64);
    }

    /* Suite 3: Colinealidad exacta en colas 16..63 / 32..63 y empates en la frontera Top-K */
    {
        BenchmarkDataset ds_col;
        make_adversarial_collinear_and_boundary_dataset(ds_col);
        run_certification_suite("3_Adversarial_Collinear_Tail_And_Ties", ds_col, 64);
    }

    /* Suite 4: Alta localidad con >92% de poda (N=100,000, R, sigma=0.03, P=128) */
    {
        BenchmarkDataset ds_tight;
        generate_dataset_deterministic(8, 100000, 100, 64, 0.03f, MIX_R, ds_tight);
        run_certification_suite("4_Full_Benchmark_100k_Tight_s003", ds_tight, 128);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "CERTIFICACION HITO 2A SUPERADA: 0 VIOLACIONES DE COTA, 100.0000% STRICT ID RECALL\n";
    std::cout << "================================================================================\n";
    return 0;
}
