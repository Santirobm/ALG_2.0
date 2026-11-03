#include "../include/alg_flat_types.h"
#include "../include/dataset_loader.h"
#include "../include/physical_layouts.h"
#include "../include/execution_plans.h"
#include "../include/ieee754_bounds.h"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdio>

using namespace alg_flat;

/*
 * Nivel 1A: IDSetRecall (SIN tolerancia de distancia `dist_sq <= kth_gt_dist` y CON verificacion
 * explicita de ausencia de IDs duplicados en la salida predicha).
 * Cada ID predicho debe ser unico y pertenecer estrictamente al conjunto exacto de los K IDs del
 * Ground Truth definido por el orden total determinista (dist_sq_orig ASC, id ASC).
 */
static double compute_id_set_recall(const Neighbor* pred, uint32_t pred_cnt,
                                    const Neighbor* gt, uint32_t gt_cnt, uint32_t k,
                                    uint64_t& inout_duplicate_id_errors) {
    uint32_t target = (gt_cnt < k) ? gt_cnt : k;
    if (target == 0) return 1.0;
    uint32_t n_p = (pred_cnt < k) ? pred_cnt : k;
    uint32_t hits = 0;
    for (uint32_t i = 0; i < n_p; ++i) {
        bool is_duplicate = false;
        for (uint32_t prev = 0; prev < i; ++prev) {
            if (pred[prev].id == pred[i].id) {
                is_duplicate = true;
                inout_duplicate_id_errors++;
                break;
            }
        }
        if (is_duplicate) continue;

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
 * Nivel 1B + Nivel 2: Identidad de Rango Posicion-a-Posicion (RankMatch@K) e Igualdad Exacta
 * de Distancia Original (MaxOrigDistErr).
 * Verifica que pred[r].id == gt[r].id en cada posicion r < K y que pred[r].dist_sq coincida
 * bit a bit con la distancia en el espacio original float32.
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
 * Prepara los vectores base para el Escenario Adversarial 3 con empates exactos de distancia
 * y pares separados por 1 ULP. Las consultas colineales respecto del centroide real c_p se
 * construyen y verifican una vez conocidas las particiones reales de K-Means.
 */
static void make_adversarial_ties_and_base_dataset(BenchmarkDataset& ds) {
    generate_dataset_deterministic(202, 20000, 60, 32, 0.05f, MIX_50_50, ds);
    const uint32_t dim = ds.header.dimension;

    /* Asegurar densidad local suficiente (>= 140 elegibles por particion) para que r_K sea local
     * a cada particion y permita podar vectores colineales x* dentro de su propia particion c_p */
    for (uint32_t i = 0; i < ds.header.num_vectors; ++i) {
        if ((i % 2U) == 0U) {
            ds.categories[i] = 0;
        }
    }

    /* En las consultas 50..59 inyectar pares de vectores elegibles con distancias identicas (empates)
     * y pares separados por 1 ULP alrededor de cada consulta */
    for (uint32_t q = 50; q < 60; ++q) {
        const float* qv = ds.queries[q].values;
        uint32_t base_vec = q * 200U;
        for (uint32_t m = 0; m < 16; ++m) {
            uint32_t vid = base_vec + m;
            float* target_v = ds.vectors + static_cast<size_t>(vid) * dim;
            ds.categories[vid] = 0; /* Elegible en todas las selectividades */
            float offset = 0.010f + 0.001f * static_cast<float>(m / 2);
            for (uint32_t d = 0; d < dim; ++d) {
                target_v[d] = qv[d] + ((d == (m % dim)) ? -offset : offset);
            }
        }
    }

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

/*
 * Construye y verifica explicitamente consultas con colinealidad exacta respecto del centroide
 * real c_p asignado por K-Means en ivf_idx:
 *   - Caso A (q in [0, 14]) : (q - c_p) = alpha * (x* - c_p) en 64D (igualdad en cota radial).
 *   - Caso B (q in [15, 34]): ||q - c_p||_2 == ||x* - c_p||_2 (cota radial = 0, no poda en radial)
 *                             y q_rot[16..63] = alpha * x*_rot[16..63] (colinealidad pura en 16..63,
 *                             donde d_orig^2(q, x*) == LB_16_exact^2(q, x*) y x* se poda en PCA 16D).
 *   - Caso C (q in [35, 49]): ||q - c_p||_2 == ||x* - c_p||_2 y ||q_rot[16..63]||_2 == ||x*_rot[16..63]||_2
 *                             con q_rot[0..15] == x*_rot[0..15] (cotas radial y PCA 16D = 0, pasa a PCA 32D)
 *                             y q_rot[32..63] = alpha * x*_rot[32..63] (colinealidad pura en 32..63,
 *                             donde d_orig^2(q, x*) == LB_32_exact^2(q, x*) y x* se poda en PCA 32D).
 */
static void inject_and_verify_true_centroid_collinearity(BenchmarkDataset& ds,
                                                         const IVFPartitionedIndex& ivf_idx,
                                                         uint32_t target_cat) {
    const uint32_t P = ivf_idx.num_partitions;
    const uint32_t dim = ivf_idx.dim;
    double min_cos_radial64 = 1.0;
    double min_cos_tail16   = 1.0;
    double min_cos_tail32   = 1.0;
    double max_rel_gap_radial64 = 0.0;
    double max_rel_gap_tail16   = 0.0;
    double max_rel_gap_tail32   = 0.0;
    double max_unadjusted_ratio_err_radial = 0.0;
    double max_unadjusted_ratio_err_16     = 0.0;
    double max_unadjusted_ratio_err_32     = 0.0;
    uint32_t injected_pairs = 0;

    for (uint32_t q = 0; q < 50; ++q) {
        uint32_t chosen_p = 0;
        uint32_t chosen_pos = 0;
        bool found = false;
        for (uint32_t attempt = 0; attempt < P; ++attempt) {
            uint32_t p = (q + attempt) % P;
            const GeometricPartition& pm = ivf_idx.partitions[p];
            if (pm.count < 180) continue;
            /* Recorrer desde el borde exterior de la particion (mayor radio) para que los vectores
             * x* con q par tengan norma suficiente para superar r_K y ser podados en 5B */
            for (int64_t i = static_cast<int64_t>(pm.aos_offset + pm.count) - 1;
                 i >= static_cast<int64_t>(pm.aos_offset); --i) {
                if (ivf_idx.part_aos_categories[i] < target_cat &&
                    ivf_idx.part_aos_tail32_norm[i] > 0.08f &&
                    ivf_idx.part_aos_tail48_norm[i] > 0.12f) {
                    chosen_p = p;
                    chosen_pos = static_cast<uint32_t>(i);
                    found = true;
                    break;
                }
            }
            if (found) break;
        }
        assert(found && "No se encontro vector elegible con cola no nula en las particiones!");

        const float* c_p = ivf_idx.centroids + static_cast<size_t>(chosen_p) * dim;
        const float* x_orig = ivf_idx.part_orig_vectors + static_cast<size_t>(chosen_pos) * dim;

        /* Calcular residuo rotado exacto de x* respecto de su centroide real c_p: x_rot = Q^T * (x - c_p) */
        double x_rot[64];
        double A2 = 0.0, B2 = 0.0, C2 = 0.0;
        for (uint32_t j = 0; j < dim; ++j) {
            const float* row = ivf_idx.ortho_basis + static_cast<size_t>(j) * dim;
            double acc = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                acc += static_cast<double>(row[d]) * (static_cast<double>(x_orig[d]) - static_cast<double>(c_p[d]));
            }
            x_rot[j] = acc;
            if (j < 16) A2 += acc * acc;
            else if (j < 32) B2 += acc * acc;
            else C2 += acc * acc;
        }

        double q_rot_target[64];
        if (q < 15) {
            /* Caso A (q in 0..14): Colinealidad completa 64D respecto del centroide real c_p.
             * Para q par: alpha = 0.08 (q cerca de c_p donde r_K es pequeno y x* en el borde exterior
             * de la particion cumple r_vec - dc_high > r_K, podandose en la etapa Radial con cos=1.0);
             * para q impar: alpha = 0.98 (x* dentro de Top-K). */
            const double alpha = (q % 2 == 0) ? 0.08 : 0.98;
            for (uint32_t j = 0; j < dim; ++j) {
                q_rot_target[j] = alpha * x_rot[j];
            }
        } else if (q < 35) {
            /* Caso B (q in 15..34): ||q - c_p||_2 == ||x* - c_p||_2 (LB_radial == 0, NO poda en radial)
             * y colinealidad exacta en 16..63: q_rot[16..63] = alpha * x_rot[16..63]. */
            if (q % 2 == 0) {
                const double alpha = 0.35;
                const double beta = std::sqrt((A2 + (1.0 - alpha * alpha) * (B2 + C2)) / A2);
                for (uint32_t j = 0; j < dim; ++j) {
                    q_rot_target[j] = (j < 16) ? (-beta * x_rot[j]) : (alpha * x_rot[j]);
                }
            } else {
                const double alpha = 0.98;
                for (uint32_t j = 0; j < dim; ++j) {
                    q_rot_target[j] = (j < 16) ? x_rot[j] : (alpha * x_rot[j]);
                }
            }
        } else {
            /* Caso C (q in 35..49): ||q - c_p||_2 == ||x* - c_p||_2 y ||q_rot[16..63]||_2 == ||x*_rot[16..63]||_2
             * con q_rot[0..15] == x_rot[0..15] (LB_radial == 0 y LB_16 == 0, pasa a PCA 32D)
             * y colinealidad exacta en 32..63: q_rot[32..63] = alpha * x_rot[32..63]. */
            if (q % 2 == 0) {
                const double alpha = 0.35;
                const double gamma = std::sqrt((B2 + (1.0 - alpha * alpha) * C2) / B2);
                for (uint32_t j = 0; j < dim; ++j) {
                    if (j < 16) q_rot_target[j] = x_rot[j];
                    else if (j < 32) q_rot_target[j] = -gamma * x_rot[j];
                    else q_rot_target[j] = alpha * x_rot[j];
                }
            } else {
                const double alpha = 0.98;
                for (uint32_t j = 0; j < dim; ++j) {
                    q_rot_target[j] = (j < 32) ? x_rot[j] : (alpha * x_rot[j]);
                }
            }
        }

        /* Reconstruir q en el espacio original: q = c_p + Q * q_rot_target (donde Q = (Q^T)^T) */
        float* qv = ds.queries[q].values;
        for (uint32_t d = 0; d < dim; ++d) {
            double acc = static_cast<double>(c_p[d]);
            for (uint32_t j = 0; j < dim; ++j) {
                const float* row = ivf_idx.ortho_basis + static_cast<size_t>(j) * dim;
                acc += static_cast<double>(row[d]) * q_rot_target[j];
            }
            qv[d] = static_cast<float>(acc);
        }

        /* Verificar ahora sobre el vector qv float32 final que (qv - c_p) y (x_orig - c_p)
         * cumplen colinealidad estricta respecto del centroide real c_p de la particion */
        double q_rot_actual[64];
        for (uint32_t j = 0; j < dim; ++j) {
            const float* row = ivf_idx.ortho_basis + static_cast<size_t>(j) * dim;
            double acc = 0.0;
            for (uint32_t d = 0; d < dim; ++d) {
                acc += static_cast<double>(row[d]) * (static_cast<double>(qv[d]) - static_cast<double>(c_p[d]));
            }
            q_rot_actual[j] = acc;
        }

        double d_orig_sq = 0.0;
        for (uint32_t d = 0; d < dim; ++d) {
            double df = static_cast<double>(qv[d]) - static_cast<double>(x_orig[d]);
            d_orig_sq += df * df;
        }

        if (q < 15) {
            double dot64 = 0.0, nq64 = 0.0, nx64 = 0.0, diff64 = 0.0;
            for (uint32_t j = 0; j < dim; ++j) {
                dot64 += q_rot_actual[j] * x_rot[j];
                nq64  += q_rot_actual[j] * q_rot_actual[j];
                nx64  += x_rot[j] * x_rot[j];
                double df = q_rot_actual[j] - x_rot[j];
                diff64 += df * df;
            }
            double cos64 = dot64 / (std::sqrt(nq64) * std::sqrt(nx64));
            if (cos64 < min_cos_radial64) min_cos_radial64 = cos64;

            double true_dist = std::sqrt(diff64);
            double rev_tri   = std::fabs(std::sqrt(nq64) - std::sqrt(nx64));
            double rel_gap   = std::fabs(true_dist - rev_tri) / true_dist;
            if (rel_gap > max_rel_gap_radial64) max_rel_gap_radial64 = rel_gap;

            double lb_radial_unadjusted = rev_tri * rev_tri;
            double ratio_err = std::fabs((d_orig_sq / lb_radial_unadjusted) - 1.0);
            if (ratio_err > max_unadjusted_ratio_err_radial) max_unadjusted_ratio_err_radial = ratio_err;
        } else if (q < 35) {
            double dot16 = 0.0, nq16 = 0.0, nx16 = 0.0, diff16 = 0.0, pref16 = 0.0;
            for (uint32_t j = 0; j < 16; ++j) {
                double df = q_rot_actual[j] - x_rot[j];
                pref16 += df * df;
            }
            for (uint32_t j = 16; j < dim; ++j) {
                dot16 += q_rot_actual[j] * x_rot[j];
                nq16  += q_rot_actual[j] * q_rot_actual[j];
                nx16  += x_rot[j] * x_rot[j];
                double df = q_rot_actual[j] - x_rot[j];
                diff16 += df * df;
            }
            double cos16 = dot16 / (std::sqrt(nq16) * std::sqrt(nx16));
            if (cos16 < min_cos_tail16) min_cos_tail16 = cos16;

            double true_tail_dist = std::sqrt(diff16);
            double rev_tri_bound  = std::fabs(std::sqrt(nq16) - std::sqrt(nx16));
            double rel_gap = std::fabs(true_tail_dist - rev_tri_bound) / true_tail_dist;
            if (rel_gap > max_rel_gap_tail16) max_rel_gap_tail16 = rel_gap;

            double lb16_unadjusted = pref16 + rev_tri_bound * rev_tri_bound;
            double ratio_err = std::fabs((d_orig_sq / lb16_unadjusted) - 1.0);
            if (ratio_err > max_unadjusted_ratio_err_16) max_unadjusted_ratio_err_16 = ratio_err;
        } else {
            double dot32 = 0.0, nq32 = 0.0, nx32 = 0.0, diff32 = 0.0, pref32 = 0.0;
            for (uint32_t j = 0; j < 32; ++j) {
                double df = q_rot_actual[j] - x_rot[j];
                pref32 += df * df;
            }
            for (uint32_t j = 32; j < dim; ++j) {
                dot32 += q_rot_actual[j] * x_rot[j];
                nq32  += q_rot_actual[j] * q_rot_actual[j];
                nx32  += x_rot[j] * x_rot[j];
                double df = q_rot_actual[j] - x_rot[j];
                diff32 += df * df;
            }
            double cos32 = dot32 / (std::sqrt(nq32) * std::sqrt(nx32));
            if (cos32 < min_cos_tail32) min_cos_tail32 = cos32;

            double true_tail_dist = std::sqrt(diff32);
            double rev_tri_bound  = std::fabs(std::sqrt(nq32) - std::sqrt(nx32));
            double rel_gap = std::fabs(true_tail_dist - rev_tri_bound) / true_tail_dist;
            if (rel_gap > max_rel_gap_tail32) max_rel_gap_tail32 = rel_gap;

            double lb32_unadjusted = pref32 + rev_tri_bound * rev_tri_bound;
            double ratio_err = std::fabs((d_orig_sq / lb32_unadjusted) - 1.0);
            if (ratio_err > max_unadjusted_ratio_err_32) max_unadjusted_ratio_err_32 = ratio_err;
        }
        injected_pairs++;
    }

    printf("[Verificacion Geom. Suite 3 respecto al centroide real c_p asignado por K-Means]\n");
    printf("  Pares colineales inyectados : %u (25 fuera de Top-K podables, 25 dentro de Top-K)\n", injected_pairs);
    printf("  Min cos(theta) respecto c_p : 64D=%.9f | Cola16..63=%.9f | Cola32..63=%.9f\n",
           min_cos_radial64, min_cos_tail16, min_cos_tail32);
    printf("  Max gap relativo desig. tri.: 64D=%.3e | Cola16..63=%.3e | Cola32..63=%.3e\n",
           max_rel_gap_radial64, max_rel_gap_tail16, max_rel_gap_tail32);
    printf("  Max |d_orig^2/LB_exact^2 - 1|: Radial=%.3e | PCA16=%.3e | PCA32=%.3e\n",
           max_unadjusted_ratio_err_radial, max_unadjusted_ratio_err_16, max_unadjusted_ratio_err_32);

    assert(min_cos_radial64 > 0.999999 && "Fallo de colinealidad 64D respecto del centroide real c_p!");
    assert(min_cos_tail16 > 0.999999 && "Fallo de colinealidad en cola 16..63 respecto del centroide real c_p!");
    assert(min_cos_tail32 > 0.999999 && "Fallo de colinealidad en cola 32..63 respecto del centroide real c_p!");
    assert(max_rel_gap_radial64 < 1e-6 && "La cota radial 64D no alcanza la igualdad en la desigualdad triangular!");
    assert(max_rel_gap_tail16 < 1e-6 && "La cola 16..63 no alcanza la igualdad en la desigualdad triangular!");
    assert(max_rel_gap_tail32 < 1e-6 && "La cola 32..63 no alcanza la igualdad en la desigualdad triangular!");
    assert(max_unadjusted_ratio_err_radial < 1e-5 && "La cota radial sin holgura debe coincidir con d_orig^2!");
    assert(max_unadjusted_ratio_err_16 < 1e-5 && "La cota 16D sin holgura debe coincidir con d_orig^2!");
    assert(max_unadjusted_ratio_err_32 < 1e-5 && "La cota 32D sin holgura debe coincidir con d_orig^2!");
}

static void run_certification_suite(const char* suite_name,
                                    BenchmarkDataset& ds,
                                    uint32_t num_partitions,
                                    bool inject_true_collinear = false) {
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

    if (inject_true_collinear) {
        inject_and_verify_true_centroid_collinearity(ds, ivf_idx, target_cat);
    }

    const uint32_t k = 100;
    Neighbor gt[100], out[100];
    uint32_t gt_cnt = 0, out_cnt = 0;
    QueryExecutionMetrics m{};

    double id_set_r10[8] = {0}, id_set_r100[8] = {0};
    uint64_t dup_id_errors[8] = {0};
    uint64_t rank_matches_10[8] = {0}, rank_totals_10[8] = {0};
    uint64_t rank_matches_100[8] = {0}, rank_totals_100[8] = {0};
    double max_orig_dist_err[8] = {0};

    PruningAuditReport audit{};
    audit.min_ratio_orig_over_lb = 1e30;
    audit.max_signed_excess = -1e30;
    audit.max_positive_excess = 0.0;

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

            id_set_r10[p] += compute_id_set_recall(out, out_cnt, gt, gt_cnt, 10, dup_id_errors[p]);
            id_set_r100[p] += compute_id_set_recall(out, out_cnt, gt, gt_cnt, 100, dup_id_errors[p]);
            double dummy_err = 0.0;
            verify_rank_and_original_distance(out, out_cnt, gt, gt_cnt, 10,
                                              rank_matches_10[p], rank_totals_10[p], dummy_err);
            verify_rank_and_original_distance(out, out_cnt, gt, gt_cnt, 100,
                                              rank_matches_100[p], rank_totals_100[p], max_orig_dist_err[p]);
        }

        /* Nivel 3: Auditar cada vector podado por Plan 5B en esta consulta (0 tolerancia) */
        audit_plan5b_pruning_soundness(ivf_idx, qv, pred, k, gt, gt_cnt, audit);
    }

    double nq = static_cast<double>(ds.header.num_queries);
    printf("%-34s | %12s | %13s | %12s | %13s | %7s | %13s\n",
           "Plan", "IDSetRec@10", "IDSetRec@100", "RankMatch@10", "RankMatch@100", "DupIDs", "MaxOrigDistErr");
    printf("-----------------------------------------------------------------------------------------------------------------------\n");
    for (int i = 0; i < 8; ++i) {
        double r10 = (id_set_r10[i] / nq) * 100.0;
        double r100 = (id_set_r100[i] / nq) * 100.0;
        double rm10 = 100.0 * static_cast<double>(rank_matches_10[i]) / static_cast<double>(rank_totals_10[i]);
        double rm100 = 100.0 * static_cast<double>(rank_matches_100[i]) / static_cast<double>(rank_totals_100[i]);
        printf("%-34s | %11.4f%% | %12.4f%% | %11.4f%% | %12.4f%% | %7llu | %13.3e\n",
               names[i], r10, r100, rm10, rm100,
               static_cast<unsigned long long>(dup_id_errors[i]), max_orig_dist_err[i]);

        assert(dup_id_errors[i] == 0 && "Nivel 1A fallo: Se detectaron IDs duplicados en la salida!");
        assert(r10 == 100.0 && "Nivel 1A fallo: IDSetRecall@10 debe ser 100.0000%!");
        assert(r100 == 100.0 && "Nivel 1A fallo: IDSetRecall@100 debe ser 100.0000%!");
        assert(rm10 == 100.0 && "Nivel 1B fallo: RankMatch@10 debe ser 100.0000%!");
        assert(rm100 == 100.0 && "Nivel 1B fallo: RankMatch@100 debe ser 100.0000%!");
        assert(max_orig_dist_err[i] == 0.0 && "Nivel 2 fallo: La distancia debe coincidir bit a bit en espacio original!");
    }

    uint64_t total_pruned = audit.pruned_by_partition + audit.pruned_by_radial_shell
                          + audit.pruned_by_pca_16d + audit.pruned_by_pca_32d;
    printf("\n[Nivel 3 - Auditoria Directa de Poda en 5B (Cero Tolerancia: LB_safe^2 <= d_orig^2)]\n");
    printf("  Elegibles auditados            : %llu\n", static_cast<unsigned long long>(audit.total_eligible_checked));
    printf("  Podados por etapa              : Particion=%llu | Radial=%llu | PCA_16D=%llu | PCA_32D=%llu (Total=%llu)\n",
           static_cast<unsigned long long>(audit.pruned_by_partition),
           static_cast<unsigned long long>(audit.pruned_by_radial_shell),
           static_cast<unsigned long long>(audit.pruned_by_pca_16d),
           static_cast<unsigned long long>(audit.pruned_by_pca_32d),
           static_cast<unsigned long long>(total_pruned));
    printf("  Violaciones estrictas (Delta>0): Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu\n",
           static_cast<unsigned long long>(audit.strict_violations_partition),
           static_cast<unsigned long long>(audit.strict_violations_radial),
           static_cast<unsigned long long>(audit.strict_violations_pca16),
           static_cast<unsigned long long>(audit.strict_violations_pca32));
    printf("  Eventos margen diag (0<D<=eps) : Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu\n",
           static_cast<unsigned long long>(audit.diag_margin_events_partition),
           static_cast<unsigned long long>(audit.diag_margin_events_radial),
           static_cast<unsigned long long>(audit.diag_margin_events_pca16),
           static_cast<unsigned long long>(audit.diag_margin_events_pca32));
    printf("  Violaciones > eps_diag         : Part=%llu | Radial=%llu | PCA16=%llu | PCA32=%llu\n",
           static_cast<unsigned long long>(audit.diag_exceed_violations_partition),
           static_cast<unsigned long long>(audit.diag_exceed_violations_radial),
           static_cast<unsigned long long>(audit.diag_exceed_violations_pca16),
           static_cast<unsigned long long>(audit.diag_exceed_violations_pca32));
    printf("  Falsos descartes Top-K         : %llu\n", static_cast<unsigned long long>(audit.false_prunings_topk));
    printf("  Max exceso con signo (LB^2-d^2): %.6e (requerido <= 0.000000e+00)\n", audit.max_signed_excess);
    printf("  Max exceso positivo            : %.6e (requerido == 0.000000e+00)\n", audit.max_positive_excess);
    printf("  Min ratio d_orig^2 / LB_safe^2 : %.6f (requerido >= 1.000000)\n", audit.min_ratio_orig_over_lb);

    assert(audit.strict_violations_partition == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda de particion!");
    assert(audit.strict_violations_radial == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda radial!");
    assert(audit.strict_violations_pca16 == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda PCA 16D!");
    assert(audit.strict_violations_pca32 == 0 && "Violacion estricta (LB_safe^2 > d_orig^2) en poda PCA 32D!");
    assert(audit.diag_margin_events_partition == 0 && "Evento en margen diagnostico en particion!");
    assert(audit.diag_margin_events_radial == 0 && "Evento en margen diagnostico en poda radial!");
    assert(audit.diag_margin_events_pca16 == 0 && "Evento en margen diagnostico en poda PCA 16D!");
    assert(audit.diag_margin_events_pca32 == 0 && "Evento en margen diagnostico en poda PCA 32D!");
    assert(audit.max_positive_excess == 0.0 && "Se registro exceso positivo (LB_safe^2 > d_orig^2)!");
    assert(audit.max_signed_excess <= 0.0 && "El maximo exceso con signo debe ser <= 0.0!");
    assert(audit.min_ratio_orig_over_lb >= 1.0 && "El ratio minimo d_orig^2 / LB_safe^2 debe ser >= 1.0!");
    assert(audit.false_prunings_topk == 0 && "Un vecino verdadero del Top-K fue podado!");
}

int main() {
    std::cout << "================================================================================\n";
    std::cout << "ALG-Flat 2.0 (Hito 2A.1): Certificacion Matematica sin Tolerancia Oculta\n";
    std::cout << "  - Cotas IEEE-754 derivadas formalmente con factores gamma_n de Higham\n";
    std::cout << "  - Nivel 1A: IDSetRecall con chequeo de no duplicados (0 tolerancia)\n";
    std::cout << "  - Nivel 1B: RankMatch posicion a posicion (orden total dist_sq ASC, id ASC)\n";
    std::cout << "  - Nivel 2 : Igualdad bit a bit de distancias en espacio original float32\n";
    std::cout << "  - Nivel 3 : Exceso con signo Delta = LB_safe^2 - d_orig^2 <= 0.0 (0 tolerancia)\n";
    std::cout << "================================================================================\n";

    printf("Constantes IEEE-754 derivadas (u = 2^-24 = %.3e):\n", IEEE754_F32_U);
    printf("  gamma_16 = %.3e | gamma_64 = %.3e | gamma_66 = %.3e\n",
           static_cast<double>(GAMMA_16_F32), static_cast<double>(GAMMA_64_F32), static_cast<double>(GAMMA_66_F32));
    printf("  ROT64_L2_COEFF_Q = %.3e | ROT64_L2_COEFF_C = %.3e | ROT64_L2_COEFF_X = %.3e\n",
           static_cast<double>(ROT64_L2_COEFF_Q),
           static_cast<double>(ROT64_L2_COEFF_C),
           static_cast<double>(ROT64_L2_COEFF_X));

    /* Suite 1: Dataset completo de benchmark (N=100,000, M50/50, sigma=0.09, P=128) */
    {
        BenchmarkDataset ds_full;
        generate_dataset_deterministic(8, 100000, 100, 64, 0.09f, MIX_50_50, ds_full);
        run_certification_suite("1_Full_Benchmark_100k_Mix_s009", ds_full, 128, false);
    }

    /* Suite 2: Centroides alejados del origen (||c_p|| >> 0, prueba de invariancia traslacional de colas) */
    {
        BenchmarkDataset ds_off;
        make_adversarial_off_origin_dataset(ds_off);
        run_certification_suite("2_Adversarial_FarOffOrigin_Centroids", ds_off, 64, false);
    }

    /* Suite 3: Colinealidad real respecto del centroide asignado c_p en colas 16..63 / 32..63 y empates */
    {
        BenchmarkDataset ds_col;
        make_adversarial_ties_and_base_dataset(ds_col);
        run_certification_suite("3_Adversarial_TrueCentroidCollinear_And_Ties", ds_col, 64, true);
    }

    /* Suite 4: Alta localidad con >92% de poda (N=100,000, R, sigma=0.03, P=128) */
    {
        BenchmarkDataset ds_tight;
        generate_dataset_deterministic(8, 100000, 100, 64, 0.03f, MIX_R, ds_tight);
        run_certification_suite("4_Full_Benchmark_100k_Tight_s003", ds_tight, 128, false);
    }

    std::cout << "\n================================================================================\n";
    std::cout << "CERTIFICACION HITO 2A.1 SUPERADA: 0 VIOLACIONES ESTRICTAS (Delta <= 0.0 EXACTO)\n";
    std::cout << "================================================================================\n";
    return 0;
}
