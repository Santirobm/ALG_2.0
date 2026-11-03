#ifndef ALG_FLAT_EXECUTION_PLANS_H
#define ALG_FLAT_EXECUTION_PLANS_H

#include "alg_flat_types.h"
#include "physical_layouts.h"
#include "ieee754_bounds.h"

namespace alg_flat {

/*
 * Reporte de Auditoria Matematica de Poda sin Tolerancia Oculta (Hito 2A.2 - Nivel 3)
 * Para cada vector elegible descartado x en cualquiera de las 5 etapas (Particion, Radial,
 * PCA 16D, PCA 32D y PCA 64D), mide el exceso firmado exacto:
 *   Delta(q, x) = LB_safe^2(q, x) - d_orig^2(q, x)
 * y separa explicitamente:
 *   1. Violaciones estrictas (Delta > 0.0, tolerancia CERO).
 *   2. Excesos numericos dentro del margen de diagnostico (0.0 < Delta <= eps_diag).
 *   3. Violaciones que superan el umbral de diagnostico (Delta > eps_diag).
 */
struct PruningAuditReport {
    uint64_t total_eligible_checked;
    uint64_t pruned_by_partition;
    uint64_t pruned_by_radial_shell;
    uint64_t pruned_by_pca_16d;
    uint64_t pruned_by_pca_32d;
    uint64_t pruned_by_pca_64d;
    uint64_t evaluated_full_64d;

    /* Categoria 1: Violaciones estrictas (Delta = LB_safe^2 - d_orig^2 > 0.0, sin tolerancia) */
    uint64_t strict_violations_partition;
    uint64_t strict_violations_radial;
    uint64_t strict_violations_pca16;
    uint64_t strict_violations_pca32;
    uint64_t strict_violations_pca64;

    /* Categoria 2: Excesos dentro del margen de diagnostico del auditor (0.0 < Delta <= eps_diag) */
    uint64_t diag_margin_events_partition;
    uint64_t diag_margin_events_radial;
    uint64_t diag_margin_events_pca16;
    uint64_t diag_margin_events_pca32;
    uint64_t diag_margin_events_pca64;

    /* Categoria 3: Violaciones superiores al umbral de diagnostico (Delta > eps_diag) */
    uint64_t diag_exceed_violations_partition;
    uint64_t diag_exceed_violations_radial;
    uint64_t diag_exceed_violations_pca16;
    uint64_t diag_exceed_violations_pca32;
    uint64_t diag_exceed_violations_pca64;

    uint64_t false_prunings_topk;

    double max_signed_excess;       /* max_{podados} (LB_safe^2 - d_orig^2); debe ser <= 0.0 */
    double max_positive_excess;     /* max(0.0, max_signed_excess); debe ser == 0.0 */
    double min_ratio_orig_over_lb;  /* min_{podados} (d_orig^2 / LB_safe^2); debe ser >= 1.000000 */
    uint64_t contract_fallbacks;    /* Consultas que cayeron a ExactScan por violacion de MXCSR o dominio */
};

/*
 * Plan 0: Referencia exacta del ExactScan de Hito 8/9
 * (std::priority_queue + 1 acumulador AVX2 + _mm_hadd_ps sobre eligible_ids)
 */
void execute_plan0_ref_hito9(const FlatAoSIndex& idx,
                             const float* q,
                             const QueryPredicate& pred,
                             uint32_t k,
                             Neighbor* out_results,
                             uint32_t* out_count,
                             QueryExecutionMetrics& metrics);

/*
 * Plan 1: Flat Scalar AoS
 * (Distancia L2 escalar estricta sin SIMD + TopKHeap fijo sin malloc)
 */
void execute_plan1_flat_scalar_aos(const FlatAoSIndex& idx,
                                   const float* q,
                                   const QueryPredicate& pred,
                                   uint32_t k,
                                   Neighbor* out_results,
                                   uint32_t* out_count,
                                   QueryExecutionMetrics& metrics);

/*
 * Plan 2A: Flat AVX2 AoS (1x1)
 * (Doble acumulador FMA + reduccion horizontal sin hadd + TopKHeap fijo)
 */
void execute_plan2a_flat_avx2_aos_1x(const FlatAoSIndex& idx,
                                     const float* q,
                                     const QueryPredicate& pred,
                                     uint32_t k,
                                     Neighbor* out_results,
                                     uint32_t* out_count,
                                     QueryExecutionMetrics& metrics);

/*
 * Plan 2B: Flat AVX2 AoS (4x1 Multi-Vector)
 * (Procesa 4 vectores elegibles en paralelo reutilizando registros de q + filtro SIMD de umbral)
 */
void execute_plan2b_flat_avx2_aos_4x(const FlatAoSIndex& idx,
                                     const float* q,
                                     const QueryPredicate& pred,
                                     uint32_t k,
                                     Neighbor* out_results,
                                     uint32_t* out_count,
                                     QueryExecutionMetrics& metrics);

/*
 * Plan 3A: Flat AVX2 SoA (Structure of Arrays global, 8 vectores simultaneos por dimension)
 */
void execute_plan3a_flat_avx2_soa(const FlatSoAIndex& idx,
                                  const float* q,
                                  const QueryPredicate& pred,
                                  uint32_t k,
                                  Neighbor* out_results,
                                  uint32_t* out_count,
                                  QueryExecutionMetrics& metrics);

/*
 * Plan 3B: Flat AVX2 AoSoA (Bloques transpuestos de 8 vectores x 64 dims = 2048B contiguos)
 */
void execute_plan3b_flat_avx2_aosoa(const FlatAoSoAIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    Neighbor* out_results,
                                    uint32_t* out_count,
                                    QueryExecutionMetrics& metrics);

/*
 * Plan 4: IVF Flat AVX2 (Particionamiento fisico K-Means contiguo en espacio original, SIN poda geometrica)
 */
void execute_plan4_ivf_flat_avx2(const IVFPartitionedIndex& idx,
                                 const float* q,
                                 const QueryPredicate& pred,
                                 uint32_t k,
                                 Neighbor* out_results,
                                 uint32_t* out_count,
                                 QueryExecutionMetrics& metrics);

/*
 * Plan 5A: IVF + Poda Exacta a Nivel de Particion en espacio original
 * (Desigualdad triangular conservadora d_low(q,c) - r_max(p) > rho_safe)
 */
void execute_plan5a_ivf_part_prune_avx2(const IVFPartitionedIndex& idx,
                                        const float* q,
                                        const QueryPredicate& pred,
                                        uint32_t k,
                                        Neighbor* out_results,
                                        uint32_t* out_count,
                                        QueryExecutionMetrics& metrics);

/*
 * Plan 5B: IVF + Poda Exacta Multi-Nivel con Cota PCA Centrada por Particion y Distancia Final en Espacio Original
 * (Particion + Cascara Radial + Base Ortogonal PCA en 16D/32D/64D con redondeo dirigido conservador)
 */
void execute_plan5b_ivf_full_exact_prune_avx2(const IVFPartitionedIndex& idx,
                                              const float* q,
                                              const QueryPredicate& pred,
                                              uint32_t k,
                                              Neighbor* out_results,
                                              uint32_t* out_count,
                                              QueryExecutionMetrics& metrics);

/*
 * Auditoria de Exactitud Nivel 3 (Hito 2A.2): Verifica sin tolerancia oculta cada descarte
 * de las 5 etapas de 5B (Particion, Radial, PCA 16D, PCA 32D y PCA 64D).
 */
void audit_plan5b_pruning_soundness(const IVFPartitionedIndex& idx,
                                    const float* q,
                                    const QueryPredicate& pred,
                                    uint32_t k,
                                    const Neighbor* true_topk,
                                    uint32_t true_topk_count,
                                    PruningAuditReport& report);

} // namespace alg_flat

#endif // ALG_FLAT_EXECUTION_PLANS_H
