#ifndef ALG_FLAT_EXECUTION_PLANS_H
#define ALG_FLAT_EXECUTION_PLANS_H

#include "alg_flat_types.h"
#include "physical_layouts.h"

namespace alg_flat {

/*
 * Reporte de Auditoria Matematica de Poda (Hito 2A - Nivel 3)
 * Verifica para cada vector descartado que su cota inferior conservadora LB^2(q, x)
 * sea <= la distancia real en espacio original d_orig^2(q, x), y que ningun
 * vecino del Top-K verdadero sea descartado en ninguna etapa.
 */
struct PruningAuditReport {
    uint64_t total_eligible_checked;
    uint64_t pruned_by_partition;
    uint64_t pruned_by_radial_shell;
    uint64_t pruned_by_pca_16d;
    uint64_t pruned_by_pca_32d;
    uint64_t evaluated_full_64d;

    uint64_t bound_violations_partition;
    uint64_t bound_violations_radial;
    uint64_t bound_violations_pca16;
    uint64_t bound_violations_pca32;
    uint64_t false_prunings_topk;

    double min_ratio_orig_over_lb; /* Debe ser >= 1.000000 en todo vector podado */
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
 * (Particion + Cascara Radial + Base Ortogonal PCA con ||q_tail - c_{p,tail}|| vs ||x_tail - c_{p,tail}|| en 16D/32D)
 */
void execute_plan5b_ivf_full_exact_prune_avx2(const IVFPartitionedIndex& idx,
                                              const float* q,
                                              const QueryPredicate& pred,
                                              uint32_t k,
                                              Neighbor* out_results,
                                              uint32_t* out_count,
                                              QueryExecutionMetrics& metrics);

/*
 * Auditoria de Exactitud Nivel 3: Verifica matematicamente cada descarte de 5B contra la distancia original exacta
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
