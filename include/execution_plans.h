#ifndef ALG_FLAT_EXECUTION_PLANS_H
#define ALG_FLAT_EXECUTION_PLANS_H

#include "alg_flat_types.h"
#include "physical_layouts.h"

namespace alg_flat {

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
 * Plan 4: IVF Flat AVX2 (Particionamiento fisico K-Means contiguo, SIN poda geometrica)
 */
void execute_plan4_ivf_flat_avx2(const IVFPartitionedIndex& idx,
                                 const float* q,
                                 const QueryPredicate& pred,
                                 uint32_t k,
                                 Neighbor* out_results,
                                 uint32_t* out_count,
                                 QueryExecutionMetrics& metrics);

/*
 * Plan 5A: IVF + Poda Exacta a Nivel de Particion (Desigualdad triangular d(q,c) - r_max >= rho_K)
 */
void execute_plan5a_ivf_part_prune_avx2(const IVFPartitionedIndex& idx,
                                        const float* q,
                                        const QueryPredicate& pred,
                                        uint32_t k,
                                        Neighbor* out_results,
                                        uint32_t* out_count,
                                        QueryExecutionMetrics& metrics);

/*
 * Plan 5B: IVF + Poda Exacta Multi-Nivel (Particion + Cascara Radial + Cota Cauchy-Schwarz 2D + Strip 16D)
 */
void execute_plan5b_ivf_full_exact_prune_avx2(const IVFPartitionedIndex& idx,
                                              const float* q,
                                              const QueryPredicate& pred,
                                              uint32_t k,
                                              Neighbor* out_results,
                                              uint32_t* out_count,
                                              QueryExecutionMetrics& metrics);

} // namespace alg_flat

#endif // ALG_FLAT_EXECUTION_PLANS_H
