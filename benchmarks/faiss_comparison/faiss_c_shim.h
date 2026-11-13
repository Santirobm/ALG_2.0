#pragma once

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Opaque handles for FAISS C Shim Interface.
 * Encapsulates all FAISS objects behind an ABI-stable, exception-free C API.
 */
typedef struct FaissFlatContext* FaissFlatHandle;
typedef struct FaissIVFContext*  FaissIVFHandle;
typedef struct FaissSelectorContext* FaissSelectorHandle;
typedef struct FaissParamsContext*   FaissParamsHandle;

/* Lifecycle: Flat Index */
FaissFlatHandle faiss_shim_create_flat_l2(uint32_t dim);
void faiss_shim_free_flat(FaissFlatHandle handle);
bool faiss_shim_flat_add(FaissFlatHandle handle, uint32_t n, const float* x);

/* Lifecycle: IVF Index */
FaissIVFHandle faiss_shim_create_ivf_flat(uint32_t dim, uint32_t nlist);
void faiss_shim_free_ivf(FaissIVFHandle handle);
bool faiss_shim_ivf_train_and_add(FaissIVFHandle handle, uint32_t n, const float* x);

/* Filter & Parameter Management */
FaissSelectorHandle faiss_shim_create_bitmap_selector(uint32_t n_bits, const uint8_t* bitmap);
void faiss_shim_free_selector(FaissSelectorHandle handle);

FaissParamsHandle faiss_shim_create_flat_params(FaissSelectorHandle sel);
FaissParamsHandle faiss_shim_create_ivf_params(FaissSelectorHandle sel, uint32_t nprobe);
void faiss_shim_params_set_selector(FaissParamsHandle params, FaissSelectorHandle sel);
void faiss_shim_params_set_nprobe(FaissParamsHandle params, uint32_t nprobe);
void faiss_shim_free_params(FaissParamsHandle handle);

/* Critical Path: Pure C search functions (zero allocation, zero exceptions) */
int faiss_shim_search_flat(
    FaissFlatHandle handle,
    uint32_t n_queries,
    const float* queries,
    uint32_t k,
    float* out_distances,
    int64_t* out_labels,
    FaissParamsHandle params);

void faiss_shim_compute_distance_subset(
    FaissFlatHandle handle,
    uint32_t n_queries,
    const float* queries,
    uint32_t k_subset,
    float* out_distances,
    const int64_t* subset_labels);

int faiss_shim_search_ivf(
    FaissIVFHandle handle,
    uint32_t n_queries,
    const float* queries,
    uint32_t k,
    float* out_distances,
    int64_t* out_labels,
    FaissParamsHandle params);

#ifdef __cplusplus
}
#endif
