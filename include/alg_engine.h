/**
 * @file alg_engine.h
 * @brief ALG Vector Engine - High Performance Filtered Vector Search Library
 * 
 * Standalone C ABI for exact filtered vector search with dynamic SIMD dispatch
 * (AVX2/FMA and portable scalar fallback).
 *
 * NUMERICAL CONTRACT & RANKING SPECIFICATION:
 * -------------------------------------------
 * 1. Mode ALG_SEARCH_FAST_FLOAT32:
 *    - Computes L2 squared distances using IEEE 754 float32 precision.
 *    - Evaluation kernel utilizes 4-way unrolled AVX2+FMA vector accumulators (when supported)
 *      or portable scalar accumulators.
 *    - Accumulation order is deterministic and fixed per dimension.
 *    - Ranking order is strictly deterministic:
 *        a) Primary sorting key: float32 squared distance (ascending).
 *        b) Tie-breaker key: vector ID (ascending) for identical distances (0 ULP difference).
 *    - Accuracy bounds: Due to floating-point non-associativity across vector accumulation
 *      lanes, float32 rankings are not guaranteed to be identical to binary64 (double precision)
 *      ground truth across arbitrary distributions without an explicit error bound pass.
 *      Exhaustive audit on real-world SimpleWiki MiniLM-L6-v2 embeddings (D in {64, 128, 256})
 *      confirmed 100.0% IDSetRecall across 9,000 queries.
 *    - Finiteness policy: Non-finite inputs (NaN, +Inf, -Inf) are strictly prohibited in
 *      corpus vectors and query vectors. Any non-finite value is explicitly rejected with
 *      ALG_INVALID_ARGUMENT, leaving existing index state completely unmodified.
 *    - Error guarantee: On any error, *out_count is guaranteed to be set to 0 and no partial
 *      results are presented as valid.
 *
 * 2. Mode ALG_SEARCH_EXACT_TOPK:
 *    - Reserved for future multi-pass uncertainty-band refinement.
 *    - In current release, returns ALG_INVALID_ARGUMENT.
 *
 * THREAD SAFETY CONTRACT:
 * -----------------------
 * - Read-only queries (alg_search, alg_search_compact, alg_search_buffer) are completely
 *   thread-safe when executed concurrently by multiple threads on the same AlgIndex instance.
 * - Index lifecycle functions (alg_set_vectors, alg_free_index) mutate internal memory and
 *   must NOT be called concurrently with searches or other mutations on the same instance.
 *
 * DUPLICATE CANDIDATE IDS POLICY:
 * -------------------------------
 * - In alg_search_compact and alg_search_buffer, candidate ID lists are processed sequentially
 *   as supplied. If eligible_ids contains duplicate IDs, each candidate instance is evaluated
 *   independently and may appear multiple times in the Top-K results. Consumers desiring
 *   deduplication should ensure the candidate ID array contains unique identifiers.
 */

#ifndef ALG_ENGINE_H
#define ALG_ENGINE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
    #if defined(ALG_BUILD_DLL)
        #define ALG_API __declspec(dllexport)
    #elif defined(ALG_USE_DLL)
        #define ALG_API __declspec(dllimport)
    #else
        #define ALG_API
    #endif
#else
    #if defined(ALG_BUILD_DLL)
        #define ALG_API __attribute__((visibility("default")))
    #else
        #define ALG_API
    #endif
#endif

/** Maximum supported vector dimension */
#define ALG_MAX_DIMENSION 1048576u

/**
 * Status and error return codes.
 */
typedef enum AlgStatus {
    ALG_OK                 = 0,  /**< Operation completed successfully */
    ALG_INVALID_ARGUMENT   = 1,  /**< An invalid argument was provided (e.g. NULL pointer, dim=0, NaN/Inf, invalid mode) */
    ALG_UNSUPPORTED_ISA    = 2,  /**< Requested ISA feature is unsupported by host CPU */
    ALG_OUT_OF_MEMORY      = 3,  /**< Memory allocation failed (or size calculation overflowed) */
    ALG_NOT_INITIALIZED    = 4,  /**< Index or component is not initialized */
    ALG_CAPACITY_EXCEEDED  = 5   /**< Target exceeds configured capacity */
} AlgStatus;

/**
 * Supported Distance Metrics.
 */
typedef enum AlgMetric {
    ALG_METRIC_L2_SQR = 0        /**< Euclidean squared distance: sum((a[i] - b[i])^2) */
} AlgMetric;

/**
 * Search Modes.
 */
typedef enum AlgSearchMode {
    ALG_SEARCH_FAST_FLOAT32 = 0, /**< High-throughput float32 SIMD with deterministic ID tie-breaking */
    ALG_SEARCH_EXACT_TOPK   = 1  /**< Reserved for future uncertainty-band refinement */
} AlgSearchMode;

/**
 * CPU Feature Flags detected at runtime via CPUID.
 */
typedef enum AlgCpuFeature {
    ALG_CPU_FEATURE_NONE  = 0,
    ALG_CPU_FEATURE_AVX2  = (1 << 0),
    ALG_CPU_FEATURE_FMA   = (1 << 1)
} AlgCpuFeature;

/**
 * Single neighbor result item.
 */
typedef struct AlgSearchResult {
    uint32_t id;         /**< Vector ID */
    float distance;      /**< Squared L2 distance */
} AlgSearchResult;

/**
 * Configuration for index initialization.
 */
typedef struct AlgIndexConfig {
    uint32_t dimension;        /**< Vector dimension (1 <= dimension <= ALG_MAX_DIMENSION) */
    uint32_t initial_capacity; /**< Pre-allocated vector count capacity (0 for default auto-grow) */
    AlgMetric metric;          /**< Distance metric (must be ALG_METRIC_L2_SQR) */
} AlgIndexConfig;

/**
 * Opaque handle representing an index instance.
 */
typedef struct AlgIndex AlgIndex;

/**
 * Create a new vector search index.
 * 
 * @param config Pointer to configuration struct.
 * @param out_index Pointer to receive the allocated index handle.
 * @return ALG_OK on success, or error code on failure.
 */
ALG_API AlgStatus alg_create_index(const AlgIndexConfig* config, AlgIndex** out_index);

/**
 * Destroy an index and release all associated resources.
 * 
 * @param index Index handle (safe to pass NULL).
 */
ALG_API void alg_free_index(AlgIndex* index);

/**
 * Ingest or replace the full contiguous vector corpus.
 * Vectors are copied into 64-byte aligned internal memory.
 * Vector IDs are implicitly assigned sequentially in range [0, num_vectors - 1].
 *
 * Atomicity guarantee: If validation fails (e.g. non-finite values or allocation error),
 * the existing index state and corpus remain completely intact and unmodified.
 *
 * @param index Pointer to index handle.
 * @param num_vectors Number of vectors in the corpus.
 * @param vectors Contiguous row-major float32 array of shape (num_vectors, dimension).
 * @return ALG_OK on success, or error code.
 */
ALG_API AlgStatus alg_set_vectors(AlgIndex* index, uint32_t num_vectors, const float* vectors);

/**
 * Query the index over a filtered list of candidate vector IDs.
 *
 * Robustness guarantee: If out_count is non-NULL, *out_count is set to 0 immediately
 * upon entry and remains 0 on any error.
 *
 * @param index Pointer to index handle.
 * @param query_vector Query vector of size 'dimension' (must contain finite values).
 * @param eligible_ids Required array of eligible candidate vector IDs (must NOT be NULL).
 * @param eligible_count Number of candidate IDs in eligible_ids.
 * @param top_k Maximum number of nearest neighbors to retrieve.
 * @param search_mode Search evaluation mode (must be ALG_SEARCH_FAST_FLOAT32).
 * @param out_results User-allocated buffer of at least top_k AlgSearchResult items.
 * @param out_count Receives the actual number of results written (min(top_k, eligible_count)).
 * @return ALG_OK on success, or error code.
 */
ALG_API AlgStatus alg_search_compact(
    const AlgIndex* index,
    const float* query_vector,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count);

/**
 * Query the entire corpus without filtering (100% selectivity).
 *
 * Robustness guarantee: If out_count is non-NULL, *out_count is set to 0 immediately
 * upon entry and remains 0 on any error.
 *
 * @param index Pointer to index handle.
 * @param query_vector Query vector of size 'dimension' (must contain finite values).
 * @param top_k Maximum number of nearest neighbors to retrieve.
 * @param search_mode Search evaluation mode (must be ALG_SEARCH_FAST_FLOAT32).
 * @param out_results User-allocated buffer of at least top_k AlgSearchResult items.
 * @param out_count Receives the actual number of results written (min(top_k, num_vectors)).
 * @return ALG_OK on success, or error code.
 */
ALG_API AlgStatus alg_search(
    const AlgIndex* index,
    const float* query_vector,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count);

/**
 * Directly compute Top-K nearest neighbors over an external contiguous buffer of vectors.
 * Useful for scenarios where candidates have already been gathered into a contiguous memory block.
 *
 * Robustness guarantee: If out_count is non-NULL, *out_count is set to 0 immediately
 * upon entry and remains 0 on any error.
 *
 * @param query_vector Query vector of size 'dimension' (must contain finite values).
 * @param contiguous_vectors Contiguous row-major float32 array of shape (candidate_count, dimension).
 * @param candidate_ids Optional array of original vector IDs (length candidate_count).
 *                      If NULL, local IDs 0 .. candidate_count - 1 are used.
 * @param candidate_count Number of vectors in contiguous_vectors.
 * @param dimension Vector dimension.
 * @param top_k Maximum number of nearest neighbors to retrieve.
 * @param search_mode Search evaluation mode (must be ALG_SEARCH_FAST_FLOAT32).
 * @param out_results User-allocated buffer of at least top_k AlgSearchResult items.
 * @param out_count Receives the actual number of results written.
 * @return ALG_OK on success, or error code.
 */
ALG_API AlgStatus alg_search_buffer(
    const float* query_vector,
    const float* contiguous_vectors,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t dimension,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count);

/**
 * Compute raw L2 squared distance between query and a single vector.
 */
ALG_API float alg_distance_l2_sqr(const float* a, const float* b, uint32_t dimension);

/**
 * Return human-readable string for status code.
 */
ALG_API const char* alg_status_to_string(AlgStatus status);

/**
 * Return bitmask of CPU features detected on current host (see AlgCpuFeature).
 */
ALG_API uint32_t alg_get_supported_isa(void);

/**
 * Return name of the active SIMD instruction set ("AVX2+FMA" or "Scalar Fallback").
 */
ALG_API const char* alg_get_active_isa_name(void);

/**
 * Return library version string.
 */
ALG_API const char* alg_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* ALG_ENGINE_H */
