#include "alg_search.h"

AlgStatus alg_search_compact(
    const AlgIndex* index,
    const float* query_vector,
    const uint32_t* eligible_ids,
    uint32_t eligible_count,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count
) {
    if (!index || !index->vector_data) {
        return ALG_NOT_INITIALIZED;
    }
    if (!query_vector || !out_results || !out_count || top_k == 0) {
        return ALG_INVALID_ARGUMENT;
    }
    if (search_mode != ALG_SEARCH_FAST_FLOAT32) {
        return ALG_INVALID_ARGUMENT;
    }

    if (eligible_count == 0) {
        *out_count = 0;
        return ALG_OK;
    }

    uint32_t dim = index->config.dimension;
    const float* corpus = index->vector_data;
    uint32_t total_vectors = index->num_vectors;
    AlgKernelDispatchTable dispatch = index->dispatch;

    AlgTopKHeap heap;
    alg_heap_init(&heap, top_k);

    if (eligible_ids != NULL) {
        // Filtered search over sparse candidate ID list
        uint32_t i = 0;
        for (; i + 4 <= eligible_count; i += 4) {
            uint32_t id0 = eligible_ids[i + 0];
            uint32_t id1 = eligible_ids[i + 1];
            uint32_t id2 = eligible_ids[i + 2];
            uint32_t id3 = eligible_ids[i + 3];

            // Safety check against out of bounds IDs
            if (id0 >= total_vectors || id1 >= total_vectors ||
                id2 >= total_vectors || id3 >= total_vectors) {
                alg_heap_free(&heap);
                return ALG_INVALID_ARGUMENT;
            }

            const float* v0 = corpus + (size_t)id0 * dim;
            const float* v1 = corpus + (size_t)id1 * dim;
            const float* v2 = corpus + (size_t)id2 * dim;
            const float* v3 = corpus + (size_t)id3 * dim;

            float dists[4];
            dispatch.batch4(query_vector, v0, v1, v2, v3, dim, dists);

            if (heap.size < top_k || dists[0] <= heap.threshold) alg_heap_push(&heap, id0, dists[0]);
            if (heap.size < top_k || dists[1] <= heap.threshold) alg_heap_push(&heap, id1, dists[1]);
            if (heap.size < top_k || dists[2] <= heap.threshold) alg_heap_push(&heap, id2, dists[2]);
            if (heap.size < top_k || dists[3] <= heap.threshold) alg_heap_push(&heap, id3, dists[3]);
        }
        for (; i < eligible_count; ++i) {
            uint32_t id = eligible_ids[i];
            if (id >= total_vectors) {
                alg_heap_free(&heap);
                return ALG_INVALID_ARGUMENT;
            }
            const float* v = corpus + (size_t)id * dim;
            float dist = dispatch.single(query_vector, v, dim);
            if (heap.size < top_k || dist <= heap.threshold) {
                alg_heap_push(&heap, id, dist);
            }
        }
    } else {
        // Sequential search over entire corpus [0 .. eligible_count - 1]
        uint32_t limit = (eligible_count < total_vectors) ? eligible_count : total_vectors;
        uint32_t i = 0;
        for (; i + 4 <= limit; i += 4) {
            const float* v0 = corpus + (size_t)(i + 0) * dim;
            const float* v1 = corpus + (size_t)(i + 1) * dim;
            const float* v2 = corpus + (size_t)(i + 2) * dim;
            const float* v3 = corpus + (size_t)(i + 3) * dim;

            float dists[4];
            dispatch.batch4(query_vector, v0, v1, v2, v3, dim, dists);

            if (heap.size < top_k || dists[0] <= heap.threshold) alg_heap_push(&heap, i + 0, dists[0]);
            if (heap.size < top_k || dists[1] <= heap.threshold) alg_heap_push(&heap, i + 1, dists[1]);
            if (heap.size < top_k || dists[2] <= heap.threshold) alg_heap_push(&heap, i + 2, dists[2]);
            if (heap.size < top_k || dists[3] <= heap.threshold) alg_heap_push(&heap, i + 3, dists[3]);
        }
        for (; i < limit; ++i) {
            const float* v = corpus + (size_t)i * dim;
            float dist = dispatch.single(query_vector, v, dim);
            if (heap.size < top_k || dist <= heap.threshold) {
                alg_heap_push(&heap, i, dist);
            }
        }
    }

    *out_count = heap.size;
    alg_heap_sort_ascending(&heap, out_results);
    alg_heap_free(&heap);
    return ALG_OK;
}

AlgStatus alg_search(
    const AlgIndex* index,
    const float* query_vector,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count
) {
    if (!index) return ALG_NOT_INITIALIZED;
    return alg_search_compact(
        index,
        query_vector,
        NULL,
        index->num_vectors,
        top_k,
        search_mode,
        out_results,
        out_count
    );
}

AlgStatus alg_search_buffer(
    const float* query_vector,
    const float* contiguous_vectors,
    const uint32_t* candidate_ids,
    uint32_t candidate_count,
    uint32_t dimension,
    uint32_t top_k,
    AlgSearchMode search_mode,
    AlgSearchResult* out_results,
    uint32_t* out_count
) {
    if (!query_vector || !contiguous_vectors || !out_results || !out_count || top_k == 0 || dimension == 0) {
        return ALG_INVALID_ARGUMENT;
    }
    if (search_mode != ALG_SEARCH_FAST_FLOAT32) {
        return ALG_INVALID_ARGUMENT;
    }

    if (candidate_count == 0) {
        *out_count = 0;
        return ALG_OK;
    }

    AlgKernelDispatchTable dispatch = alg_get_kernel_dispatch();
    AlgTopKHeap heap;
    alg_heap_init(&heap, top_k);

    uint32_t i = 0;
    for (; i + 4 <= candidate_count; i += 4) {
        const float* v0 = contiguous_vectors + (size_t)(i + 0) * dimension;
        const float* v1 = contiguous_vectors + (size_t)(i + 1) * dimension;
        const float* v2 = contiguous_vectors + (size_t)(i + 2) * dimension;
        const float* v3 = contiguous_vectors + (size_t)(i + 3) * dimension;

        uint32_t id0 = candidate_ids ? candidate_ids[i + 0] : (i + 0);
        uint32_t id1 = candidate_ids ? candidate_ids[i + 1] : (i + 1);
        uint32_t id2 = candidate_ids ? candidate_ids[i + 2] : (i + 2);
        uint32_t id3 = candidate_ids ? candidate_ids[i + 3] : (i + 3);

        float dists[4];
        dispatch.batch4(query_vector, v0, v1, v2, v3, dimension, dists);

        if (heap.size < top_k || dists[0] <= heap.threshold) alg_heap_push(&heap, id0, dists[0]);
        if (heap.size < top_k || dists[1] <= heap.threshold) alg_heap_push(&heap, id1, dists[1]);
        if (heap.size < top_k || dists[2] <= heap.threshold) alg_heap_push(&heap, id2, dists[2]);
        if (heap.size < top_k || dists[3] <= heap.threshold) alg_heap_push(&heap, id3, dists[3]);
    }
    for (; i < candidate_count; ++i) {
        const float* v = contiguous_vectors + (size_t)i * dimension;
        uint32_t id = candidate_ids ? candidate_ids[i] : i;
        float dist = dispatch.single(query_vector, v, dimension);
        if (heap.size < top_k || dist <= heap.threshold) {
            alg_heap_push(&heap, id, dist);
        }
    }

    *out_count = heap.size;
    alg_heap_sort_ascending(&heap, out_results);
    alg_heap_free(&heap);
    return ALG_OK;
}
