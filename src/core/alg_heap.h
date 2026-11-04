#ifndef ALG_CORE_ALG_HEAP_H
#define ALG_CORE_ALG_HEAP_H

#include "../../include/alg_engine.h"
#include <float.h>
#include <stdlib.h>
#include <string.h>

#define ALG_HEAP_INLINE_CAPACITY 512

/**
 * Deterministic comparison:
 * A neighbor 'a' is strictly better (closer) than 'b' if:
 * 1. a.distance < b.distance
 * 2. or (a.distance == b.distance && a.id < b.id)
 */
static inline bool alg_neighbor_is_better(const AlgSearchResult* a, const AlgSearchResult* b) {
    if (a->distance != b->distance) {
        return a->distance < b->distance;
    }
    return a->id < b->id;
}

/**
 * Bounded Max-Heap for Top-K tracking.
 * data[0] is the worst element in the current Top-K set.
 */
typedef struct AlgTopKHeap {
    AlgSearchResult inline_buffer[ALG_HEAP_INLINE_CAPACITY];
    AlgSearchResult* data;
    uint32_t size;
    uint32_t k;
    float threshold;
    bool is_dynamic;
} AlgTopKHeap;

static inline void alg_heap_init(AlgTopKHeap* heap, uint32_t target_k) {
    heap->size = 0;
    heap->k = target_k;
    heap->threshold = FLT_MAX;
    heap->is_dynamic = false;

    if (target_k <= ALG_HEAP_INLINE_CAPACITY) {
        heap->data = heap->inline_buffer;
    } else {
        heap->data = (AlgSearchResult*)malloc(target_k * sizeof(AlgSearchResult));
        heap->is_dynamic = true;
    }
}

static inline void alg_heap_free(AlgTopKHeap* heap) {
    if (heap->is_dynamic && heap->data) {
        free(heap->data);
        heap->data = NULL;
    }
}

static inline void alg_heap_push(AlgTopKHeap* heap, uint32_t id, float dist) {
    AlgSearchResult cand;
    cand.id = id;
    cand.distance = dist;

    if (heap->size < heap->k) {
        uint32_t i = heap->size++;
        while (i > 0) {
            uint32_t parent = (i - 1) >> 1;
            // Max-heap: parent should be worse (larger dist/id) than child
            if (!alg_neighbor_is_better(&heap->data[parent], &cand)) {
                break;
            }
            heap->data[i] = heap->data[parent];
            i = parent;
        }
        heap->data[i] = cand;
        if (heap->size == heap->k) {
            heap->threshold = heap->data[0].distance;
        }
        return;
    }

    // Full heap: candidate must be strictly better than the worst element (root data[0])
    if (!alg_neighbor_is_better(&cand, &heap->data[0])) {
        return;
    }

    // Replace root and sift down
    uint32_t i = 0;
    uint32_t half = heap->k >> 1;
    while (i < half) {
        uint32_t left = (i << 1) + 1;
        uint32_t right = left + 1;
        uint32_t worst_child = left;
        if (right < heap->k && alg_neighbor_is_better(&heap->data[left], &heap->data[right])) {
            worst_child = right;
        }
        if (!alg_neighbor_is_better(&cand, &heap->data[worst_child])) {
            break;
        }
        heap->data[i] = heap->data[worst_child];
        i = worst_child;
    }
    heap->data[i] = cand;
    heap->threshold = heap->data[0].distance;
}

/**
 * Extract heap elements in ascending sorted order (best/closest first).
 */
static inline void alg_heap_sort_ascending(AlgTopKHeap* heap, AlgSearchResult* out_results) {
    uint32_t count = heap->size;
    for (uint32_t i = 0; i < count; ++i) {
        out_results[count - 1 - i] = heap->data[0];
        AlgSearchResult last = heap->data[--heap->size];
        if (heap->size == 0) break;

        uint32_t cur = 0;
        uint32_t half = heap->size >> 1;
        while (cur < half) {
            uint32_t left = (cur << 1) + 1;
            uint32_t right = left + 1;
            uint32_t worst = left;
            if (right < heap->size && alg_neighbor_is_better(&heap->data[left], &heap->data[right])) {
                worst = right;
            }
            if (!alg_neighbor_is_better(&last, &heap->data[worst])) {
                break;
            }
            heap->data[cur] = heap->data[worst];
            cur = worst;
        }
        heap->data[cur] = last;
    }
}

#endif /* ALG_CORE_ALG_HEAP_H */
