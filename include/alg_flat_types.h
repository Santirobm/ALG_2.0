#ifndef ALG_FLAT_TYPES_H
#define ALG_FLAT_TYPES_H

#include <stdint.h>
#include <stddef.h>
#include <float.h>
#include <math.h>
#include <malloc.h>
#include <string.h>
#include <algorithm>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

namespace alg_flat {

static constexpr uint32_t MAX_TOPK_CAPACITY = 512;
static constexpr size_t CACHE_LINE_BYTES = 64;

inline void* aligned_alloc_64(size_t bytes) {
    if (bytes == 0) bytes = 64;
    size_t padded = (bytes + 63) & ~size_t(63);
#ifdef _WIN32
    void* ptr = _aligned_malloc(padded, CACHE_LINE_BYTES);
#else
    void* ptr = aligned_alloc(CACHE_LINE_BYTES, padded);
#endif
    if (ptr) memset(ptr, 0, padded);
    return ptr;
}

inline void aligned_free_64(void* ptr) {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    free(ptr);
#endif
}

struct Neighbor {
    uint32_t id;
    float dist_sq;
};

/* Comparador determinista: menor distancia primero; en empate, menor ID primero */
inline bool neighbor_less(const Neighbor& a, const Neighbor& b) {
    if (a.dist_sq != b.dist_sq) return a.dist_sq < b.dist_sq;
    return a.id < b.id;
}

/*
 * Bounded Max-Heap sin asignaciones dinamicas en tiempo de consulta.
 * Mantiene `threshold_sq` directamente accesible para rechazo inmediato O(1).
 */
struct TopKHeap {
    Neighbor data[MAX_TOPK_CAPACITY];
    uint32_t size;
    uint32_t k;
    float threshold_sq;

    inline void init(uint32_t target_k) {
        size = 0;
        k = (target_k <= MAX_TOPK_CAPACITY) ? target_k : MAX_TOPK_CAPACITY;
        threshold_sq = FLT_MAX;
    }

    inline void push(uint32_t id, float dist_sq) {
        if (size < k) {
            uint32_t i = size++;
            while (i > 0) {
                uint32_t p = (i - 1) >> 1;
                Neighbor cand{id, dist_sq};
                if (!neighbor_less( data[p], cand )) break;
                data[i] = data[p];
                i = p;
            }
            data[i] = {id, dist_sq};
            if (size == k) {
                threshold_sq = data[0].dist_sq;
            }
            return;
        }

        /* Solo entra si es estrictamente mejor que la raiz del Max-Heap */
        Neighbor cand{id, dist_sq};
        if (!neighbor_less(cand, data[0])) return;

        uint32_t i = 0;
        uint32_t half = k >> 1;
        while (i < half) {
            uint32_t left = (i << 1) + 1;
            uint32_t right = left + 1;
            uint32_t largest = left;
            if (right < k && neighbor_less(data[left], data[right])) {
                largest = right;
            }
            if (!neighbor_less(cand, data[largest])) break;
            data[i] = data[largest];
            i = largest;
        }
        data[i] = cand;
        threshold_sq = data[0].dist_sq;
    }

    inline void sort_ascending() {
        std::sort(data, data + size, neighbor_less);
    }
};

struct QueryPredicate {
    uint32_t target_category;
    const uint32_t* eligible_ids;
    uint32_t eligible_count;
    uint32_t sel_index;
};

struct QueryExecutionMetrics {
    double prep_us;
    double calc_us;
    double total_us;
    uint64_t cpu_cycles;
    uint32_t dists_evaluated;
    uint32_t eligible_total;
    uint32_t partitions_visited;
    uint32_t partitions_total;
    uint64_t bytes_read;
};

/*
 * Evictor de cache L1/L2/L3 (64 MB > 24 MB Intel Smart Cache L3 en i5-13500)
 * para mediciones reales en frio (Cold Cache).
 */
class ColdCacheEvictor {
    uint8_t* buffer_;
    size_t size_;
    volatile uint64_t sink_;
public:
    explicit ColdCacheEvictor(size_t size_mb = 64)
        : size_(size_mb * 1024ULL * 1024ULL), sink_(0) {
        buffer_ = static_cast<uint8_t*>(aligned_alloc_64(size_));
        if (buffer_) {
            for (size_t i = 0; i < size_; i += 64) {
                buffer_[i] = static_cast<uint8_t>((i >> 6) & 0xFF);
            }
        }
    }

    ~ColdCacheEvictor() {
        aligned_free_64(buffer_);
    }

    void flush_llc() {
        if (!buffer_) return;
        uint64_t acc = 0;
        for (size_t i = 0; i < size_; i += 64) {
            buffer_[i] ^= 1;
            acc += buffer_[i];
        }
        sink_ = acc;
        _mm_mfence();
    }
};

} // namespace alg_flat

#endif // ALG_FLAT_TYPES_H
