/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "allocator.h"
#include <stdlib.h>
#include <string.h>
#include <atomic>

#if defined(_WIN32)
#include <malloc.h>
#endif

static std::atomic<bool> g_simulate_oom{false};

void alg_test_simulate_oom(int enable) {
    g_simulate_oom.store(enable != 0, std::memory_order_seq_cst);
}

bool alg_test_is_oom_simulated(void) {
    return g_simulate_oom.load(std::memory_order_relaxed);
}

void* alg_aligned_alloc_64(size_t bytes) {
    if (g_simulate_oom.load(std::memory_order_relaxed)) {
        return NULL;
    }

    if (bytes == 0) bytes = 64;

    // Arithmetic overflow protection on bytes + 63
    if (bytes > (SIZE_MAX - 63)) {
        return NULL;
    }

    size_t padded = (bytes + 63) & ~size_t(63);

#if defined(_WIN32)
    void* ptr = _aligned_malloc(padded, 64);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    void* ptr = aligned_alloc(64, padded);
#elif defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L
    void* ptr = NULL;
    if (posix_memalign(&ptr, 64, padded) != 0) {
        ptr = NULL;
    }
#else
    // Fallback using manual padding
    if (padded > (SIZE_MAX - 64 - sizeof(void*))) {
        return NULL;
    }
    void* raw = malloc(padded + 64 + sizeof(void*));
    if (!raw) return NULL;
    void** ptr = (void**)(((size_t)raw + sizeof(void*) + 63) & ~size_t(63));
    ptr[-1] = raw;
    memset(ptr, 0, padded);
    return (void*)ptr;
#endif

    if (ptr) {
        memset(ptr, 0, padded);
    }
    return ptr;
}

void alg_aligned_free_64(void* ptr) {
    if (!ptr) return;

#if defined(_WIN32)
    _aligned_free(ptr);
#elif defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L) || (defined(_POSIX_C_SOURCE) && _POSIX_C_SOURCE >= 200112L)
    free(ptr);
#else
    void** p = (void**)ptr;
    free(p[-1]);
#endif
}
