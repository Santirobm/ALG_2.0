/*
 * Copyright (c) 2026 Santiago Robledo.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "cpu_features.h"
#include "../../include/alg_engine.h"
#include <atomic>

#if defined(_MSC_VER) || defined(_WIN32)
#include <immintrin.h>
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#include <immintrin.h>
#include <cpuid.h>
#include <x86intrin.h>
#endif

static std::atomic<uint32_t> g_cached_features{0xFFFFFFFFu};
static std::atomic<bool> g_force_scalar{false};

static inline void run_cpuid(int cpu_info[4], int func_id, int subfunc_id) {
#if defined(_MSC_VER) || defined(_WIN32)
    __cpuidex(cpu_info, func_id, subfunc_id);
#elif defined(__GNUC__) || defined(__clang__)
    __cpuid_count(func_id, subfunc_id, cpu_info[0], cpu_info[1], cpu_info[2], cpu_info[3]);
#else
    cpu_info[0] = cpu_info[1] = cpu_info[2] = cpu_info[3] = 0;
#endif
}

static inline uint64_t run_xgetbv(uint32_t xcr) {
#if defined(_MSC_VER) || defined(_WIN32)
    return _xgetbv(xcr);
#elif defined(__GNUC__) || defined(__clang__)
    uint32_t eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(xcr));
    return ((uint64_t)edx << 32) | eax;
#else
    return 0;
#endif
}

static uint32_t detect_features_internal(void) {
    uint32_t features = ALG_CPU_FEATURE_NONE;
    int info[4] = {0};

    // Check maximum basic CPUID function
    run_cpuid(info, 0, 0);
    int max_func = info[0];
    if (max_func < 1) {
        return features;
    }

    // Function 1: ECX has OSXSAVE (bit 27), FMA (bit 12), AVX (bit 28)
    run_cpuid(info, 1, 0);
    bool osxsave = (info[2] & (1 << 27)) != 0;
    bool fma     = (info[2] & (1 << 12)) != 0;
    bool avx     = (info[2] & (1 << 28)) != 0;

    if (!osxsave || !avx) {
        return features;
    }

    // Check XGETBV to verify OS supports saving YMM state (XMM bit 1, YMM bit 2)
    uint64_t xcr0 = run_xgetbv(0);
    if ((xcr0 & 0x6) != 0x6) {
        return features;
    }

    if (fma) {
        features |= ALG_CPU_FEATURE_FMA;
    }

    // Function 7, subfunction 0: EBX has AVX2 (bit 5)
    if (max_func >= 7) {
        run_cpuid(info, 7, 0);
        bool avx2 = (info[1] & (1 << 5)) != 0;
        if (avx2) {
            features |= ALG_CPU_FEATURE_AVX2;
        }
    }

    return features;
}

uint32_t alg_platform_detect_cpu_features(void) {
    // Thread-safe lock-free double-checked acquire/release
    uint32_t cached = g_cached_features.load(std::memory_order_acquire);
    if (cached == 0xFFFFFFFFu) {
        uint32_t detected = detect_features_internal();
        g_cached_features.store(detected, std::memory_order_release);
        cached = detected;
    }
    if (g_force_scalar.load(std::memory_order_relaxed)) {
        return ALG_CPU_FEATURE_NONE;
    }
    return cached;
}

bool alg_platform_has_avx2(void) {
    if (g_force_scalar.load(std::memory_order_relaxed)) {
        return false;
    }
    uint32_t feat = alg_platform_detect_cpu_features();
    return (feat & ALG_CPU_FEATURE_AVX2) && (feat & ALG_CPU_FEATURE_FMA);
}

bool alg_platform_has_fma(void) {
    if (g_force_scalar.load(std::memory_order_relaxed)) {
        return false;
    }
    uint32_t feat = alg_platform_detect_cpu_features();
    return (feat & ALG_CPU_FEATURE_FMA) != 0;
}

const char* alg_platform_get_isa_string(void) {
    if (alg_platform_has_avx2()) {
        return "AVX2+FMA (Vectorized 4x-unrolled)";
    }
    return "Scalar Fallback (IEEE 754 float32)";
}

void alg_platform_set_force_scalar(bool force) {
    g_force_scalar.store(force, std::memory_order_seq_cst);
}

void alg_test_reset_cpu_cache(void) {
    g_cached_features.store(0xFFFFFFFFu, std::memory_order_seq_cst);
    g_force_scalar.store(false, std::memory_order_seq_cst);
}

void alg_test_force_scalar_isa(int force) {
    alg_platform_set_force_scalar(force != 0);
}
