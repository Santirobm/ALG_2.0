#ifndef ALG_PLATFORM_ALLOCATOR_H
#define ALG_PLATFORM_ALLOCATOR_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 64-byte aligned memory allocation with arithmetic overflow protection.
 * Returns NULL if bytes > SIZE_MAX - 63 or if OS allocation fails.
 */
void* alg_aligned_alloc_64(size_t bytes);

/**
 * Release 64-byte aligned memory allocated with alg_aligned_alloc_64.
 * Safe to pass NULL.
 */
void alg_aligned_free_64(void* ptr);

/**
 * Safe multiplication of two size_t values.
 * Returns true if an overflow would occur, false otherwise.
 */
static inline bool alg_mul_overflow_2(size_t a, size_t b, size_t* out_result) {
    if (a == 0 || b == 0) {
        if (out_result) *out_result = 0;
        return false;
    }
    if (a > SIZE_MAX / b) {
        if (out_result) *out_result = 0;
        return true;
    }
    if (out_result) *out_result = a * b;
    return false;
}

/**
 * Safe multiplication of three size_t values: (a * b * c).
 * Returns true if an overflow would occur, false otherwise.
 */
static inline bool alg_mul_overflow_3(size_t a, size_t b, size_t c, size_t* out_result) {
    size_t ab;
    if (alg_mul_overflow_2(a, b, &ab)) {
        if (out_result) *out_result = 0;
        return true;
    }
    return alg_mul_overflow_2(ab, c, out_result);
}

#ifdef __cplusplus
}
#endif

#endif /* ALG_PLATFORM_ALLOCATOR_H */
