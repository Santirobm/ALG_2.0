/**
 * @file alg_test_hooks.h
 * @brief Internal Testing & Diagnostic Hooks for ALG Vector Engine
 * 
 * NOTE: This header is intended solely for internal unit testing and regression suites.
 * It is not part of the public production SDK distribution.
 */

#ifndef ALG_TEST_HOOKS_H
#define ALG_TEST_HOOKS_H

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Force the engine to execute the scalar fallback path (1) or restore automatic ISA detection (0).
 */
void alg_test_force_scalar_isa(int force);

/**
 * Simulate allocation failure (OOM) across engine allocators (1 = simulate OOM, 0 = normal allocation).
 */
void alg_test_simulate_oom(int enable);

/**
 * Reset cached CPU features to uninitialized state to test concurrent first-time detection.
 */
void alg_test_reset_cpu_cache(void);

#ifdef __cplusplus
}
#endif

#endif /* ALG_TEST_HOOKS_H */
