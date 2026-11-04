#ifndef ALG_PLATFORM_CPU_FEATURES_H
#define ALG_PLATFORM_CPU_FEATURES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

uint32_t alg_platform_detect_cpu_features(void);
bool alg_platform_has_avx2(void);
bool alg_platform_has_fma(void);
const char* alg_platform_get_isa_string(void);

#ifdef __cplusplus
}
#endif

#endif /* ALG_PLATFORM_CPU_FEATURES_H */
