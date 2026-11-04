#ifndef ALG_PLATFORM_ALLOCATOR_H
#define ALG_PLATFORM_ALLOCATOR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void* alg_aligned_alloc_64(size_t bytes);
void alg_aligned_free_64(void* ptr);

#ifdef __cplusplus
}
#endif

#endif /* ALG_PLATFORM_ALLOCATOR_H */
