# Changelog - ALG Vector Engine

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-11-04

### Added
- **Public C ABI** (`include/alg_engine.h`): Standard C99/C11 compatible header with opaque handle `AlgIndex`, explicit search modes, distance metrics, status codes, and configuration types.
- **Dynamic SIMD Dispatch**: Runtime ISA detection via CPUID and XGETBV (AVX2+FMA with portable scalar fallback). Core library compiles with baseline flags; AVX2+FMA kernel is isolated in a dedicated compilation unit.
- **Contiguous 64-Byte Aligned Storage**: Memory allocator (`alg_aligned_alloc_64`) preventing cache-line split penalties on AVX2 transactions.
- **Search Capabilities**:
  - `alg_search`: Sequential full-corpus scan.
  - `alg_search_compact`: Filtered scan over an explicit candidate ID list.
  - `alg_search_buffer`: Direct scan over an external contiguous buffer.
- **Deterministic Tie-Breaking**: Guarantees identical Top-K ordering by smaller vector ID on exact float32 distance ties (0 ULP difference).
- **Hardening and Error Robustness**:
  - Thread-safe lock-free CPUID detection using `std::atomic<uint32_t>`.
  - Non-finite input policy: explicit rejection of `NaN`, `+Inf`, `-Inf` in corpus and queries returning `ALG_INVALID_ARGUMENT`.
  - Atomicity in `alg_set_vectors`: existing index remains unmodified if reallocation or validation fails.
  - Guaranteed `*out_count = 0` invariant upon entry for all error paths.
  - Dynamic heap reallocation with arithmetic overflow checks and `ALG_OUT_OF_MEMORY` propagation for $K > 512$.
  - Arithmetic overflow protection in 64-bit size calculations (`alg_mul_overflow_2`, `alg_mul_overflow_3`).
- **Comprehensive 14-Suite Test Suite** (`tests/test_engine_unit.cpp`): 100% pass covering invalid arguments, finite checks, OOM propagation, dynamic heaps, duplicate candidate IDs, exhaustive scalar vs AVX2 ranking equivalence across 10 dimensions, 16-thread read concurrency with serial reference validation, and concurrent index initialization.
- **Standalone Examples and Benchmark Harness**: Pure C example, modern C++ RAII example, and CLI benchmark harness.
