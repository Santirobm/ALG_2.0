# ALG Vector Engine

> **English** | [Español](README.es.md)

## What is ALG?

**ALG Vector Engine** is a zero-dependency C/C++ library for **exact k-NN vector search** (squared L2 distance) over `float32` embeddings.

In production retrieval pipelines, queries rarely run over an entire unfiltered corpus. Metadata filters (tenant ID, permissions, SQL predicates, or time windows) narrow millions of vectors down to a specific list of eligible candidate IDs:
- Approximate graph indexes (such as HNSW) can lose recall or traversal efficiency under tight filters.
- Full-corpus bitmap scans (such as FAISS `IDSelectorBitmap`) waste memory bandwidth evaluating non-eligible vectors.
- Copying candidate vectors into temporary per-query sub-indexes adds allocation and copy overhead.

ALG solves this specific bottleneck via `alg_search_compact()`, evaluating **only the supplied `uint32_t` candidate IDs** directly against a 64-byte aligned in-memory corpus with deterministic Top-K ranking.

---

## Key Features

- **Pure C99 ABI (`include/alg_engine.h`):** Direct FFI compatibility with Python (`ctypes`), C, C++, Rust, Go, and C#.
- **Dynamic SIMD Dispatch:** Automatically selects the 4-way unrolled **AVX2+FMA** kernel or portable scalar fallback at runtime via CPUID.
- **Zero External Dependencies:** Compiles with standard C++17/C11 compilers (GCC, Clang, MSVC, Zig) without Python, BLAS, MKL, or OpenMP.
- **64-Byte Aligned Memory:** Contiguous corpus storage aligned to cache lines for predictable L1/L2/L3 access.
- **Safety & Determinism:** Rejects `NaN`/`Inf` inputs, guards against integer overflow, provides atomic corpus updates, and guarantees deterministic ascending-ID tie-breaking.
- **Thread-Safe Reads:** Read-only searches (`alg_search`, `alg_search_compact`, `alg_search_buffer`) execute concurrently across multiple threads without locks.
- **Multi-Platform SDK:** Supports Windows x64 (`alg.dll`, `libalg.a`) and Linux x64 (`libalg.so`, `libalg.a`).

---

## Benchmarks

### 1. Comparative Validation vs. FAISS ([docs/benchmarks/external-validation-2026/](docs/benchmarks/external-validation-2026/))

Reproducible measurements of `bin/alg.dll` (`v1.0.0`) against **Meta FAISS 1.15.1 (`IndexFlatL2`)** using identical vectors, queries, candidate ID lists, squared L2 metric, and deterministic ID tie-breaking (`omp_set_num_threads(1)`, Windows 10 x64, Intel Raptor Lake CPU, `zig c++ 0.17.0 -O3`).

To avoid asymmetric comparisons against full-corpus bitmaps (`IDSelectorBitmap`), the fair baseline evaluates the exact same `M` candidate IDs using `faiss::IndexFlat::compute_distance_subset`.

| Suite | Corpus & Origin | Filter / Top-K | ALG (`alg_search_compact`) | FAISS (`compute_distance_subset`) | ALG Speedup | Recall vs `binary64` |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Python (`ctypes`)** | 90,000 real (SimpleWiki PCA `D=64`) | `M=900` (1%), `K=10` | `26.10 µs` (`p50`) / `33.55 µs` (mean) | `47.85 µs` (`p50`) / `60.03 µs` (mean) | **1.83× (`p50`)** | 100.0% |
| **Python (`ctypes`)** | 90,000 real (SimpleWiki PCA `D=64`) | `M=9,000` (10%), `K=10` | `246.40 µs` (`p50`) / `279.29 µs` (mean) | `330.70 µs` (`p50`) / `397.52 µs` (mean) | **1.34× (`p50`)** | 100.0% |
| **Native C++ (WARM/COLD)** | 90,000 real (SimpleWiki PCA `D=64`) | 1% & 10%, `K=10, 100` | `4.79 – 394.30 µs` | `10.21 – 706.96 µs` | **1.44× – 2.13×** | 100.0% |
| **Native C++ 500K (scale)** | 500,000 (`18%` real + `82%` synthetic) | 1% (`5K`) & 10% (`50K`), `K=10, 100` | `85.59 – 1,883.35 µs` | `136.81 – 3,581.43 µs` | **1.51× – 1.97×** | 100.0% |

*Memory footprint:* Exact tie with `FAISS IndexFlatL2` (`22.02 MB` at 90K; `122.31 MB` at 500K — `0.0%` overhead).

### 2. Historical Standalone CLI Baseline (`tools/alg_benchmark_cli.cpp`, Synthetic)

Historical single-engine measurements (no FAISS comparison) on **100,000 synthetic vectors** (`200` queries, `Top-10`, Intel Core i5-13500, derived `QPS = 1,000,000 / Mean_µs`):

| Dimension | Selectivity | Candidates | Mean (µs) | p50 (µs) | p95 (µs) | QPS (derived) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 64 | 1% | 1,000 | 4.58 | 4.40 | 6.20 | 218,293 |
| 64 | 10% | 10,000 | 77.47 | 72.00 | 102.40 | 12,909 |
| 64 | 100% | 100,000 | 913.48 | 715.20 | 1,543.60 | 1,095 |
| 128 | 1% | 1,000 | 11.33 | 11.20 | 12.10 | 88,261 |
| 128 | 10% | 10,000 | 134.47 | 129.10 | 160.70 | 7,437 |
| 128 | 100% | 100,000 | 2,836.26 | 2,462.60 | 4,512.90 | 353 |
| 256 | 1% | 1,000 | 24.57 | 24.60 | 25.10 | 40,703 |
| 256 | 10% | 10,000 | 331.02 | 289.90 | 661.90 | 3,021 |
| 256 | 100% | 100,000 | 5,662.05 | 5,388.10 | 7,019.70 | 177 |

### Scope & Known Limitations

- **Primary Use Case:** ALG's advantage is concentrated in `alg_search_compact()` over pre-filtered candidate ID lists (`1.34×–2.13×` vs. FAISS on the same candidates).
- **Unfiltered Full-Corpus Search (`alg_search` at 100%):** Advantage over FAISS `IndexFlatL2` is modest (**1.05×–1.17×** at 90K–500K), bounded by DRAM bandwidth. ALG is not positioned as a general replacement for unfiltered ANN/Flat search.
- **`alg_search_buffer()` in `v1.0.0`:** Re-validates `NaN`/`Inf` in scalar code across all `M × D` floats on every query, consuming 53%–85% of call time (`0.35×–0.96×` vs. FAISS). Use `alg_search_compact()` in `v1.0.0`.
- **Pending Validation:** Benchmarks are local developer measurements on Windows x64; independent Linux x64 hardware benchmarks and 500K+ real-embedding validation remain pending.

---

## Architecture

ALG separates **corpus ingestion** from **filtered query execution**:
1. **Ingestion (`alg_set_vectors`):** Validates all `N × D` floats once (`NaN`/`Inf` rejection) and copies vectors into 64-byte aligned contiguous memory.
2. **Query Dispatch (`alg_search_compact`):** Validates the `D`-element query and `eligible_ids[i] < N` bounds, then dispatches to the runtime-selected kernel (`simd_avx2.cpp` or `simd_scalar.cpp`).
3. **4-Way Unrolled Kernel & Top-K Heap:** Evaluates 4 candidate vectors per loop iteration using AVX2+FMA accumulators and maintains a deterministic max-heap (`alg_heap.h`) with ascending-ID tie-breaking.

```text
ALG_2.0/
├── include/alg_engine.h        # Public C99 ABI header
├── src/
│   ├── core/                   # Index lifecycle, search dispatch, Top-K heap
│   ├── kernels/                # AVX2+FMA 4x kernel & scalar fallback
│   └── platform/               # CPUID detection & 64-byte aligned allocator
├── tests/                      # 14-test unit & regression suite
├── examples/                   # Minimal C and C++ integration examples
├── tools/alg_benchmark_cli.cpp # Standalone CLI benchmark harness
└── docs/
    ├── benchmarks/external-validation-2026/
    └── COMMERCIAL/             # Acquisition brief, asset inventory, ownership
```

---

## Build & Installation

### Requirements
- C++17 / C11 compiler (GCC 7+, Clang 6+, MSVC 2017+, or Zig)
- CMake 3.14+ (or PowerShell + Zig on Windows)

### CMake (Linux & Windows)
```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
ctest --output-on-failure
```

### Windows PowerShell (`build.ps1`)
```powershell
.\build.ps1
```
Builds Windows x64 (`bin/alg.dll`, `bin/libalg.a`), cross-compiles Linux x64 (`bin/linux-x64/libalg.so`, `libalg.a`), runs all 14 unit tests, and packages the SDK into `dist/alg-vector-engine-sdk/`.

---

## API Usage

### C (`alg_search_compact`)
```c
#include <stdio.h>
#include "alg_engine.h"

int main(void) {
    AlgIndexConfig config = { .dimension = 64, .initial_capacity = 1000, .metric = ALG_METRIC_L2_SQR };
    AlgIndex* index = NULL;
    if (alg_create_index(&config, &index) != ALG_OK) return 1;

    // Ingest corpus_f32 (1000 * 64 floats)
    alg_set_vectors(index, 1000, corpus_f32);

    uint32_t eligible_ids[] = { 10, 25, 42, 99, 150, 320 };
    AlgSearchResult results[5];
    uint32_t out_count = 0;

    if (alg_search_compact(index, query_f32, eligible_ids, 6, 5,
                           ALG_SEARCH_FAST_FLOAT32, results, &out_count) == ALG_OK) {
        for (uint32_t i = 0; i < out_count; ++i) {
            printf("Rank %u: ID %u, L2^2 = %f\n", i + 1, results[i].id, results[i].distance);
        }
    }

    alg_free_index(index);
    return 0;
}
```

### C++ (`alg_search`)
```cpp
#include <vector>
#include "alg_engine.h"

int main() {
    AlgIndexConfig config{ 128, 5000, ALG_METRIC_L2_SQR };
    AlgIndex* index = nullptr;
    if (alg_create_index(&config, &index) != ALG_OK) return 1;

    std::vector<float> data(5000 * 128, 0.5f);
    alg_set_vectors(index, 5000, data.data());

    std::vector<float> query(128, 0.5f);
    std::vector<AlgSearchResult> results(10);
    uint32_t returned = 0;
    alg_search(index, query.data(), 10, ALG_SEARCH_FAST_FLOAT32, results.data(), &returned);

    alg_free_index(index);
    return 0;
}
```

---

## License

Distributed under the **Apache License 2.0**. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.

---

## Commercial Inquiries

For technical evaluation, custom integration, commercial licensing, or IP acquisition inquiries:
- **Acquisition Brief:** [docs/COMMERCIAL/ACQUISITION_BRIEF.md](docs/COMMERCIAL/ACQUISITION_BRIEF.md)
- **Maintainer:** Santiago Robledo (`santiagorobledo198@gmail.com`)
