# ALG Vector Engine (ALG 2.0) — Acquisition Brief

**Release:** `v1.0.0` (Reference Commit `275d417`)  
**License:** Apache License 2.0  
**Repository:** `https://github.com/Santirobm/ALG_2.0`

---

## 1. Executive Summary

**ALG Vector Engine** is a lightweight, zero-dependency C/C++ library for **exact k-NN vector search (squared L2 distance)** over `float32` embeddings. Rather than attempting to replace general-purpose approximate vector databases, ALG targets a specific, high-value bottleneck in production retrieval pipelines: **computing exact Top-K neighbors over a pre-filtered list of candidate IDs** (e.g., IDs produced by SQL queries, tenant isolation, or ACL metadata filters).

---

## 2. Technical Differentiator

In hybrid retrieval pipelines, metadata filters often narrow a corpus down to hundreds or tens of thousands of eligible vector IDs before similarity ranking:
- Approximate graph indexes (such as HNSW) degrade in recall or traversal efficiency under selective filters.
- Scanning the full corpus with a bitmap selector (such as FAISS `IDSelectorBitmap`) wastes memory bandwidth on non-eligible vectors.
- Copying candidate vectors into a temporary sub-index per query adds allocation and copy overhead.

ALG exposes `alg_search_compact()`, which evaluates **only the supplied `uint32_t` candidate IDs** directly against a 64-byte aligned in-memory corpus using a 4-way unrolled **AVX2+FMA** kernel (with runtime CPUID dispatch and portable scalar fallback) and a deterministic Top-K heap.

---

## 3. Current Capabilities (`v1.0.0`)

- **Stable C99 ABI (`include/alg_engine.h`):** Direct FFI compatibility with Python (`ctypes`), C, C++, Rust, Go, and C#.
- **Zero Runtime Dependencies:** No Python, BLAS, MKL, OpenMP, or external runtime required.
- **Deterministic Ranking:** Exact `float32` squared L2 distance with ascending vector ID tie-breaking (`0-ULP` deterministic).
- **Safety & Concurrency:** Strict `NaN`/`Inf` rejection, integer overflow guards, atomic corpus updates, and lock-free thread-safe concurrent reads.
- **Multi-Platform Packaging:** Builds static and shared libraries for **Windows x64** (`alg.dll`, `libalg.a`) and **Linux x64** (`libalg.so`, `libalg.a`) via `CMakeLists.txt` or `build.ps1`.

---

## 4. Verified Performance Results

All comparative benchmarks below were run against **Meta FAISS 1.15.1 (`IndexFlatL2`)** using identical vectors, queries, candidate IDs, and `binary64` ground-truth verification (`omp_set_num_threads(1)` on Windows 10 x64, Intel Raptor Lake CPU). Evidence scripts, CSVs, and SHA-256 checksums are preserved in `docs/benchmarks/external-validation-2026/`.

| Benchmark Suite | Comparison | ALG Speedup vs Fair FAISS Baseline | Recall vs `binary64` | Memory Overhead vs FAISS Flat |
| :--- | :--- | :---: | :---: | :---: |
| **Python (`ctypes`) Integration** (`N=90,000` real SimpleWiki `D=64`, `M=900` & `M=9,000`, `Top-K=10`) | `alg_search_compact` vs `FAISS compute_distance_subset + np.argpartition` | **1.34×–1.83×** (`p50`: 26.1 µs vs 47.9 µs @ 1%; 246.4 µs vs 330.7 µs @ 10%) | **100.0%** | **0.0%** (`22.02 MB` vs `22.02 MB`) |
| **Native C++ (`alg.dll`)** (`N=90,000` real `D=64`, `Top-K ∈ {10, 100}`, WARM & COLD) | `alg_search_compact` vs `FAISS compute_distance_subset + Top-K` | **1.44×–2.13×** | **100.0%** | **0.0%** |
| **Native C++ Scalability** (`N=500,000` `D=64`, `M=5,000` & `M=50,000`, WARM & COLD) | `alg_search_compact` vs `FAISS compute_distance_subset + Top-K` | **1.51×–1.97×** | **100.0%** | **0.0%** (`122.31 MB` vs `122.31 MB`) |

*(Note: Compared against FAISS's standard `IDSelectorBatch` / `IDSelectorBitmap` APIs, `alg_search_compact` is 4.45×–9.27× faster in Python `p50` and 2.35×–25.00× faster in native C++ because those selectors scan the full corpus. `compute_distance_subset` is used above as the fair algorithmic baseline.)*

---

## 5. Known Limitations & Unresolved Items

- **Unfiltered Full-Corpus Search (`alg_search` at 100%):** Advantage over FAISS `IndexFlatL2` is modest (**1.05×–1.17×** at `N=90K–500K`), governed by main memory bandwidth. ALG is not a general replacement for unfiltered ANN/Flat search.
- **`alg_search_buffer()` Overhead in `v1.0.0`:** Present in `v1.0.0` itself (not a post-release regression): `alg_search_buffer()` runs a per-call scalar `NaN`/`Inf` check over all `M × D` floats (`alg_is_finite_buffer()`), consuming 52.7%–85.2% of call time and making it slower than FAISS (`0.35×–0.96×`). Production callers in `v1.0.0` should use `alg_search_compact()`, which validates corpus finiteness once at ingestion (`alg_set_vectors()`).
- **500K Scalability Corpus:** Composed of 18% real embeddings (90K) and 82% L2-normalized synthetic combinations (410K); not yet benchmarked on 500K independent production embeddings.
- **Feature Scope (`v1.0.0`):** Supports `float32` and `ALG_METRIC_L2_SQR` only (no cosine/IP normalization helpers, quantization, or `ALG_SEARCH_EXACT_TOPK` refinement pass yet).
- **Test Environment:** Benchmarks were executed locally by the developer on Windows x64; Linux x64 binaries cross-compile cleanly, but comparative FAISS benchmarks have not yet been run on Linux hardware.

---

## 6. Development Status, Licensing & Commercial Paths

- **Status:** `v1.0.0` released (`275d417`). Current build (`build.ps1`) compiles cleanly for Windows x64 and Linux x64, and all **14/14 unit and regression tests** pass.
- **Licensing:** Apache License 2.0 with zero runtime third-party dependencies.
- **IP Diligence Note:** 27 of 39 Git commits carry a local workstation Git configuration (`Sgi.LM@tubhier.local`); standard written authorship confirmation (and employer IP waiver if applicable) should be completed prior to closing an exclusive IP transfer.
- **Commercial Paths:** Available for **technical evaluation** on prospective users' own filtered-ID workloads, **direct SDK integration** (Apache 2.0), **custom/commercial licensing**, or **full IP acquisition** (subject to diligence).

---

## 7. What Is Included in an Acquisition

Full source code (`src/`, `include/`), CMake and PowerShell multi-platform build scripts, 14-test unit suite, reproducible C++ and Python benchmark harnesses, historical milestone reports, Git history and `v1.0.0` release tag, and maintainer transition support (subject to negotiation).

---

## 8. Contact

**Author / Maintainer:** Santiago Robledo  
**Email:** `santiagorobledo198@gmail.com`  
**Repository:** `https://github.com/Santirobm/ALG_2.0`
