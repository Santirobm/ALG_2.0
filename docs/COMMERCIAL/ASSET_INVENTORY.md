# ALG Vector Engine — Commercial Asset Inventory

**Repository:** `Santirobm/ALG_2.0`  
**Reference Release:** `v1.0.0` (Commit `275d417`)  
**License:** Apache License 2.0

This document lists the exact technical and repository assets included in a potential transfer, alongside third-party components and excluded artifacts.

---

## 1. Core Source Code & Public API (`src/`, `include/`)

- **Public C99 ABI Header:** `include/alg_engine.h` (opaque `AlgIndex*`, `alg_create_index`, `alg_set_vectors`, `alg_search`, `alg_search_compact`, `alg_search_buffer`, `alg_distance_l2_sqr`, ISA/version query helpers).
- **Core Engine (`src/core/`):**
  - `alg_index.cpp` / `alg_index.h`: Index lifecycle, atomic `alg_set_vectors` validation (NaN/Inf rejection, overflow checks), 64-byte aligned corpus storage.
  - `alg_search.cpp` / `alg_search.h`: Search dispatch for full-corpus (`alg_search`), pre-filtered ID list (`alg_search_compact`), and contiguous buffer (`alg_search_buffer`).
  - `alg_heap.h`: Deterministic Top-K max-heap with ascending vector ID tie-breaking.
- **SIMD & Scalar Kernels (`src/kernels/`):**
  - `simd_avx2.cpp`: 4-way unrolled AVX2+FMA L2 squared distance kernel (compiled in an isolated translation unit with `-mavx2 -mfma` / `/arch:AVX2`).
  - `simd_scalar.cpp`: Portable scalar fallback kernel.
  - `kernels.h`: Kernel dispatch table.
- **Platform Abstraction (`src/platform/`):**
  - `cpu_features.cpp` / `cpu_features.h`: Runtime CPUID detection for AVX2 and FMA.
  - `allocator.cpp` / `allocator.h`: 64-byte aligned memory allocation/deallocation and safe arithmetic overflow helpers.
- **Research / Legacy Headers & Modules (present in `src/` and `include/`):**
  - `execution_plans.cpp`, `physical_layouts.cpp`, `dataset_loader.cpp`, and corresponding headers in `include/` (`alg_flat_types.h`, `execution_plans.h`, `ieee754_bounds.h`, `physical_layouts.h`, `simd_kernels.h`, `alg_test_hooks.h`).

---

## 2. Build & Packaging Infrastructure

- **`CMakeLists.txt`**: Builds `alg_static`, `alg_shared` (`alg.dll` / `libalg.so`), `test_engine_unit`, examples, and `alg_benchmark_cli`; registers `CTest`.
- **`build.ps1`**: Multi-platform build script using `zig c++` for Windows x64 (`bin/libalg.a`, `bin/alg.dll`, `bin/libalg_dll.a`) and Linux x64 cross-compilation (`bin/linux-x64/libalg.a`, `bin/linux-x64/libalg.so`), test execution, and SDK packaging into `dist/alg-vector-engine-sdk/`.

---

## 3. Tests, Examples & Benchmarks

- **Unit & Regression Test Suite (`tests/test_engine_unit.cpp`):** 14 verified tests covering argument validation, `out_count=0` error guarantees, NaN/Inf rejection, atomic `alg_set_vectors`, dimension bounds & integer overflow, OOM handling, out-of-bounds ID rejection, `top_k > 512`, `top_k > candidate_count`, duplicate IDs, scalar vs AVX2 ranking equivalence, multi-threaded concurrent reads, concurrent index initialization, and deterministic ID tie-breaking.
- **Exactness & Equivalence Tests (`tests/`):** `test_exactness.cpp`, `test_faiss_shim_equivalence.cpp`, `test_pure_cpp_faiss.cpp`.
- **Integration Examples (`examples/`):** `c_example.c`, `cpp_example.cpp`, `CMakeLists.txt`, `build_examples.bat`, `build_examples.sh`.
- **CLI & Comparison Benchmarks (`tools/`, `benchmarks/`):**
  - `tools/alg_benchmark_cli.cpp`
  - `benchmarks/sdk_comparison/`: `run_sdk_vs_faiss_native.cpp` (native `alg.dll` vs FAISS `IndexFlatL2` + `compute_distance_subset`) and `run_python_integration_bench.py` (Python `ctypes` vs `faiss-cpu`).
  - `benchmarks/faiss_comparison/` and `benchmarks/realworld/`: Historical milestone harnesses (Hito 3.2, 3.3, 3.3.1, 3.3.2).

---

## 4. Documentation & Benchmark Evidence

- **Core Docs:** `README.md`, `CHANGELOG.md`, `AUTHORS.md`, `CONTRIBUTING.md`, `LICENSE`, `THIRD_PARTY_NOTICES.md`.
- **External Validation Baseline (`docs/benchmarks/external-validation-2026/`):** `README.md`, `methodology.md`, reproducible C++/Python benchmark scripts, 5 CSV result files, and SHA-256 checksums.
- **Historical Reports (`results/`):** Milestone reports and CSVs from Hito 1, 3, 3.2, 3.3, 3.3.1, 3.3.2, and `results/sdk_comparison/`.
- **Commercial Readiness Docs (`docs/COMMERCIAL/`):** `OWNERSHIP.md`, `ASSET_INVENTORY.md`, `ACQUISITION_BRIEF.md`.

---

## 5. Git History & Release Tags

- Full Git commit history (39 commits on `chore/commercial-readiness`, plus milestone branches).
- Release tags: `v1.0.0` (at `275d417`), `v2.0-hito2b-baseline`, `v2.0-hito2b1-rejected`.

---

## 6. Third-Party Components & Excluded Items

| Item | Status | Notes |
| :--- | :--- | :--- |
| `third_party/faiss/` | **Third-Party (MIT)** | Meta FAISS `v1.15.1` headers, `.def`, `.lib` used solely for benchmark comparison. Copyright Meta Platforms, Inc. |
| `data/*.f32`, `*.u32`, `*.jsonl` | **Excluded** | Gitignored evaluation embeddings derived from Hugging Face `lsb/simplewiki2023-all-minilm-l6-v2-embedding`; reproducible via `prepare_hf_embeddings.py`. |
| Root `*.exe` / `*.pdb` / `bin/` / `dist/` | **Excluded** | Build artifacts ignored by `.gitignore` (rebuilt cleanly via `build.ps1` or CMake). |
