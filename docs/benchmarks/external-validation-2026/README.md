# ALG Vector Engine — External Validation Baseline (2026)

> **Preliminary Developer Results.** These benchmarks were executed locally by the developer against `alg.dll` (`v1.0.0`, unmodified). They do **not** constitute independent third-party validation or a universal performance guarantee in production.

## Summary Highlights

| Metric | Result | Scope & Context |
| :--- | :--- | :--- |
| **1.34×–1.83×** | Median latency advantage (`p50`) from **Python (`ctypes`)** | Filtered search by ID (`alg_search_compact` vs `FAISS compute_distance_subset + np.argpartition`) on `N=90,000` real embeddings (`M=900` and `M=9,000`, `Top-K=10`) |
| **1.44×–2.13×** | Native C++ latency advantage (`N=90,000`) | `alg_search_compact` vs `FAISS IndexFlatL2::compute_distance_subset + Top-K` (WARM & COLD, `Top-K ∈ {10, 100}`) |
| **1.51×–1.97×** | Native C++ latency advantage (`N=500,000`) | Partially synthetic scalability test (`18%` real + `82%` derived synthetic, `M=5,000` and `M=50,000`) |
| **100%** | `IDSetRecall` vs `binary64` oracle | Exact match on all evaluated queries across both ALG and FAISS with identical deterministic ID tie-breaking |
| **64** | Embedding dimensions evaluated | `SimpleWiki` `all-MiniLM-L6-v2` projected corpus-only via PCA to `D=64` (L2 unit-normalized) |

---

## Where ALG Shows an Advantage — and Where It Does Not

1. **Filtered Search over Pre-Selected IDs (`alg_search_compact`):**
   - **Target use case:** Hybrid search / metadata pre-filtering where the application already holds an explicit array of eligible `uint32_t` IDs and needs exact L2 Top-K without building per-query sub-indexes or scanning a full-corpus bitmap.
   - **Fair comparison (same `M` candidates):** **1.44×–2.13×** faster in native C++ (`N=90K`), **1.51×–1.97×** in native C++ (`N=500K` partially synthetic), and **1.34×–1.83×** (`p50`) from Python `ctypes` (`N=90K`), with **0% memory overhead** (`22.02 MB` at 90K; `122.31 MB` at 500K, identical to `FAISS IndexFlatL2`).
   - *(Secondary reference)* Against FAISS standard `IDSelectorBitmap` / `IDSelectorBatch`, speedups are higher (**2.35×–25.00×** native, **4.45×–9.27×** Python `p50`) because those selectors evaluate membership over the full corpus. We do **not** use bitmap/batch selectors as our primary competitive claim.

2. **Full-Corpus Unfiltered Search (`alg_search` vs `FAISS IndexFlatL2`):**
   - Advantage is modest: **1.05×–1.17×** (`N=90K`) and **1.08×–1.11×** (`N=500K`), memory-bandwidth bound. ALG is **not** positioned as a general replacement for FAISS on unfiltered search.

3. **Contiguous Buffer Search (`alg_search_buffer`):**
   - In `v1.0.0`, `alg_search_buffer` **loses** to FAISS (`0.35×–0.96×`).
   - Isolation measurements ([buffer_validation_isolation.csv](buffer_validation_isolation.csv)) show this is caused by a per-query scalar finiteness check (`alg_is_finite_buffer` over all `M × D` floats) inside `src/core/alg_search.cpp`, which consumes **52.7%–85.2%** of total call time.

---

## Included Artifacts

| File | Description |
| :--- | :--- |
| [methodology.md](methodology.md) | Environment, compiler/library versions, dataset provenance, synthetic expansion rules, and SHA-256 checksums |
| [run_sdk_vs_faiss_native.cpp](run_sdk_vs_faiss_native.cpp) | Native C++ benchmark harness linking `bin/alg.dll` (`include/alg_engine.h`) and `faiss.dll` |
| [run_python_integration_bench.py](run_python_integration_bench.py) | Python `ctypes` vs `faiss` Python integration benchmark |
| [native_n90k_d64.csv](native_n90k_d64.csv) | Initial native run (`N=90,000`, `D=64`, 200 queries × 3 reps) |
| [native_n90k_d64_v2.csv](native_n90k_d64_v2.csv) | Native run including fair `compute_distance_subset` comparison (`2b_ByID_Subset`) |
| [buffer_validation_isolation.csv](buffer_validation_isolation.csv) | Isolated per-call validation cost of `alg_search_buffer` across `M ∈ {100..30,000}` |
| [native_n500k_d64_synthetic.csv](native_n500k_d64_synthetic.csv) | Partially synthetic scalability run (`N=500,000`, `D=64`, 100 queries × 2 reps) |
| [python_n90k_d64.csv](python_n90k_d64.csv) | Python `ctypes` vs FAISS Python run (`N=90,000`, `D=64`, 200 queries × 5 reps) |
