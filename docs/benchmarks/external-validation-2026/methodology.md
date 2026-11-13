# Benchmark Methodology & Reproducibility Guide

## 1. Test Environment & Software Versions

- **OS:** `Windows-10-10.0.19045-SP0` (Windows 10 x64)
- **CPU:** `Intel64 Family 6 Model 191 Stepping 2, GenuineIntel`
- **Compiler:** `zig c++` `0.17.0` (`-O3 -std=c++17`, `simd_avx2.cpp` compiled with `-mavx2 -mfma`)
- **ALG SDK:** `v1.0.0` (`bin/alg.dll`, `AVX2+FMA (Vectorized 4x-unrolled)`)
- **Python:** `3.13.5`
- **FAISS:** `1.15.1` (`faiss-cpu`)
- **NumPy:** `2.3.2`
- **Threading & Affinity:**
  - Native C++: `HIGH_PRIORITY_CLASS`, pinned to P-Core 0 (`SetThreadAffinityMask(..., 0x1)`), `omp_set_num_threads(1)`.
  - Python: `faiss.omp_set_num_threads(1)`.

---

## 2. Dataset Provenance (Embeddings Not Committed)

Raw `.f32` embedding binaries are not committed to this folder. They are generated from the public Hugging Face dataset:
- **Source:** `lsb/simplewiki2023-all-minilm-l6-v2-embedding` (`sentence-transformers/all-MiniLM-L6-v2`, 384D, CC-BY-SA).
- **Preparation Script:** `benchmarks/realworld/prepare_hf_embeddings.py`
  - 100,000 rows extracted: **90,000** corpus vectors + **10,000** held-out query vectors.
  - Projected to `D=64` via `IncrementalPCA` fitted **only on the 90,000 corpus vectors**, then L2 unit-normalized (`||x||₂ = 1.0`).

### Dataset SHA-256 Checksums
```text
F1F541F63BF6CACD82418EDAA76B8E091E83F94EC574327794181665F9CABD6E  data/hito3_3/D64/corpus.f32
56F49B15A7AE10FCF0087C77C8CA6BF5A60BDE60842169004D91E383B36E89CD  data/hito3_3/D64/queries.f32
CADF362E8A5D4AE7F27D2EE1D389884EE52FA3E8A7FE7C81A9285F1753E17E30  data/hito3_3/D64/manifest.json
```

### Synthetic `N=500,000` Expansion (Scalability Test Only)
For `native_n500k_d64_synthetic.csv`, the corpus is expanded in memory as follows:
- **18.0% (90,000 vectors):** Original real SimpleWiki `D=64` embeddings.
- **82.0% (410,000 vectors):** Synthetic vectors generated via `w * v_a + (1 - w) * v_b` (`a != b` sampled uniformly from `[0, 89999]`, `w ~ Uniform(0.15, 0.85)`, fixed seed `20261112`), then L2 unit-normalized.
- **Observed $L_2^2$ distance distribution (5,000 samples vs queries):**
  - Real: `mean=1.99838, std=0.294031, min=0.378383, max=2.84385`
  - Synthetic: `mean=2.00897, std=0.289423, min=0.53398, max=2.95729`

---

## 3. Fairness & Verification Controls

1. **Identical Inputs:** Both engines receive the exact same corpus vectors, query vectors, sorted unique `eligible_ids`, `top_k`, and L2 squared metric.
2. **Cache Regimes (Native C++):**
   - `WARM_CACHE`: Pre-touched immediately before timing.
   - `COLD_LLC_FLUSH`: 36 MB buffer strided at 64 bytes (`_mm_mfence()`) before each timed query.
3. **Interleaved Order:** Execution order alternates (`ALG -> FAISS` / `FAISS -> ALG`) across queries and repetitions to cancel thermal/frequency drift.
4. **Accuracy Oracle:** Independent `binary64` (`double`) distance computation with deterministic ascending ID tie-breaking. Both engines achieved `100.00%` `IDSetRecall` in all reported cells.
5. **Fair FAISS By-ID Path:** Rather than comparing `alg_search_compact` only against `IDSelectorBitmap`/`IDSelectorBatch` (which scan the full corpus), the primary comparison uses `faiss::IndexFlat::compute_distance_subset` (which evaluates only the `M` candidate IDs) + Top-K selection.

---

## 4. Reproducing the Results

From the repository root (`ALG_2.0/`):

### A. Build ALG SDK (`bin/alg.dll`)
```powershell
.\build.ps1
```

### B. Prepare `D=64` Dataset (if not already in `data/hito3_3/D64/`)
```powershell
python benchmarks/realworld/prepare_hf_embeddings.py
```

### C. Compile & Run Native C++ Benchmark
```powershell
$ZIG = (Get-Command zig.exe).Source
$PY_ROOT = (python -c "import sys; print(sys.prefix)")
$NP_INC = (python -c "import numpy; print(numpy.get_include())")
$FAISS_DIR = (python -c "import os, faiss; print(os.path.dirname(faiss.__file__))")

& $ZIG c++ -O3 -std=c++17 -Iinclude "-I$PY_ROOT\include" "-I$NP_INC" `
    benchmarks\sdk_comparison\run_sdk_vs_faiss_native.cpp `
    benchmarks\faiss_comparison\faiss_c_shim.cpp `
    bin\libalg_dll.a "$FAISS_DIR\faiss.dll" "$PY_ROOT\libs\python313.lib" -lpsapi `
    -o benchmarks\sdk_comparison\run_sdk_vs_faiss_native.exe

$env:PATH = "$PWD\bin;$PY_ROOT;$FAISS_DIR;$PY_ROOT\Lib\site-packages\faiss_cpu.libs;$PY_ROOT\Lib\site-packages\numpy.libs;$env:PATH"

# N=90,000 real corpus + validation isolation
.\benchmarks\sdk_comparison\run_sdk_vs_faiss_native.exe --dim 64 --n 90000 --queries 200 --reps 3 --out results\sdk_comparison\native_n90k_d64_v2.csv

# N=500,000 partially synthetic scalability test
.\benchmarks\sdk_comparison\run_sdk_vs_faiss_native.exe --dim 64 --n 500000 --queries 100 --reps 2 --skip-buffer --out results\sdk_comparison\native_n500k_d64_synthetic.csv
```

### D. Run Python (`ctypes` vs FAISS) Integration Benchmark
```powershell
python benchmarks\sdk_comparison\run_python_integration_bench.py
```

---

## 5. Artifact SHA-256 Checksums

```text
D205042CF68B7E2A82E00C28143001D3AB15B6F6E927607BDD64E170FA522192  docs/benchmarks/external-validation-2026/native_n90k_d64.csv
780641E1B4BED4B36F1D7278169F130A4D37D154544F00E4C09F21F6EE1A3B29  docs/benchmarks/external-validation-2026/native_n90k_d64_v2.csv
0EC454E7D4317F0B1AADD78EFA888E3B763A54B3CC8E4170C2D8E1F4E54FF888  docs/benchmarks/external-validation-2026/native_n500k_d64_synthetic.csv
E7889A98FAFE18E10B873E3E3CCC40A31117E5ECD27E5F9465D96F6E16E84EDF  docs/benchmarks/external-validation-2026/buffer_validation_isolation.csv
8ECCDE1B2745485DA9F9F5140DB6D1C0E1BDC23E0F0153CF13C0A52B91F570DA  docs/benchmarks/external-validation-2026/python_n90k_d64.csv
14563CD40E252970CA194452AEB26F6F0E0D81534E4D603224B44A61310A16DA  docs/benchmarks/external-validation-2026/run_sdk_vs_faiss_native.cpp
3371F6E0CA0FE6FC92F4CC648C45C460F5AC1052D53545000D0FDA6FD077BAE3  docs/benchmarks/external-validation-2026/run_python_integration_bench.py
```
