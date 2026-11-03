# Hito 3.3.1 — Benchmark Integrity & Filter Ablation

Status: protocol and dataset-preparation utility. This document does not claim that the four-way executable or benchmark results have been completed.

## Non-negotiable controls

- Keep Plan 5A and its certified source files unchanged.
- Use the same corpus, query vectors, eligible ID set, Top-K, output contract, warm/cold protocol, and query order for all applicable variants.
- Keep index construction, filter construction, ID compaction, and buffer allocation outside the timed query region. Report build/preparation costs separately.
- Use at least 1,000 independently evaluated queries per condition; prefer all 10,000 held-out queries when runtime permits.
- Record exact eligible count per query; target selectivity is not a substitute for realized selectivity.
- Ground truth is binary64 over exactly the same eligible IDs. Report IDSetRecall and StrictRankMatch separately.
- Report distance evaluations, candidate IDs visited, partitions visited/pruned, and filter operations. Do not infer geometric pruning from wall-clock speedup alone.

## Four variants and their interpretation

### A — ExactScan AVX2 + explicit eligible-ID list
A purpose-built exact scan over the compact eligible ID list. It computes distance only for those IDs. Document the distance kernel, Top-K implementation, tie policy, and numeric mode. This is the lower-overhead compact-list reference.

### B — ALG Plan 5A + the same eligible set
Canonical Plan 5A execution on the partitioned layout. Verify mapping from original corpus IDs to partition-local/physical positions. Log eligible_count, dists_evaluated, partitions visited, and partitions pruned. Do not modify Plan 5A.

### C1 — FAISS Flat over a compacted eligible subset
Build a separate IndexFlatL2 from vectors in eligible_ids order, outside the timed region. Search this compact index with no selector, then map local labels back through eligible_ids. This is the actual compact-list FAISS control. It is NOT equivalent to passing IDSelectorBatch to an original full-corpus IndexFlat.

### C2 — FAISS Flat over the full corpus + IDSelectorBitmap
Search the original full-corpus IndexFlat with a bitmap selector. This measures full traversal plus selector membership. It is intentionally a different filter strategy, not the symmetric compact-list control.

## Required comparisons

- A vs B: net effect of ALG partition organization and geometric pruning relative to contiguous exact scan over the same eligible IDs.
- A vs C1: distance-kernel / Top-K / compact-index differences on the same compact candidate population.
- C1 vs C2: cost of compacting/iterating an eligible subset versus scanning full Flat with bitmap membership.
- B vs C2: end-to-end architecture comparison, explicitly labeled asymmetric in filter execution.
- Do not present B vs C2 as an isolated measurement of geometric pruning.

## Filter regimes

Run two independent families:

1. Uniform/random control: eligible IDs selected independently of vector coordinates. This tests no-correlation behavior and partition occupancy.
2. Embedding-correlated synthetic filters: use prepare_semantic_filters.py. MiniBatchKMeans is fit on corpus embeddings only; held-out query vectors are assigned to frozen centroids. Eligibility is the union of nearest centroid clusters until the requested population is reached. These are synthetic labels, not authentic tenant/category metadata.

If genuine metadata columns become available in a source dataset, add a third family using those values and preserve provenance. Never relabel synthetic cluster IDs as real tenant, category, or topic fields.

## Timing protocol

- Separate setup/build/filter materialization from per-query search.
- Rotate variant execution order using a seeded balanced schedule to avoid systematic thermal/cache bias.
- WARM: explicit warm-up pass, then measured repetitions.
- COLD: flush LLC immediately before each timed search; verify eviction buffer exceeds LLC capacity.
- Record raw per-query latency; report mean, median, p95, p99, QPS and confidence intervals. A p99 from only 100 samples is not decision-grade.
- Pin to the same physical core and use identical thread settings.
- Run correctness before timing and verify timed outputs as well.

## Numerical audit correction

The statement delta < epsilon_machine(float32) does NOT prove two values round to the same float32. One ULP depends on exponent and rounding-boundary position; two values separated by less than one ULP can still round to adjacent representable values. The forensic report's displayed float32 values (1.0741607 and 1.0741608) appear different, while its prose says they collapse identically. Resolve this by recording raw 32-bit hexadecimal bit patterns of both independently recomputed float distances, exact accumulation order, and comparator/heap tie behavior. Until then, classify the mismatch as a near-ULP precision discrepancy, not a proven exact float32 tie.

## Acceptance criteria

- A, B, C1 and C2 use the same eligible ID set for a given query/condition.
- Compare all variants against one binary64 oracle on those same eligible IDs.
- C1 labels are mapped back to original corpus IDs before scoring.
- No missing/duplicate IDs; candidate counts and checksums are recorded.
- All per-query raw timings and a machine-readable run manifest are retained.
- No performance claim is accepted unless variant and filter family are named.
