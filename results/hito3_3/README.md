# Hito 3.3 — Real-World Validation & Dimensional Scaling

## Baseline and invariants

- Base commit: `9e59436206c6db07b3e2d23511c24ed3e77807b1`.
- Branch: `hito3.3-realworld-validation`.
- `src/execution_plans.cpp`, `include/ieee754_bounds.h` and the certified Plan 5A kernels are frozen for this milestone.
- No benchmark may silently substitute synthetic data for a real embedding corpus.
- Exactness is checked against an independent binary64 ground truth; report both ID-set recall and strict rank agreement.
- FAISS Flat is the exact baseline. IVF/HNSW/ScaNN, if included, are explicitly approximate baselines and must report recall alongside latency.

## Dataset classes (do not conflate)

1. **Native-dimensional source:** a corpus whose stored embedding dimension is already 32, 64, 128 or 256. Report it as native only when this is true.
2. **Controlled PCA projection:** for a source with D >= target D, fit PCA on corpus rows only, then transform held-out queries with the same components. Report explained variance. These results are projected, not native.
3. Never zero-pad, truncate coordinates and call the result an equivalent embedding, or fit PCA on corpus + query together.

The initial exporter accepts a Hugging Face dataset ID and an existing dense vector column. It writes row-major little-endian float32 matrices and JSONL metadata sidecars with SHA-256 manifests. The source dataset, split, column names, dimensions, seed and projection mode are recorded.

## First implementation

`benchmarks/realworld/prepare_hf_embeddings.py`

Dependencies:

```powershell
py -m pip install "datasets>=3,<5" "numpy>=1.26" "scikit-learn>=1.4"
```

Example (replace dataset/column names with columns verified in the selected dataset card):

```powershell
py benchmarks/realworld/prepare_hf_embeddings.py `
  --dataset ORGANIZATION/DATASET `
  --split train `
  --vector-column embedding `
  --id-column id `
  --tenant-column tenant_id `
  --category-column category `
  --timestamp-column timestamp `
  --require-metadata `
  --rows 100000 `
  --dimensions 32,64,128,256 `
  --seed 3303 `
  --output data/hito3_3
```

Output per dimension:

- `corpus.f32`: [N_corpus, D] little-endian float32, row-major, no header.
- `queries.f32`: [N_query, D] same representation; held out before PCA fitting.
- `corpus_metadata.jsonl`, `query_metadata.jsonl`: row_id, external_id, tenant_id, category, timestamp.
- `manifest.json`: provenance, split sizes, normalization, PCA variance, checksums and metadata completeness.

The script is an ingestion/export stage. It does not yet claim to be an ALG runtime loader or a benchmark result. The next adapter must mmap these files, validate manifest dimensions/byte lengths/checksums, and translate metadata into deterministic filter bitmaps without touching Plan 5A.

## Dimensional battery

Fixed initial matrix: N_corpus = 90,000 (for 100,000 source rows with 10% held out), D = {32, 64, 128, 256}, TopK = 10 and 100, selectivity = 1% and 10%. Add 100% as a no-filter control. Keep query IDs disjoint from corpus IDs.

For each D, report:

- source/native vs PCA-projected status and explained variance;
- L2 norm distribution, centroid-radius distribution and pairwise-distance quantiles;
- covariance eigenvalue spectrum and effective rank;
- hubness (k-occurrence skew/Gini and maximum occurrence);
- partition-pruning ratio, exact distance evaluations, IDSetRecall, StrictRankMatch;
- ALG vs FAISS Flat COLD/WARM latency, p50/p95/p99, QPS and memory.

## Metadata filters

Benchmark independently and in combination:

- `tenant_id == X`
- `category == Y`
- `timestamp >= T`
- `tenant_id == X AND category == Y AND timestamp >= T`

Use identical materialized eligibility sets for ALG and FAISS. Record actual eligible count and selectivity per query. Do not assume independent columns when computing expected selectivity.

## Construction protocol

Five fresh-process repetitions per engine and N in {100k, 250k, 500k, 1M}; alternate engine order across repetitions. Separate allocation/reservation, vector ingestion, index training/build and post-build validation. Report median, mean, standard deviation and all raw samples. Do not use one run's timing as a build-time claim.

## Acceptance gates

- 100% IDSetRecall against binary64 ground truth for every query/filter/dimension.
- No silent metadata loss; required metadata completeness = 100% for fields used in a filter.
- Reproducible manifests and checksums.
- Native and projected datasets reported separately.
- Any regression at D=128/256 is reported, not hidden by aggregate averages.
- No change to Plan 5A kernels in this milestone.

## Important limitation

The previous `run_realistic_embeddings_hito32.cpp` creates a synthetic semantic-like distribution. It is useful as a controlled synthetic stress test, but it is not a real-world corpus. Do not use it as evidence for the real-world validation gate.
