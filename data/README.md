# Evaluation and Benchmark Datasets

The files in this directory are offline evaluation fixtures and PCA-projected embedding arrays used exclusively for algorithmic research, regression testing, and performance benchmarking (Milestones 3.2 and 3.3).

These data files are NOT part of the production library (`libalg`) and are NOT distributed with the ALG Vector Engine SDK (`dist/alg-vector-engine-sdk`).

---

## 1. Provenance and Immutable Dataset Reference

- **Source Dataset (Hugging Face):** [`lsb/simplewiki2023-all-minilm-l6-v2-embedding`](https://huggingface.co/datasets/lsb/simplewiki2023-all-minilm-l6-v2-embedding)
- **Immutable Revision (Commit SHA):** `14270bcfcf2b7854cddce4395c0b38a199edef05`
- **Download and Ingestion Timestamp:** `2026-11-03T16:05:00-03:00`
- **Reproducible Export Command:**
  ```bash
  python benchmarks/realworld/prepare_hf_embeddings.py \
    --dataset lsb/simplewiki2023-all-minilm-l6-v2-embedding \
    --id-column id_int \
    --rows 100000 \
    --query-fraction 0.10 \
    --seed 3303 \
    --dimensions 32,64,128,256 \
    --output data/hito3_3
  ```

---

## 2. Ingestion Schema vs. Retained Data

The upstream Hugging Face dataset contains 225,332 records with the following schema:
- `id` (`string`): Wikipedia article identifier.
- `url` (`string`): Full URL to Simple English Wikipedia article.
- `title` (`string`): Article title.
- `text` (`string`): Article plain text body.
- `embedding` (`list[float32]`, dimension 384): Sentence-Transformers dense embedding.
- `id_int` (`int64`): Numerical integer identifier.

### Data Retained in Local Fixtures
During export via `prepare_hf_embeddings.py`, **all natural language text, article bodies, titles, and URLs were explicitly discarded**:
- Only the first 100,000 rows were loaded and partitioned into 90,000 corpus rows and 10,000 held-out query rows with seed `3303`.
- By passing `--id-column id_int`, the `external_id` field in `corpus_metadata.jsonl` and `query_metadata.jsonl` strictly stores the 64-bit integer ID (`id_int`):
  ```json
  {"row_id": 13752, "external_id": 50883, "tenant_id": null, "category": null, "timestamp": null}
  ```
- No textual data from Simple English Wikipedia is persisted on disk or redistributed in this repository.

---

## 3. Git Tracking Policy and Cryptographic Manifests

To keep the repository lightweight and avoid committing hundreds of megabytes of raw binary files, `.gitignore` excludes the data fixtures:
```gitignore
data/**/*.f32
data/**/*.u32
data/**/*.jsonl
!data/**/manifest.json
```

The cryptographic proof and parameter configuration of the fixtures are preserved and committed to Git in the `manifest.json` files for each dimension.

### Dimensional Projection and Hash Inventory

The original 384-dimensional embeddings were projected to lower dimensional subspaces using `sklearn.decomposition.IncrementalPCA` (fitted strictly on the 90,000 corpus vectors, then applied to the 10,000 query vectors):

#### Metadata Files (Common to all dimensions)
- `corpus_metadata.jsonl`: `e57f05309281379e1aa23451eb23e72c1106f013928f23737e70865809713abb` (90,000 rows)
- `query_metadata.jsonl`: `970864f989d133d8c32e3cf1c7c3cf3ba74fff079ef08a7cfcd0d5a48571bd31` (10,000 rows)

#### Dimension D=32 (`data/hito3_3/D32/manifest.json`)
- `corpus.f32`: `bbf7d152243c9c68f76cfa45374e718461c57e4a503da39534c7318713f382c8` (11,520,000 bytes, 90,000 rows)
- `queries.f32`: `c03db0c4d27a17d9a10187c3d04858ed0130b71b6f6716ccc3148aa3a7d8f1f1` (1,280,000 bytes, 10,000 rows)
- PCA Explained Variance Ratio: `39.76%`

#### Dimension D=64 (`data/hito3_3/D64/manifest.json`)
- `corpus.f32`: `f1f541f63bf6cacd82418edaa76b8e091e83f94ec574327794181665f9cabd6e` (23,040,000 bytes, 90,000 rows)
- `queries.f32`: `56f49b15a7ae10fcf0087c77c8ca6bf5a60bde60842169004d91e383b36e89cd` (2,560,000 bytes, 10,000 rows)
- PCA Explained Variance Ratio: `58.28%`

#### Dimension D=128 (`data/hito3_3/D128/manifest.json`)
- `corpus.f32`: `f7e5bd42c2e67bfad4e599ec4b2dddde0d23f0f9ed430ceda053343c8c6575c5` (46,080,000 bytes, 90,000 rows)
- `queries.f32`: `ebccc78d1374ff02f0bb0d08b3a57c90c5377e28d34c1affd1c33b6340fa12cb` (5,120,000 bytes, 10,000 rows)
- PCA Explained Variance Ratio: `79.61%`

#### Dimension D=256 (`data/hito3_3/D256/manifest.json`)
- `corpus.f32`: `074ea70b57b9dd584e1e989c997ba631e447365cd26d59de5324492922365fb9` (92,160,000 bytes, 90,000 rows)
- `queries.f32`: `7737e59fec5eac448d0b3d527471dce9a1655f2f16d9b32e5f4f10969d887847` (10,240,000 bytes, 10,000 rows)
- PCA Explained Variance Ratio: `97.31%`

---

## 4. Legal Status and Intellectual Property Assessment

The exported fixtures do not contain source article text, titles, or URLs. They retain numerical embeddings derived from the upstream dataset. The licensing status of derived embeddings should be assessed separately from the licensing of the original textual content:

1. **Source Text (Simple English Wikipedia):**  
   The underlying text used to compute embeddings belongs to Wikimedia contributors under CC BY-SA 3.0 / GFDL.
2. **Inference Model:**  
   The embeddings were calculated upstream using `sentence-transformers/all-MiniLM-L6-v2`, distributed under the **Apache License 2.0** by Nils Reimers and contributors.
3. **Derived Float32 Vector Arrays:**  
   The `.f32` files are numerical mathematical transformations (projections) generated strictly as benchmarking test fixtures to measure search recall and latency. They are independent of the core library code and carry no proprietary encumbrance affecting the commercial acquisition or licensing of ALG Vector Engine.
