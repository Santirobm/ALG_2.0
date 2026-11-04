# Evaluation and Benchmark Datasets

The files in this directory are evaluation fixtures and synthetic/projected embeddings used exclusively for historical research, regression testing, and performance benchmarking (Milestones 3.2 and 3.3).

These data files are NOT part of the production library (`libalg`) and are NOT distributed with the ALG Vector Engine SDK (`dist/alg-vector-engine-sdk`).

---

## Provenance and Licensing Breakdown

The real-world evaluation fixtures located in `data/hito3_3/` are composed of three distinct layers with separate licensing considerations:

### 1. Source Text Corpus: Simple English Wikipedia
- Source: Dump of Simple English Wikipedia articles (2023).
- License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) and GNU Free Documentation License (GFDL).
- Copyright: Wikimedia Foundation and Wikipedia contributors.

### 2. Embedding Model: sentence-transformers/all-MiniLM-L6-v2
- Upstream: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Authors: Nils Reimers and Sentence-Transformers contributors.
- License: Apache License, Version 2.0.

### 3. Derived Embedding Vectors and Projections
- Origin: Generated from the Hugging Face dataset `lsb/simplewiki2023-all-minilm-l6-v2-embedding` via `benchmarks/realworld/prepare_hf_embeddings.py`.
- Transformations: 384-dimensional embeddings projected to dimensions $D \in \{32, 64, 128, 256\}$ using `sklearn.decomposition.IncrementalPCA` (fitted strictly on corpus rows, then evaluated on query rows).
- Formats:
  - `corpus.f32`, `queries.f32`: Raw binary IEEE-754 float32 contiguous arrays (row-major).
  - `corpus_metadata.jsonl`, `query_metadata.jsonl`: Document identifiers and Wikipedia titles.
- Redistribution Notice: Used solely as test fixtures for engine recall and speed benchmarking. Any commercial distribution or reuse of these derived files must comply with CC BY-SA 3.0 attribution requirements for the underlying Wikipedia article titles and content.
