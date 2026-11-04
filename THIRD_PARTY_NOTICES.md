# Third-Party Notices and Licenses

ALG Vector Engine includes or references the following third-party software components and evaluation datasets for testing, evaluation, and benchmarking purposes.

The standalone core library (`libalg`) does not link against, include, or depend on any of these third-party components or datasets in production.

---

## 1. FAISS (Facebook AI Similarity Search)

- Directory: `third_party/faiss/`
- Upstream Repository: https://github.com/facebookresearch/faiss
- Version: 1.15.1 (defined by `FAISS_VERSION_MAJOR 1`, `MINOR 15`, `PATCH 1` in `include/faiss/Index.h`)
- License: MIT License
- Purpose: Benchmarking, comparative baseline measurements, and numerical equivalence verification in `benchmarks/faiss_comparison/` and `tests/`.

### Binary Provenance (`faiss.lib`)
- The file `third_party/faiss/faiss.lib` is a 64-bit Windows import library (stub) generated from `third_party/faiss/faiss.def` using the MSVC library manager (`lib.exe /def:faiss.def /machine:x64 /out:faiss.lib`).
- It contains symbol export references for linking benchmark executables against an external dynamic `faiss.dll` runtime on Windows.
- It is NOT included in or linked by the distributed SDK (`dist/alg-vector-engine-sdk`).

### MIT License Text

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## 2. Evaluation Datasets and Embeddings

- Directory: `data/`
- Detailed documentation: `data/README.md`
- Purpose: Real-world vector distribution validation across research and historical ablation benchmarks (Milestones 3.2 and 3.3).

The evaluation fixtures separate into three components with distinct licensing terms:

### A. Original Source Text (Simple English Wikipedia)
- Content: Article titles and text from Simple English Wikipedia dump (2023).
- License: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) and GNU Free Documentation License (GFDL).
- Copyright: Wikimedia Foundation and individual Wikipedia contributors.

### B. Embedding Model (sentence-transformers/all-MiniLM-L6-v2)
- Upstream: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Authors: Nils Reimers and Sentence-Transformers contributors.
- License: Apache License, Version 2.0.

### C. Derived Embeddings and PCA Projections (`data/hito3_3/`)
- Generation: 384-dimensional embeddings sourced from `lsb/simplewiki2023-all-minilm-l6-v2-embedding`, projected to dimensions 32, 64, 128, and 256 using `IncrementalPCA` via `benchmarks/realworld/prepare_hf_embeddings.py`.
- Status: Binary test fixtures (`corpus.f32`, `queries.f32`) and JSONL metadata used exclusively for offline accuracy and latency testing. Not bundled into the production SDK or the `libalg` library.
- Redistribution: Any distribution of the derived files that includes Wikipedia titles or text snippets is subject to CC BY-SA 3.0 attribution requirements.
