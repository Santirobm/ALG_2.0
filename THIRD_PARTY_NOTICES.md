# Third-Party Notices and Licenses

ALG Vector Engine includes or references the following third-party software components and evaluation datasets for testing, evaluation, and benchmarking purposes.

The standalone core library (`libalg`) does not link against, include, or depend on any of these third-party components or datasets in production.

---

## 1. FAISS (Facebook AI Similarity Search)

- Directory: `third_party/faiss/`
- Documentation: `third_party/faiss/README.md`
- Upstream Repository: https://github.com/facebookresearch/faiss
- Release Tag: [`v1.15.1`](https://github.com/facebookresearch/faiss/releases/tag/v1.15.1)
- Release Date: September 15, 2026
- Upstream Tag Commit SHA: `75c755c5a1fc854756aaf3bd1131213aeb21d8d1`
- License: MIT License (Meta Platforms, Inc. and affiliates)
- Purpose: Benchmarking, comparative baseline measurements, and numerical equivalence verification in `benchmarks/faiss_comparison/` and `tests/`.

### Binary Provenance (`faiss.lib`)
- `third_party/faiss/faiss.def`: Module definition file containing 8,138 exported symbols for Windows x64.
  - SHA-256: `E9437311BDD3640A155A7AE3C3C2EA492FA9EDF0AEFD02AEE51D4578C380ECB8`
- `third_party/faiss/faiss.lib`: Windows x64 MSVC import library generated from `faiss.def` using `lib.exe /nologo /def:faiss.def /machine:x64 /out:faiss.lib` (Microsoft (R) Library Manager Version 14.12.25835.0).
  - SHA-256: `A089148BC1294A2B78DCA4DEA5B7266C1198E4E6AAB3DC644EAB28883607EE12`
- Role: Resolves external calls against an external dynamic `faiss.dll` runtime during benchmark runs on Windows x64. It is NOT linked by or distributed with the production SDK (`dist/alg-vector-engine-sdk`).

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
- Detailed documentation and hash inventory: `data/README.md`
- Source Dataset (Hugging Face): [`lsb/simplewiki2023-all-minilm-l6-v2-embedding`](https://huggingface.co/datasets/lsb/simplewiki2023-all-minilm-l6-v2-embedding)
- Immutable Revision (Commit SHA): `14270bcfcf2b7854cddce4395c0b38a199edef05`
- Download Timestamp: `2026-11-03T16:05:00-03:00`
- Purpose: Real-world vector distribution validation across research and historical ablation benchmarks (Milestones 3.2 and 3.3).

### Reproducible Export Command
```bash
python benchmarks/realworld/prepare_hf_embeddings.py \
  --dataset lsb/simplewiki2023-all-minilm-l6-v2-embedding \
  --revision 14270bcfcf2b7854cddce4395c0b38a199edef05 \
  --vector-column embedding \
  --id-column id_int \
  --rows 100000 \
  --query-fraction 0.10 \
  --seed 3303 \
  --dimensions 32,64,128,256 \
  --output data/hito3_3
```

### Intellectual Property & Data Scope
- **Data Scope:** The exported fixtures do not contain source article text, titles, or URLs. They retain numerical embeddings derived from the upstream dataset. The licensing status of derived embeddings should be assessed separately from the licensing of the original textual content.
- **Source Text:** The underlying text from Simple English Wikipedia is licensed under CC BY-SA 3.0 / GFDL.
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` is licensed under the Apache License 2.0.
- **Derived Arrays:** Binary float32 projections (`corpus.f32`, `queries.f32`) generated via IncrementalPCA used exclusively for testing recall and latency. They are excluded from git distribution and are not bundled into the distributed SDK.
