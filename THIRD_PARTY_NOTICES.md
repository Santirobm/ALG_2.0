# Third-Party Notices and Licenses

ALG Vector Engine includes or references the following third-party software components for testing, evaluation, and benchmarking purposes.

The standalone core library (`libalg`) does not link against or depend on any of these third-party components in production.

---

## 1. FAISS (Facebook AI Similarity Search)

- Directory: `third_party/faiss/`
- Upstream: https://github.com/facebookresearch/faiss
- Purpose: Benchmarking, comparative baseline measurements, and numerical equivalence verification.
- License: MIT License

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

## 2. Evaluation Datasets

- Directory: `data/`
- Upstream: Embeddings derived from Simple English Wikipedia dump using `sentence-transformers/all-MiniLM-L6-v2`.
- Purpose: Real-world vector distribution validation across research and historical ablation benchmarks.
- Licenses: Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0) / Apache License 2.0.
