# Vendored FAISS Headers and Import Library

This directory contains headers and a Windows x64 import stub library for FAISS (Facebook AI Similarity Search), used solely for comparative benchmarking and numerical equivalence testing in `benchmarks/faiss_comparison/` and `tests/`.

The standalone production engine (`libalg.a`, `alg.dll`, `libalg.so`) and the distributed SDK in `dist/alg-vector-engine-sdk/` do NOT link against, include, or distribute any artifact from this directory.

---

## Provenance and Version Metadata

- **Upstream Repository:** https://github.com/facebookresearch/faiss
- **Release Tag:** [`v1.15.1`](https://github.com/facebookresearch/faiss/releases/tag/v1.15.1) (Release date: September 15, 2026)
- **Upstream Commit Hash:** `3bb4864b2260683416a49db21ebcb66b0a94464c`
- **Internal Version Macros:** `FAISS_VERSION_MAJOR 1`, `FAISS_VERSION_MINOR 15`, `FAISS_VERSION_PATCH 1` (verified in `include/faiss/Index.h`)
- **License:** MIT License (Meta Platforms, Inc. and affiliates). Full text in `LICENSE`.

---

## Artifact Inventory and Hashes

| File | Description | SHA-256 Hash |
| :--- | :--- | :--- |
| `faiss.def` | Module definition file (8,138 exported C++ symbols) | `E9437311BDD3640A155A7AE3C3C2EA492FA9EDF0AEFD02AEE51D4578C380ECB8` |
| `faiss.lib` | Windows x64 MSVC import library (stub for `faiss.dll`) | `A089148BC1294A2B78DCA4DEA5B7266C1198E4E6AAB3DC644EAB28883607EE12` |
| `include/faiss/` | C++ header tree from upstream tag `v1.15.1` | Matching upstream commit `3bb4864` |

---

## Binary Generation and Toolchain

The file `faiss.lib` is an import library that resolves external calls against a dynamic `faiss.dll` runtime during benchmark runs on Windows x64.

- **Toolchain:** Microsoft (R) Library Manager Version `14.12.25835.0` (MSVC C++ SDK x64).
- **Generation Command:**
  ```cmd
  lib.exe /nologo /def:faiss.def /machine:x64 /out:faiss.lib
  ```
- **Target Subsystem:** Windows x64 (AMD64 / x86_64).
