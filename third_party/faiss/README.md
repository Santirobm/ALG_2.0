# Vendored FAISS Headers and Import Library

Upstream: https://github.com/facebookresearch/faiss
Version: 1.15.1 (defined by FAISS_VERSION_MAJOR 1, MINOR 15, PATCH 1 in include/faiss/Index.h)
License: MIT (see LICENSE in this directory)
Purpose: Benchmarking and baseline verification in benchmarks/ and tests/

## Artifacts in this directory

- `include/faiss/`: C++ header files from upstream FAISS 1.15.1.
- `faiss.def`: Module definition file declaring 8,138 exported symbols for Windows x64 targeting `faiss.dll`.
- `faiss.lib`: MSVC x64 import library generated from `faiss.def` (`lib /def:faiss.def /machine:x64 /out:faiss.lib`).

## Binary Provenance and Build Configuration

- Architecture: x86_64 (Windows x64)
- Generation method: MSVC `lib.exe` import library generated from the exported symbol table (`faiss.def`).
- Role: Allows native C++ test harnesses in `tests/` and `benchmarks/faiss_comparison/` to resolve FAISS symbols against a dynamic `faiss.dll` runtime during comparative benchmark execution.
- Production isolation: The standalone production library `libalg` (`libalg.a`, `alg.dll`, `libalg.so`) and the distributed SDK in `dist/alg-vector-engine-sdk/` DO NOT link, include, or distribute `faiss.lib` or any headers from this directory.
