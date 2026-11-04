#!/usr/bin/env bash
# Copyright (c) 2026 Santiago Robledo.
# SPDX-License-Identifier: Apache-2.0

set -e

echo "============================================================"
echo "Compiling Standalone ALG Vector Engine Consumer Examples (Linux)"
echo "============================================================"

SDK_DIR=".."
INC="-I${SDK_DIR}/include"
LIB_DIR="${SDK_DIR}/lib/linux-x64"

mkdir -p bin

CC="${CC:-gcc}"
CXX="${CXX:-g++}"

echo "[1/2] Compiling C example (c_example.c) linking dynamically against libalg.so..."
${CC} -O3 -std=c11 ${INC} c_example.c -L"${LIB_DIR}" -lalg -Wl,-rpath,"${LIB_DIR}" -o bin/c_example

echo "[2/2] Compiling C++ example (cpp_example.cpp) linking statically against libalg.a..."
${CXX} -O3 -std=c++17 ${INC} cpp_example.cpp "${LIB_DIR}/libalg.a" -o bin/cpp_example

echo ""
echo "Standalone examples built successfully in bin/!"
echo ""
echo "[Execution 1/2] Running c_example (Dynamic Linkage)..."
./bin/c_example
echo ""
echo "[Execution 2/2] Running cpp_example (Static Linkage)..."
./bin/cpp_example
