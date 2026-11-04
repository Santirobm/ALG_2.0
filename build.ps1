# Build script for ALG Vector Engine using Zig C++ on Windows
param (
    [switch]$RunTests = $true,
    [switch]$RunBenchmark = $false
)

$ErrorActionPreference = "Stop"

$ZIG = (Get-Command zig.exe -ErrorAction SilentlyContinue).Source
if (-not $ZIG) {
    $ZIG = "$env:LOCALAPPDATA\Microsoft\WinGet\Links\zig.exe"
}
if (-not (Test-Path $ZIG)) {
    Write-Error "zig.exe not found in PATH or WinGet Links."
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Building ALG Vector Engine with Zig Clang..." -ForegroundColor Cyan
Write-Host "Compiler: $ZIG" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

$BinDir = "bin"
if (-not (Test-Path $BinDir)) {
    New-Item -ItemType Directory -Path $BinDir | Out-Null
}

$Inc = "-Iinclude", "-Isrc"

# 1. Compile isolated AVX2+FMA kernel
Write-Host "[1/6] Compiling AVX2+FMA kernel..." -ForegroundColor Yellow
& $ZIG c++ -O3 -std=c++17 -mavx2 -mfma $Inc -c src/kernels/simd_avx2.cpp -o "$BinDir/simd_avx2.o"

# 2. Compile portable baseline sources
Write-Host "[2/6] Compiling platform and core components..." -ForegroundColor Yellow
& $ZIG c++ -O3 -std=c++17 $Inc -c src/kernels/simd_scalar.cpp -o "$BinDir/simd_scalar.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/platform/cpu_features.cpp -o "$BinDir/cpu_features.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/platform/allocator.cpp -o "$BinDir/allocator.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/core/alg_index.cpp -o "$BinDir/alg_index.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/core/alg_search.cpp -o "$BinDir/alg_search.o"

# 3. Create static library libalg.a
Write-Host "[3/6] Archiving static library bin/libalg.a..." -ForegroundColor Yellow
$Objs = @(
    "$BinDir/simd_avx2.o",
    "$BinDir/simd_scalar.o",
    "$BinDir/cpu_features.o",
    "$BinDir/allocator.o",
    "$BinDir/alg_index.o",
    "$BinDir/alg_search.o"
)
& $ZIG ar rcs "$BinDir/libalg.a" @Objs

# 4. Create shared library / DLL bin/alg.dll
Write-Host "[4/6] Linking shared library bin/alg.dll..." -ForegroundColor Yellow
& $ZIG c++ -shared -O3 $Objs -o "$BinDir/alg.dll" "-Wl,--out-implib,$BinDir/libalg_dll.a"

# 5. Build test suite
Write-Host "[5/6] Building test suite..." -ForegroundColor Yellow
& $ZIG c++ -O3 -std=c++17 $Inc tests/test_engine_unit.cpp "$BinDir/libalg.a" -o "$BinDir/test_engine_unit.exe"

# 6. Build examples & benchmark CLI (testing both static and DLL linkage)
Write-Host "[6/6] Building examples and benchmark CLI..." -ForegroundColor Yellow
& $ZIG cc -O3 -std=c11 $Inc examples/c_example.c "$BinDir/libalg_dll.a" -o "$BinDir/c_example.exe"
& $ZIG c++ -O3 -std=c++17 $Inc examples/cpp_example.cpp "$BinDir/libalg.a" -o "$BinDir/cpp_example.exe"
& $ZIG c++ -O3 -std=c++17 $Inc tools/alg_benchmark_cli.cpp "$BinDir/libalg.a" -o "$BinDir/alg_benchmark_cli.exe"

Write-Host "`nBuild completed successfully!" -ForegroundColor Green

if ($RunTests) {
    Write-Host "`n--- Running Engine Unit Test Suite ---" -ForegroundColor Cyan
    & "$BinDir/test_engine_unit.exe"
}

if ($RunBenchmark) {
    Write-Host "`n--- Running Benchmark Harness ---" -ForegroundColor Cyan
    & "$BinDir/alg_benchmark_cli.exe"
}
