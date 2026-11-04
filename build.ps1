# Build script for ALG Vector Engine (Windows x64 & Linux x64 Cross-Compilation)
param (
    [switch]$RunTests = $true,
    [switch]$RunBenchmark = $false,
    [switch]$Package = $true
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
Write-Host "Building ALG Vector Engine Multi-Platform Artifacts..." -ForegroundColor Cyan
Write-Host "Compiler: $ZIG" -ForegroundColor DarkGray
Write-Host "============================================================" -ForegroundColor Cyan

$BinDir = "bin"
$LinuxBinDir = "bin/linux-x64"
if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Path $BinDir | Out-Null }
if (-not (Test-Path $LinuxBinDir)) { New-Item -ItemType Directory -Path $LinuxBinDir | Out-Null }

$Inc = "-Iinclude", "-Isrc"

# --- Part 1: Windows x64 Build ---
Write-Host "`n[1/8] Compiling Windows x64 AVX2+FMA kernel..." -ForegroundColor Yellow
& $ZIG c++ -O3 -std=c++17 -mavx2 -mfma $Inc -c src/kernels/simd_avx2.cpp -o "$BinDir/simd_avx2.o"

Write-Host "[2/8] Compiling Windows x64 platform and core components..." -ForegroundColor Yellow
& $ZIG c++ -O3 -std=c++17 $Inc -c src/kernels/simd_scalar.cpp -o "$BinDir/simd_scalar.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/platform/cpu_features.cpp -o "$BinDir/cpu_features.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/platform/allocator.cpp -o "$BinDir/allocator.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/core/alg_index.cpp -o "$BinDir/alg_index.o"
& $ZIG c++ -O3 -std=c++17 $Inc -c src/core/alg_search.cpp -o "$BinDir/alg_search.o"

Write-Host "[3/8] Archiving Windows static library bin/libalg.a..." -ForegroundColor Yellow
$WinObjs = @(
    "$BinDir/simd_avx2.o",
    "$BinDir/simd_scalar.o",
    "$BinDir/cpu_features.o",
    "$BinDir/allocator.o",
    "$BinDir/alg_index.o",
    "$BinDir/alg_search.o"
)
& $ZIG ar rcs "$BinDir/libalg.a" @WinObjs

Write-Host "[4/8] Linking Windows shared library bin/alg.dll..." -ForegroundColor Yellow
& $ZIG c++ -shared -O3 $WinObjs -o "$BinDir/alg.dll" "-Wl,--out-implib,$BinDir/libalg_dll.a"

# --- Part 2: Linux x64 Cross-Compilation ---
Write-Host "`n[5/8] Cross-compiling Linux x64 objects with -fPIC..." -ForegroundColor Yellow
& $ZIG c++ -target x86_64-linux-gnu -O3 -std=c++17 -mavx2 -mfma $Inc -fPIC -c src/kernels/simd_avx2.cpp -o "$LinuxBinDir/simd_avx2.o"
& $ZIG c++ -target x86_64-linux-gnu -O3 -std=c++17 $Inc -fPIC -c src/kernels/simd_scalar.cpp -o "$LinuxBinDir/simd_scalar.o"
& $ZIG c++ -target x86_64-linux-gnu -O3 -std=c++17 $Inc -fPIC -c src/platform/cpu_features.cpp -o "$LinuxBinDir/cpu_features.o"
& $ZIG c++ -target x86_64-linux-gnu -O3 -std=c++17 $Inc -fPIC -c src/platform/allocator.cpp -o "$LinuxBinDir/allocator.o"
& $ZIG c++ -target x86_64-linux-gnu -O3 -std=c++17 $Inc -fPIC -c src/core/alg_index.cpp -o "$LinuxBinDir/alg_index.o"
& $ZIG c++ -target x86_64-linux-gnu -O3 -std=c++17 $Inc -fPIC -c src/core/alg_search.cpp -o "$LinuxBinDir/alg_search.o"

Write-Host "[6/8] Linking Linux x64 libraries bin/linux-x64/libalg.so and libalg.a..." -ForegroundColor Yellow
$LinuxObjs = @(
    "$LinuxBinDir/simd_avx2.o",
    "$LinuxBinDir/simd_scalar.o",
    "$LinuxBinDir/cpu_features.o",
    "$LinuxBinDir/allocator.o",
    "$LinuxBinDir/alg_index.o",
    "$LinuxBinDir/alg_search.o"
)
& $ZIG ar rcs "$LinuxBinDir/libalg.a" @LinuxObjs
& $ZIG c++ -target x86_64-linux-gnu -shared -O3 $LinuxObjs -o "$LinuxBinDir/libalg.so"

# --- Part 3: Windows Binaries & Tests ---
Write-Host "`n[7/8] Building unit test suite..." -ForegroundColor Yellow
& $ZIG c++ -O3 -std=c++17 $Inc tests/test_engine_unit.cpp "$BinDir/libalg.a" -o "$BinDir/test_engine_unit.exe"

Write-Host "[8/8] Building examples and benchmark CLI..." -ForegroundColor Yellow
& $ZIG cc -O3 -std=c11 $Inc examples/c_example.c "$BinDir/libalg_dll.a" -o "$BinDir/c_example.exe"
& $ZIG c++ -O3 -std=c++17 $Inc examples/cpp_example.cpp "$BinDir/libalg.a" -o "$BinDir/cpp_example.exe"
& $ZIG c++ -O3 -std=c++17 $Inc tools/alg_benchmark_cli.cpp "$BinDir/libalg.a" -o "$BinDir/alg_benchmark_cli.exe"

Write-Host "`nBuild completed successfully for Windows x64 and Linux x64!" -ForegroundColor Green

if ($RunTests) {
    Write-Host "`n--- Running Certified Unit & Regression Test Suite ---" -ForegroundColor Cyan
    & "$BinDir/test_engine_unit.exe"
}

if ($RunBenchmark) {
    Write-Host "`n--- Running Benchmark Harness ---" -ForegroundColor Cyan
    & "$BinDir/alg_benchmark_cli.exe"
}

if ($Package) {
    Write-Host "`n--- Packaging Multi-Platform SDK Distribution (dist/) ---" -ForegroundColor Cyan
    $DistRoot = "dist/alg-vector-engine-sdk"
    if (Test-Path $DistRoot) {
        Remove-Item -Recurse -Force $DistRoot
    }
    New-Item -ItemType Directory -Force -Path "$DistRoot/include", "$DistRoot/lib/windows-x64", "$DistRoot/lib/linux-x64", "$DistRoot/examples" | Out-Null

    # Headers
    Copy-Item "include/alg_engine.h" -Destination "$DistRoot/include/alg_engine.h"

    # Windows x64
    Copy-Item "$BinDir/alg.dll" -Destination "$DistRoot/lib/windows-x64/alg.dll"
    Copy-Item "$BinDir/libalg_dll.a" -Destination "$DistRoot/lib/windows-x64/libalg_dll.a"
    Copy-Item "$BinDir/libalg.a" -Destination "$DistRoot/lib/windows-x64/libalg.a"

    # Linux x64
    Copy-Item "$LinuxBinDir/libalg.so" -Destination "$DistRoot/lib/linux-x64/libalg.so"
    Copy-Item "$LinuxBinDir/libalg.a" -Destination "$DistRoot/lib/linux-x64/libalg.a"

    # Examples & build scripts
    Copy-Item "examples/c_example.c" -Destination "$DistRoot/examples/c_example.c"
    Copy-Item "examples/cpp_example.cpp" -Destination "$DistRoot/examples/cpp_example.cpp"
    Copy-Item "examples/CMakeLists.txt" -Destination "$DistRoot/examples/CMakeLists.txt"
    Copy-Item "examples/build_examples.bat" -Destination "$DistRoot/examples/build_examples.bat"
    Copy-Item "examples/build_examples.sh" -Destination "$DistRoot/examples/build_examples.sh"

    # Metadata & documentation
    Copy-Item "README.md" -Destination "$DistRoot/README.md"
    Copy-Item "LICENSE" -Destination "$DistRoot/LICENSE"
    Copy-Item "CHANGELOG.md" -Destination "$DistRoot/CHANGELOG.md"

    Write-Host "Multi-Platform SDK packaged cleanly at $DistRoot" -ForegroundColor Green
}
