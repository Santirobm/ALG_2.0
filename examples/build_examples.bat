@echo off
setlocal
echo ============================================================
echo Compiling Standalone ALG Vector Engine Consumer Examples
echo ============================================================

set SDK_DIR=..
set INC=-I%SDK_DIR%/include
set LIB_DIR=%SDK_DIR%/lib/windows-x64

if not exist bin mkdir bin

echo [1/2] Compiling C example (c_example.c) linking dynamically against alg.dll...
zig cc -O3 -std=c11 %INC% c_example.c "%LIB_DIR%/libalg_dll.a" -o bin/c_example.exe
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed for c_example.c
    exit /b 1
)

echo [2/2] Compiling C++ example (cpp_example.cpp) linking statically against libalg.a...
zig c++ -O3 -std=c++17 %INC% cpp_example.cpp "%LIB_DIR%/libalg.a" -o bin/cpp_example.exe
if %ERRORLEVEL% NEQ 0 (
    echo Compilation failed for cpp_example.cpp
    exit /b 1
)

copy "%LIB_DIR%\alg.dll" bin\ >nul 2>&1

echo.
echo Standalone examples built successfully in bin/!
echo.
echo [Execution 1/2] Running c_example.exe (Dynamic Linkage)...
bin\c_example.exe
echo.
echo [Execution 2/2] Running cpp_example.exe (Static Linkage)...
bin\cpp_example.exe
