cmake_minimum_required(VERSION 3.14)
project(alg_vector_engine VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckCXXCompilerFlag)

# Common include directories
set(ALG_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Core source files (portable baseline ISA)
set(ALG_CORE_SOURCES
    src/platform/cpu_features.cpp
    src/platform/allocator.cpp
    src/kernels/simd_scalar.cpp
    src/core/alg_index.cpp
    src/core/alg_search.cpp
)

# SIMD AVX2 kernel source (isolated compilation unit)
set(ALG_AVX2_SOURCES
    src/kernels/simd_avx2.cpp
)

# Apply AVX2 and FMA flags specifically to simd_avx2.cpp
if(MSVC)
    set_source_files_properties(${ALG_AVX2_SOURCES} PROPERTIES COMPILE_FLAGS "/arch:AVX2")
else()
    set_source_files_properties(${ALG_AVX2_SOURCES} PROPERTIES COMPILE_FLAGS "-mavx2 -mfma -O3")
endif()

# 1. Static Library: alg_static
add_library(alg_static STATIC ${ALG_CORE_SOURCES} ${ALG_AVX2_SOURCES})
target_include_directories(alg_static PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${ALG_INCLUDE_DIRS}
)
target_compile_definitions(alg_static PUBLIC ALG_STATIC=1)

# 2. Shared Library: alg_shared (DLL on Windows)
add_library(alg_shared SHARED ${ALG_CORE_SOURCES} ${ALG_AVX2_SOURCES})
target_include_directories(alg_shared PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE ${ALG_INCLUDE_DIRS}
)
target_compile_definitions(alg_shared PRIVATE ALG_BUILD_DLL=1)
set_target_properties(alg_shared PROPERTIES OUTPUT_NAME "alg")

# 3. Test Suite
add_executable(test_engine_unit tests/test_engine_unit.cpp)
target_link_libraries(test_engine_unit PRIVATE alg_static)
target_include_directories(test_engine_unit PRIVATE ${ALG_INCLUDE_DIRS})

# 4. Examples
add_executable(c_example examples/c_example.c)
target_link_libraries(c_example PRIVATE alg_static)
target_include_directories(c_example PRIVATE ${ALG_INCLUDE_DIRS})

add_executable(cpp_example examples/cpp_example.cpp)
target_link_libraries(cpp_example PRIVATE alg_static)
target_include_directories(cpp_example PRIVATE ${ALG_INCLUDE_DIRS})

# 5. CLI Benchmark Harness
add_executable(alg_benchmark_cli tools/alg_benchmark_cli.cpp)
target_link_libraries(alg_benchmark_cli PRIVATE alg_static)
target_include_directories(alg_benchmark_cli PRIVATE ${ALG_INCLUDE_DIRS})

# Enable CTest
enable_testing()
add_test(NAME EngineUnitTests COMMAND test_engine_unit)
