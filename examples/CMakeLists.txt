cmake_minimum_required(VERSION 3.14)
project(alg_sdk_examples LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Locate SDK root directory (parent of examples directory)
set(ALG_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories("${ALG_SDK_DIR}/include")

# Find static library
find_library(ALG_LIB NAMES alg libalg alg_static
    PATHS "${ALG_SDK_DIR}/lib/windows-x64" "${ALG_SDK_DIR}/lib/linux-x64"
    REQUIRED
)

# 1. C Consumer Example
add_executable(c_example c_example.c)
target_link_libraries(c_example PRIVATE ${ALG_LIB})

# 2. C++ Consumer Example
add_executable(cpp_example cpp_example.cpp)
target_link_libraries(cpp_example PRIVATE ${ALG_LIB})
