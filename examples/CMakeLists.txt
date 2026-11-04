cmake_minimum_required(VERSION 3.14)
project(alg_sdk_examples LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

# Locate SDK root directory (parent of examples directory)
set(ALG_SDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

include_directories("${ALG_SDK_DIR}/include")

# Detect platform library directory
if(WIN32)
    set(ALG_LIB_DIR "${ALG_SDK_DIR}/lib/windows-x64")
    set(ALG_SHARED_NAME "alg")
    set(ALG_STATIC_NAME "libalg")
else()
    set(ALG_LIB_DIR "${ALG_SDK_DIR}/lib/linux-x64")
    set(ALG_SHARED_NAME "alg")
    set(ALG_STATIC_NAME "alg")
endif()

# Find static library
find_library(ALG_STATIC_LIB NAMES ${ALG_STATIC_NAME} libalg alg
    PATHS "${ALG_LIB_DIR}"
    NO_DEFAULT_PATH
    REQUIRED
)

# Find shared library / import library
find_library(ALG_SHARED_LIB NAMES ${ALG_SHARED_NAME} libalg_dll alg
    PATHS "${ALG_LIB_DIR}"
    NO_DEFAULT_PATH
)

# 1. C Consumer Example (demonstrates dynamic linkage if available, otherwise static)
add_executable(c_example c_example.c)
if(ALG_SHARED_LIB)
    target_link_libraries(c_example PRIVATE ${ALG_SHARED_LIB})
    target_compile_definitions(c_example PRIVATE ALG_USE_DLL=1)
else()
    target_link_libraries(c_example PRIVATE ${ALG_STATIC_LIB})
endif()

# 2. C++ Consumer Example (demonstrates static linkage)
add_executable(cpp_example cpp_example.cpp)
target_link_libraries(cpp_example PRIVATE ${ALG_STATIC_LIB})

if(WIN32 AND ALG_SHARED_LIB)
    # Copy DLL to target directory for execution
    add_custom_command(TARGET c_example POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ALG_LIB_DIR}/alg.dll"
        $<TARGET_FILE_DIR:c_example>
    )
endif()
