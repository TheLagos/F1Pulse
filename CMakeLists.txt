cmake_minimum_required(VERSION 3.22)

project(F1Pulse LANGUAGES C CXX)

# The C++ standard itself is requested per target (cxx_std_23 in f1_add_module).
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# User-facing options
# ---------------------------------------------------------------------------
set(F1_GPU_BACKEND "CPU" CACHE STRING "GPU backend for llama.cpp: CPU, VULKAN, CUDA or METAL")
set_property(CACHE F1_GPU_BACKEND PROPERTY STRINGS CPU VULKAN CUDA METAL)

option(F1_BUILD_TESTS "Build F1Pulse unit tests" OFF)
option(F1_WARNINGS_AS_ERRORS "Treat compiler warnings in F1Pulse code as errors" OFF)

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(Warnings)
include(F1Module)
include(Backends)

# ---------------------------------------------------------------------------
# llama.cpp (third_party/llama.cpp, not modified)
#
# Options are set as ordinary variables, not cache entries: the submodule
# requires CMake 3.14...3.28, so CMP0077 is NEW and option() honours them
# without writing anything into the cache. Names verified against
# third_party/llama.cpp/CMakeLists.txt and ggml/CMakeLists.txt.
# ---------------------------------------------------------------------------

# Link llama/ggml statically: no DLLs to copy next to f1_pulse.exe.
set(BUILD_SHARED_LIBS OFF)

set(LLAMA_BUILD_COMMON OFF)   # common/ + vendor/cpp-httplib (OpenSSL) are not needed
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_TOOLS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_APP OFF)
set(LLAMA_BUILD_UI OFF)       # Web UI may download a prebuilt bundle
set(LLAMA_OPENSSL OFF)        # no HTTPS/OpenSSL dependency

set(GGML_BUILD_TESTS OFF)
set(GGML_BUILD_EXAMPLES OFF)

add_subdirectory(third_party/llama.cpp)

# ---------------------------------------------------------------------------
# Project sources
# ---------------------------------------------------------------------------
if(F1_BUILD_TESTS)
    enable_testing()
endif()

add_subdirectory(src)
