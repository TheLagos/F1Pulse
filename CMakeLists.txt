cmake_minimum_required(VERSION 3.22)
project(F1Pulse CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_CUDA ON CACHE BOOL "Build with CUDA support" FORCE)

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)

add_subdirectory(third_party/llama.cpp)

set(SOURCES
    src/main.cpp
    src/math/VectorMath.cpp
    src/storage/Database.cpp
)

add_executable(f1_pulse ${SOURCES})

target_include_directories(f1_pulse PRIVATE src)

target_link_libraries(f1_pulse PRIVATE llama)

target_compile_features(f1_pulse PRIVATE cxx_std_20)