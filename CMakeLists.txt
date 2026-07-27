cmake_minimum_required(VERSION 3.22)
project(F1Pulse CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GGML_CUDA OFF CACHE BOOL "Build with CUDA support" FORCE)

set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "Build llama.cpp examples" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "Build llama.cpp tests" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "Build llama.cpp server" FORCE)

set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)

if(MINGW)
    add_compile_definitions(_WIN32_WINNT=0x0601)
endif()

add_subdirectory(third_party/llama.cpp)

set(SOURCES
    src/main.cpp
    src/math/VectorMath.cpp
    src/storage/Database.cpp
    src/ai/LlamaEngine.cpp
)

add_executable(f1_pulse ${SOURCES})

target_include_directories(f1_pulse PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/include"
)

target_link_libraries(f1_pulse PRIVATE llama ggml)

target_compile_features(f1_pulse PRIVATE cxx_std_20)

if(MSVC)
    set_target_properties(f1_pulse PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
endif()