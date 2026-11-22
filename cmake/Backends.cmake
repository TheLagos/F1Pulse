# Maps F1_GPU_BACKEND to the llama.cpp/ggml backend options.
# Must be included before add_subdirectory(third_party/llama.cpp).
# Variables are ordinary (non-cache) on purpose, see the root CMakeLists.txt.
# GGML_* names verified against third_party/llama.cpp/ggml/CMakeLists.txt.

set(_f1_valid_backends CPU VULKAN CUDA METAL)

if(NOT F1_GPU_BACKEND IN_LIST _f1_valid_backends)
    list(JOIN _f1_valid_backends ", " _f1_valid_list)
    message(FATAL_ERROR "F1_GPU_BACKEND='${F1_GPU_BACKEND}' is not valid. Valid values: ${_f1_valid_list}")
endif()

if(F1_GPU_BACKEND STREQUAL "CPU")
    # CPU backend is always built by ggml, nothing to enable.
elseif(F1_GPU_BACKEND STREQUAL "VULKAN")
    find_package(Vulkan QUIET)
    if(NOT Vulkan_FOUND OR NOT Vulkan_GLSLC_EXECUTABLE)
        message(FATAL_ERROR
            "F1_GPU_BACKEND=VULKAN requires the Vulkan SDK (headers, loader and the glslc shader compiler).\n"
            "  Vulkan_FOUND=${Vulkan_FOUND}, Vulkan_GLSLC_EXECUTABLE=${Vulkan_GLSLC_EXECUTABLE}\n"
            "  Windows: install the LunarG Vulkan SDK (https://vulkan.lunarg.com) and reopen the shell so VULKAN_SDK is set.\n"
            "  Ubuntu 24.04: sudo apt install libvulkan-dev glslc\n"
            "Or configure with a CPU preset / -DF1_GPU_BACKEND=CPU.")
    endif()
    set(GGML_VULKAN ON)
elseif(F1_GPU_BACKEND STREQUAL "CUDA")
    set(GGML_CUDA ON)
elseif(F1_GPU_BACKEND STREQUAL "METAL")
    if(NOT APPLE)
        message(FATAL_ERROR "F1_GPU_BACKEND=METAL is only available on Apple platforms")
    endif()
    set(GGML_METAL ON)
endif()

message(STATUS "F1Pulse: GPU backend = ${F1_GPU_BACKEND}")

unset(_f1_valid_backends)
unset(_f1_valid_list)
