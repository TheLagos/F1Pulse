# INTERFACE target carrying the warning flags for F1Pulse code.
# Link it PRIVATE to project targets only, never to llama.cpp/ggml.

add_library(f1_warnings INTERFACE)

if(MSVC)
    target_compile_options(f1_warnings INTERFACE
        /W4
        /permissive-
        $<$<BOOL:${F1_WARNINGS_AS_ERRORS}>:/WX>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(f1_warnings INTERFACE
        -Wall
        -Wextra
        -Wpedantic
        $<$<BOOL:${F1_WARNINGS_AS_ERRORS}>:-Werror>)
else()
    message(WARNING "F1Pulse: unsupported compiler '${CMAKE_CXX_COMPILER_ID}', no warning flags set")
endif()
