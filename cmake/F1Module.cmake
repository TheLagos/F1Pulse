# f1_add_module(<name>
#     [SOURCES <files>...]
#     [PUBLIC_DEPS <targets>...]
#     [PRIVATE_DEPS <targets>...])
#
# Creates the STATIC library f1_<name> with alias f1::<name>.
# Call it from the module's own CMakeLists.txt; SOURCES are relative to that
# directory. Headers are included as "<module>/<Header>.hpp" via
# ${PROJECT_SOURCE_DIR}/src. If F1_BUILD_TESTS is ON and the module has a
# tests/CMakeLists.txt, that directory is added too.

function(f1_add_module name)
    cmake_parse_arguments(PARSE_ARGV 1 ARG "" "" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS")

    if(ARG_UNPARSED_ARGUMENTS)
        message(FATAL_ERROR "f1_add_module(${name}): unknown arguments: ${ARG_UNPARSED_ARGUMENTS}")
    endif()

    if(NOT ARG_SOURCES)
        message(FATAL_ERROR "f1_add_module(${name}): SOURCES is required for a STATIC library")
    endif()

    set(target f1_${name})

    list(TRANSFORM ARG_SOURCES PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/")

    add_library(${target} STATIC ${ARG_SOURCES})
    add_library(f1::${name} ALIAS ${target})

    target_include_directories(${target} PUBLIC "${PROJECT_SOURCE_DIR}/src")
    target_compile_features(${target} PUBLIC cxx_std_23)
    target_link_libraries(${target} PRIVATE f1_warnings)

    if(ARG_PUBLIC_DEPS)
        target_link_libraries(${target} PUBLIC ${ARG_PUBLIC_DEPS})
    endif()

    if(ARG_PRIVATE_DEPS)
        target_link_libraries(${target} PRIVATE ${ARG_PRIVATE_DEPS})
    endif()

    if(F1_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests")
    endif()
endfunction()
