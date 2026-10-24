find_package(Threads REQUIRED)

add_executable(f1_pulse main.cpp)

target_compile_features(f1_pulse PRIVATE cxx_std_23)
target_link_libraries(f1_pulse PRIVATE f1::ai Threads::Threads f1_warnings)

if(MSVC)
    set_target_properties(f1_pulse PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
endif()
