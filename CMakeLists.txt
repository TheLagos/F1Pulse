cmake_minimum_required(VERSION 3.22)
project(F1Pulse CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDART_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(f1_pulse
    src/main.cpp
    src/math/VectorMath.cpp
    src/storage/Database.cpp
)

target_include_directories(f1_pulse PRIVATE src)