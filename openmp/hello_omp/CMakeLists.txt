cmake_minimum_required(VERSION 3.6)
project(hello_omp)

set(CMAKE_CXX_STANDARD 14)

set(SOURCE_FILES main.cpp)
add_executable(hello_omp ${SOURCE_FILES})

# for openmp
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
