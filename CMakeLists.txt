cmake_minimum_required(VERSION 3.14)
project(tensor_optimizer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ---- Dependencies ----
include(FetchContent)

# Google Benchmark
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.9.4
)

# Eigen
FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG 3.4.0
        GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(googlebenchmark eigen)



# ---- Tensor Operations Library ----
add_library(tensor_ops
        tensor-ops/NaiveOperations.cpp
        tensor-ops/EigenOperations.cpp
        tensor-ops/NativeOperations.cpp
        tensor-ops/ExperimentalOperations.cpp
        native-tensor-library/NativeTensorLibrary.cpp
        matrix-library/MatrixLibrary.cpp
        experimental-tensor-library/ETensor.cpp
        custom-allocator/AlignedCustomAllocator.cpp
        custom-allocator/AlignedCustomAllocator.h


)

option(TENSOR_OPT_DISABLE_EIGEN_VECTORIZE "Disable Eigen vectorization" OFF)

if (TENSOR_OPT_DISABLE_EIGEN_VECTORIZE)
    target_compile_definitions(tensor_ops PRIVATE EIGEN_DONT_VECTORIZE)
endif()

target_include_directories(tensor_ops PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(tensor_ops PUBLIC
        Eigen3::Eigen
)

# ---- Main Executable ----
add_executable(tensor_optimizer main.cpp)

target_link_libraries(tensor_optimizer
        PRIVATE
        tensor_ops
        benchmark::benchmark
)