cmake_minimum_required(VERSION 3.18)
project(CudaMatmulOptimizer LANGUAGES CXX CUDA)

# Enforce C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Configure CUDA compilation options
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Auto-detect target GPU architectures
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86) # Common Nvidia architectures (Turing, Ampere)
endif()

# Set search directory headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Define executable build target
add_executable(matmul_test 
    src/main.cpp 
    src/matmul.cu
)

# Optional flags for performance tuning
target_compile_options(matmul_test PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math -O3>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# Add pybind11 subdirectory and compile Python module
add_subdirectory(third_party/pybind11)
pybind11_add_module(cuda_matmul_backend src/bindings.cpp src/matmul.cu)
target_include_directories(cuda_matmul_backend PRIVATE ${PROJECT_SOURCE_DIR}/include)

