# Claude Code Guidelines - CUDA Matmul Optimizer

## Project Overview
This repository contains high-performance CUDA Matrix Multiplication kernels with pybind11 Python bindings.

## Technology Stack
*   **CUDA C++** (NVCC compiler, CUDA Toolkit)
*   **C++17** (Host side code)
*   **Python 3.10+** (pybind11 bindings, NumPy, PyTorch for validation, Matplotlib for benchmarking)
*   **Build Tools:** CMake / Makefile

## Coding Standards & Conventions
*   Annotate CUDA kernels clearly with `__global__` and helper functions with `__device__` / `__host__`.
*   Ensure grid and block dimensions are computed dynamically based on input matrix dimensions to prevent out-of-bounds memory accesses.
*   Explicitly document shared memory allocation sizing and warp/thread layout assumptions.
*   Check all CUDA system calls using a standard check macro (e.g., `gpuErrchk`).
*   Ensure proper synchronization using `__syncthreads()` within tiled kernels.

## Workflow Rules & Commands
*   **Virtual Environment Setup:** `python -m venv .venv && source .venv/bin/activate` (Linux/WSL) or `.venv\Scripts\activate` (Windows)
*   **Install Dependencies:** `pip install numpy torch matplotlib pybind11`
*   **Build C++ Extensions:** `cmake -B build && cmake --build build` or `make`
*   **Verify Accuracy:** Run the validation script `python scripts/test_correctness.py`
*   **Run Benchmarks:** `python scripts/benchmark.py`
