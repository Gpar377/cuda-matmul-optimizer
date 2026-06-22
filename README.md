# CUDA Matrix Multiplication Optimizer

A high-performance C++ / CUDA library that implements and optimizes General Matrix Multiplication (GEMM) kernels from scratch, comparing throughput and latencies against CPython, NumPy, and PyTorch (cuBLAS). Exposes GPU routines natively to Python using PyBind11.

## 🚀 Optimization Roadmap

This repository progresses through the following optimization stages:
1. **Naive Implementation:** The baseline global memory access kernel where each thread calculates a single cell.
2. **Shared Memory Tiling:** Blocking matrices into static $16 \times 16$ Shared Memory tiles (`__shared__` arrays) to reuse loaded values and minimize global memory read loops.
3. **Thread Tiling (Register Caching):** Register-level accumulation where each thread computes a block of $8 \times 8$ elements, dramatically reducing shared memory load pressure.
4. **Vectorized Global Memory Reads (`float4`):** Memory bus utilization by loading elements using 128-bit wide vectorized instructions, optimizing access speeds.

---

## 🛠️ Build and Installation

Ensure you have the **CUDA Toolkit** (with `nvcc` compiler) and **CMake (>=3.18)** installed on your host system or WSL2.

### 1. Initialize Submodules
```bash
git submodule update --init --recursive
```

### 2. Set Up Virtual Environment (Python)
```bash
python -m venv .venv
# Activate virtual environment
source .venv/bin/activate  # Linux/WSL2
.venv\Scripts\activate     # Windows

# Install required profiling packages
pip install numpy torch matplotlib
```

### 3. Compile Modules
Using CMake to compile both the standalone C++ testing executable and the Python module:
```bash
cmake -B build
cmake --build build --config Release
```

---

## 🧪 Validation & Profiling

### Verify Accuracy
Runs comprehensive validation matching output values of each CUDA optimization kernel vs. NumPy CPU reference values:
```bash
python scripts/test_correctness.py
```

### Run Benchmarks
Calculates runtime latencies across matrix sizes (from $128 \times 128$ up to $4096 \times 4096$) for all custom kernels vs. PyTorch (which uses cuBLAS/CUTLASS). Automatically plots results to `benchmark_results.png`:
```bash
python scripts/benchmark.py
```

---

## 📈 Benchmarks Result Plot
*Upon running the benchmarks, the comparison plots will be generated here:*

![Benchmarks Comparison](benchmark_results.png)
