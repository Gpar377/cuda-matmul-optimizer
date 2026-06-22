# CUDA Matrix Multiply Kernel
High-performance CUDA GEMM (General Matrix Multiply) implementations, progressing from naive global memory kernels to highly optimized shared memory tiling, thread tiling, vectorization, and register caching. wrapped with Python bindings via PyBind11 to run performance benchmarks against cuBLAS and PyTorch.

## Proposed Git Repo Name
`cuda-matmul-optimizer`

## Architecture & Scope
*   **Naive Matmul:** Baseline kernel using simple global memory access.
*   **Shared Memory Tiling:** Blocking matrices into Shared Memory tiles ($16 \times 16$ or $32 \times 32$) to reduce global memory load bandwidth by a factor of the tile size. Eliminates memory coalescing issues.
*   **Thread Tiling / Register Accumulation:** Each thread calculates a sub-tile ($4 \times 4$ or $8 \times 8$) of the output matrix to maximize register reuse.
*   **Vectorization (Float4):** Using `float4` to load/store vectors of 4 floats at once to maximize memory bus utilization.
*   **Warp-Level Optimizations:** Restructuring access patterns to avoid shared memory bank conflicts (using padding/striding).
*   **PyBind11 Interface:** C++ wrapper compiling the host-side code, invoking NVCC-compiled CUDA kernels, and returning NumPy arrays.
*   **Benchmark Suite:** Python-based comparison script running benchmarks against PyTorch (which uses cuBLAS/CUTLASS underneath) across matrix sizes from $128 \times 128$ to $8192 \times 8192$. Plots throughput (TFLOPS) and latency.

## Target Milestones
1. Naive kernel running and validated via Python.
2. Shared memory tiled kernel implemented, avoiding bank conflicts.
3. Thread-tiling/register caching implementation with profiling via Nsight Compute.
4. Float4 vectorization and memory access optimization.
5. Benchmark plots and final report/blog post.
