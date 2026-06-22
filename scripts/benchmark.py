import numpy as np
import torch
import time
import sys
import os
import matplotlib.pyplot as plt

# Include build directory to load custom pybind11 module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

try:
    import cuda_matmul_backend as backend
    print("SUCCESS: Loaded cuda_matmul_backend module successfully for benchmarking!")
except ImportError as e:
    print(f"ERROR: Could not load custom CUDA backend module: {e}")
    sys.exit(1)

# List of matrix dimensions to benchmark (M = N = K)
SIZES = [128, 256, 512, 1024, 2048, 4096]
WARMUP_RUNS = 3
BENCHMARK_RUNS = 10

def benchmark_all():
    print(f"Starting performance benchmarks (Warmups: {WARMUP_RUNS}, Runs: {BENCHMARK_RUNS})...")
    
    # Store timing results
    times_naive = []
    times_shared = []
    times_tiled = []
    times_vec = []
    times_pytorch = []
    
    for size in SIZES:
        M, N, K = size, size, size
        print(f"\nBenchmarking Matrix Size: {size}x{size}...")
        
        # Prepare inputs
        A_np = np.random.rand(M, K).astype(np.float32)
        B_np = np.random.rand(K, N).astype(np.float32)
        
        # PyTorch Tensors on GPU
        if torch.cuda.is_available():
            A_pt = torch.from_numpy(A_np).cuda()
            B_pt = torch.from_numpy(B_np).cuda()
        else:
            A_pt = torch.from_numpy(A_np)
            B_pt = torch.from_numpy(B_np)

        # ----------------- Benchmark PyTorch (cuBLAS) -----------------
        for _ in range(WARMUP_RUNS):
            _ = torch.matmul(A_pt, B_pt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = torch.matmul(A_pt, B_pt)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        t_pt = ((end - start) / BENCHMARK_RUNS) * 1000
        times_pytorch.append(t_pt)
        print(f"  PyTorch / cuBLAS: {t_pt:.3f} ms")

        # ----------------- Benchmark Naive Kernel -----------------
        for _ in range(WARMUP_RUNS):
            _ = backend.matmul_naive(A_np, B_np)
        
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = backend.matmul_naive(A_np, B_np)
        end = time.perf_counter()
        t_naive = ((end - start) / BENCHMARK_RUNS) * 1000
        times_naive.append(t_naive)
        print(f"  Custom Naive:     {t_naive:.3f} ms")

        # ----------------- Benchmark Shared Memory Kernel -----------------
        for _ in range(WARMUP_RUNS):
            _ = backend.matmul_shared(A_np, B_np)
        
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = backend.matmul_shared(A_np, B_np)
        end = time.perf_counter()
        t_shared = ((end - start) / BENCHMARK_RUNS) * 1000
        times_shared.append(t_shared)
        print(f"  Shared Tiled:     {t_shared:.3f} ms")

        # ----------------- Benchmark Thread-Tiled Kernel -----------------
        for _ in range(WARMUP_RUNS):
            _ = backend.matmul_tiled(A_np, B_np)
        
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = backend.matmul_tiled(A_np, B_np)
        end = time.perf_counter()
        t_tiled = ((end - start) / BENCHMARK_RUNS) * 1000
        times_tiled.append(t_tiled)
        print(f"  Thread Tiled:     {t_tiled:.3f} ms")

        # ----------------- Benchmark Vectorized Kernel -----------------
        # Ensure sizes are multiples of 8 for vectorized float4 operations
        for _ in range(WARMUP_RUNS):
            _ = backend.matmul_vectorized(A_np, B_np)
        
        start = time.perf_counter()
        for _ in range(BENCHMARK_RUNS):
            _ = backend.matmul_vectorized(A_np, B_np)
        end = time.perf_counter()
        t_vec = ((end - start) / BENCHMARK_RUNS) * 1000
        times_vec.append(t_vec)
        print(f"  Vectorized Float4:{t_vec:.3f} ms")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(SIZES, times_naive, marker='o', label='Custom Naive')
    plt.plot(SIZES, times_shared, marker='s', label='Shared Tiled (16x16)')
    plt.plot(SIZES, times_tiled, marker='^', label='Thread Tiled (8x8)')
    plt.plot(SIZES, times_vec, marker='x', label='Vectorized Float4')
    plt.plot(SIZES, times_pytorch, marker='*', linestyle='--', label='PyTorch (cuBLAS)')
    
    plt.xlabel('Matrix Dimension (N)')
    plt.ylabel('Execution Time (ms)')
    plt.title('CUDA Matrix Multiplication Kernel Benchmarks')
    plt.yscale('log') # Logarithmic scale since naive times are huge for large size matrices
    plt.grid(True, which="both", ls="--")
    plt.legend()
    
    output_img = os.path.abspath(os.path.join(os.path.dirname(__file__), '../benchmark_results.png'))
    plt.savefig(output_img, dpi=300)
    print(f"\nBenchmarking complete! Results plot saved to: {output_img}")

if __name__ == "__main__":
    benchmark_all()
