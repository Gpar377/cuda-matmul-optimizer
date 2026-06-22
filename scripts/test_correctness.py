import numpy as np
import torch
import time
import sys
import os

# Include build directory to load custom pybind11 module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../build')))

try:
    import cuda_matmul_backend as backend
    print("SUCCESS: Loaded cuda_matmul_backend module successfully!")
except ImportError as e:
    print(f"ERROR: Could not load custom CUDA backend module: {e}")
    print("Please verify you compiled it using: cmake -B build && cmake --build build")
    sys.exit(1)

def verify_correctness():
    M, N, K = 512, 512, 512
    print(f"Initializing random validation matrices ({M}x{K} * {K}x{N})...")
    
    # Generate random floats
    A = np.random.rand(M, K).astype(np.float32)
    B = np.random.rand(K, N).astype(np.float32)
    
    # Run CPU NumPy comparison
    print("Running CPU NumPy matmul reference...")
    start_cpu = time.perf_counter()
    C_cpu = np.matmul(A, B)
    end_cpu = time.perf_counter()
    print(f"NumPy CPU execution time: {(end_cpu - start_cpu)*1000:.2f} ms")
    
    # Run custom GPU naive kernel
    print("Running Custom GPU naive matmul...")
    try:
        start_gpu = time.perf_counter()
        C_gpu = backend.matmul_naive(A, B)
        end_gpu = time.perf_counter()
        print(f"Custom GPU naive execution time: {(end_gpu - start_gpu)*1000:.2f} ms")
    except Exception as e:
        print(f"GPU Execution error: {e}")
        sys.exit(1)
        
    # Check difference within tolerance
    diff = np.abs(C_cpu - C_gpu)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Naive GPU metrics - Max discrepancy: {max_diff:.6f} | Mean discrepancy: {mean_diff:.6f}")
    if max_diff >= 1e-4:
        print("FAILED: Naive GPU output discrepancy exceeds precision tolerances!")
        sys.exit(1)
    print("PASSED: Custom CUDA naive results match CPU reference!")

    # Run custom GPU shared memory kernel
    print("Running Custom GPU shared memory matmul...")
    try:
        start_gpu_shared = time.perf_counter()
        C_gpu_shared = backend.matmul_shared(A, B)
        end_gpu_shared = time.perf_counter()
        print(f"Custom GPU shared memory execution time: {(end_gpu_shared - start_gpu_shared)*1000:.2f} ms")
    except Exception as e:
        print(f"GPU Shared Execution error: {e}")
        sys.exit(1)

    # Check difference within tolerance for shared memory kernel
    diff_shared = np.abs(C_cpu - C_gpu_shared)
    max_diff_shared = np.max(diff_shared)
    mean_diff_shared = np.mean(diff_shared)

    print(f"Shared GPU metrics - Max discrepancy: {max_diff_shared:.6f} | Mean discrepancy: {mean_diff_shared:.6f}")
    if max_diff_shared < 1e-4:
        print("PASSED: Custom CUDA shared memory results match CPU reference output within tolerance!")
    else:
        print("FAILED: Shared GPU output discrepancy exceeds precision tolerances!")
        sys.exit(1)

    # Run custom GPU thread tiled kernel
    print("Running Custom GPU thread tiled matmul...")
    try:
        start_gpu_tiled = time.perf_counter()
        C_gpu_tiled = backend.matmul_tiled(A, B)
        end_gpu_tiled = time.perf_counter()
        print(f"Custom GPU thread tiled execution time: {(end_gpu_tiled - start_gpu_tiled)*1000:.2f} ms")
    except Exception as e:
        print(f"GPU Tiled Execution error: {e}")
        sys.exit(1)

    # Check difference within tolerance for thread tiled kernel
    diff_tiled = np.abs(C_cpu - C_gpu_tiled)
    max_diff_tiled = np.max(diff_tiled)
    mean_diff_tiled = np.mean(diff_tiled)

    print(f"Tiled GPU metrics - Max discrepancy: {max_diff_tiled:.6f} | Mean discrepancy: {mean_diff_tiled:.6f}")
    if max_diff_tiled < 1e-4:
        print("PASSED: Custom CUDA thread tiled results match CPU reference output within tolerance!")
    else:
        print("FAILED: Tiled GPU output discrepancy exceeds precision tolerances!")
        sys.exit(1)

if __name__ == "__main__":
    verify_correctness()


