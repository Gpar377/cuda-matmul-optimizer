#include <cuda_runtime.h>
#include <iostream>
#include "../include/matmul.h"

// CUDA error-checking helper implementation
void cudaCheck(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error at " << file << ":" << line << " - " 
                  << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// CUDA Naive Matrix Multiplication Kernel
// Computes C = A * B
// A is M x K, B is K x N, C is M x N
__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host launcher wrapper for naive kernel
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    CUDA_CHECK();

    // Copy host memory to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    CUDA_CHECK();

    // Configure thread block and grid dimensions
    // Block size of 16x16 (256 threads per block)
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch CUDA Kernel
    matmul_naive_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    CUDA_CHECK();

    // Copy results back to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    CUDA_CHECK();

    // Free device allocations
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    CUDA_CHECK();
}
