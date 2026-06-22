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

// CUDA Shared Memory Tiled Matrix Multiplication Kernel
// Block size: 16x16 (TILE_SIZE = 16)
#define TILE_SIZE 16

__global__ void matmul_shared_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Allocate shared memory for tiles
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    // Loop over all shared memory tiles required to compute the output element
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tile from A into shared memory (with bounds checking)
        if (row < M && (t * TILE_SIZE + tx) < K) {
            s_A[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory (with bounds checking)
        if (col < N && (t * TILE_SIZE + ty) < K) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }

        // Synchronize threads in block to ensure tiles are fully loaded
        __syncthreads();

        // Multiply elements of current tiles and accumulate partial sums
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += s_A[ty][k] * s_B[k][tx];
        }

        // Synchronize again before loading next tile
        __syncthreads();
    }

    // Write computed element back to global memory (bounds checking)
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host launcher wrapper for shared memory kernel
void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K) {
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    CUDA_CHECK();

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    CUDA_CHECK();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_shared_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    CUDA_CHECK();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    CUDA_CHECK();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    CUDA_CHECK();
}

