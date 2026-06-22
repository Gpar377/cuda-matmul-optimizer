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

// CUDA Thread-Tiled Matrix Multiplication Kernel (Coarse Tiling)
// Each block computes a TILE_M x TILE_N output region.
// Each thread computes a sub-region using register variables to cache outputs.
#define TILE_M 64
#define TILE_N 64
#define TILE_K 8
#define THREAD_M 8
#define THREAD_N 8

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Shared memory allocations for tiling A and B
    __shared__ float s_A[TILE_M][TILE_K];
    __shared__ float s_B[TILE_K][TILE_N];

    // Thread index identifiers
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Output tile indices matching output block matrix coordinates
    int blockRow = blockIdx.y * TILE_M;
    int blockCol = blockIdx.x * TILE_N;

    // Registers cache to accumulate results of size THREAD_M x THREAD_N
    float accum[THREAD_M][THREAD_N];
    for (int i = 0; i < THREAD_M; ++i) {
        for (int j = 0; j < THREAD_N; ++j) {
            accum[i][j] = 0.0f;
        }
    }

    // Number of threads along dimensions
    int threads_x = blockDim.x; // TILE_N / THREAD_N
    int threads_y = blockDim.y; // TILE_M / THREAD_M

    // Unique thread ID mapped to load coordinates in shared memory
    int tid = ty * threads_x + tx;

    // Loops over tiles of matrix dimension K
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Coalesced loads from Global Memory to Shared Memory for current tile
        // Each thread loads multiple elements to fill s_A and s_B
        
        // Load s_A: size is TILE_M x TILE_K. Total values: TILE_M * TILE_K = 512.
        // Total threads in block: (64/8)*(64/8) = 8*8 = 64 threads.
        // Each thread needs to load 512 / 64 = 8 elements.
        for (int loadOffset = 0; loadOffset < TILE_M * TILE_K; loadOffset += threads_x * threads_y) {
            int currId = tid + loadOffset;
            int a_row = currId / TILE_K;
            int a_col = currId % TILE_K;
            int global_a_row = blockRow + a_row;
            int global_a_col = t * TILE_K + a_col;

            if (global_a_row < M && global_a_col < K) {
                s_A[a_row][a_col] = A[global_a_row * K + global_a_col];
            } else {
                s_A[a_row][a_col] = 0.0f;
            }
        }

        // Load s_B: size is TILE_K x TILE_N. Total values: TILE_K * TILE_N = 512.
        // Each thread needs to load 512 / 64 = 8 elements.
        for (int loadOffset = 0; loadOffset < TILE_K * TILE_N; loadOffset += threads_x * threads_y) {
            int currId = tid + loadOffset;
            int b_row = currId / TILE_N;
            int b_col = currId % TILE_N;
            int global_b_row = t * TILE_K + b_row;
            int global_b_col = blockCol + b_col;

            if (global_b_row < K && global_b_col < N) {
                s_B[b_row][b_col] = B[global_b_row * N + global_b_col];
            } else {
                s_B[b_row][b_col] = 0.0f;
            }
        }

        // Synchronize threads to guarantee shared memory load phase completes
        __syncthreads();

        // Perform inner multiplications inside current tile and accumulate
        for (int k = 0; k < TILE_K; ++k) {
            // Read values for registers
            float reg_B[THREAD_N];
            for (int j = 0; j < THREAD_N; ++j) {
                reg_B[j] = s_B[k][tx * THREAD_N + j];
            }

            for (int i = 0; i < THREAD_M; ++i) {
                float val_A = s_A[ty * THREAD_M + i][k];
                for (int j = 0; j < THREAD_N; ++j) {
                    accum[i][j] += val_A * reg_B[j];
                }
            }
        }

        // Synchronize before moving to next global memory tile load
        __syncthreads();
    }

    // Write thread values from registers back to global memory (C)
    for (int i = 0; i < THREAD_M; ++i) {
        int global_row = blockRow + ty * THREAD_M + i;
        for (int j = 0; j < THREAD_N; ++j) {
            int global_col = blockCol + tx * THREAD_N + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = accum[i][j];
            }
        }
    }
}

// Host launcher wrapper for thread tiled kernel
void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
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

    // Block dimensions are defined by threads count mapping
    // Thread block computes TILE_M x TILE_N outputs.
    // Each thread computes THREAD_M x THREAD_N outputs.
    // Block size along X = TILE_N / THREAD_N = 64 / 8 = 8 threads.
    // Block size along Y = TILE_M / THREAD_M = 64 / 8 = 8 threads.
    // Total threads per block: 8 * 8 = 64 threads.
    dim3 threadsPerBlock(TILE_N / THREAD_N, TILE_M / THREAD_M);
    dim3 numBlocks((N + TILE_N - 1) / TILE_N, 
                   (M + TILE_M - 1) / TILE_M);

    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();
    CUDA_CHECK();

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    CUDA_CHECK();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    CUDA_CHECK();
}


