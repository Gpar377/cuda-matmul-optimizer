#ifndef MATMUL_H
#define MATMUL_H

// Matrix dimensions structure
struct MatrixDim {
    int rows;
    int cols;
};

// Error checking wrapper for CUDA calls
void cudaCheck(const char* file, int line);
#define CUDA_CHECK() cudaCheck(__FILE__, __LINE__)

// Host-side entry points to CUDA implementations
void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K);
void matmul_shared(const float* A, const float* B, float* C, int M, int N, int K);

#endif // MATMUL_H
