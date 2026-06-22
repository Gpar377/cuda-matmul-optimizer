#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>
#include "../include/matmul.h"

// Initialize matrix elements with random floats
void initMatrix(std::vector<float>& mat, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; ++i) {
        mat[i] = dis(gen);
    }
}

// CPU reference matrix multiplication to verify correctness
void matmulCPU(const std::vector<float>& A, const std::vector<float>& B, std::vector<float>& C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Verify correctness between CPU and GPU
bool verifyCorrectness(const std::vector<float>& cpu_res, const std::vector<float>& gpu_res, float tolerance = 1e-4f) {
    if (cpu_res.size() != gpu_res.size()) return false;
    for (size_t i = 0; i < cpu_res.size(); ++i) {
        if (std::abs(cpu_res[i] - gpu_res[i]) > tolerance) {
            std::cerr << "Verification failed at index " << i 
                      << " | CPU: " << cpu_res[i] << " | GPU: " << gpu_res[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    // Keep sizes small for initial correctness check on host CPU verification
    const int M = 256;
    const int N = 256;
    const int K = 256;

    std::cout << "Initializing matrices: A(" << M << "x" << K << "), B(" << K << "x" << N << ")..." << std::endl;
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_gpu(M * N, 0.0f);
    std::vector<float> h_C_cpu(M * N, 0.0f);

    initMatrix(h_A, M, K);
    initMatrix(h_B, K, N);

    std::cout << "Running CPU reference multiplication..." << std::endl;
    matmulCPU(h_A, h_B, h_C_cpu, M, N, K);

    std::cout << "Running GPU naive multiplication..." << std::endl;
    matmul_naive(h_A.data(), h_B.data(), h_C_gpu.data(), M, N, K);

    std::cout << "Checking naive results..." << std::endl;
    if (verifyCorrectness(h_C_cpu, h_C_gpu)) {
        std::cout << "SUCCESS: GPU naive and CPU results match." << std::endl;
    } else {
        std::cout << "FAIL: GPU naive and CPU results mismatch." << std::endl;
        return -1;
    }

    // Reset output buffer and verify shared memory kernel
    std::fill(h_C_gpu.begin(), h_C_gpu.end(), 0.0f);
    std::cout << "Running GPU shared memory multiplication..." << std::endl;
    matmul_shared(h_A.data(), h_B.data(), h_C_gpu.data(), M, N, K);

    std::cout << "Checking shared memory results..." << std::endl;
    if (verifyCorrectness(h_C_cpu, h_C_gpu)) {
        std::cout << "SUCCESS: GPU shared memory and CPU results match." << std::endl;
    } else {
        std::cout << "FAIL: GPU shared memory and CPU results mismatch." << std::endl;
        return -1;
    }

    return 0;
}

