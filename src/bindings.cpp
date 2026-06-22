#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "matmul.h"

namespace py = pybind11;

// PyBind11 wrapper wrapping host launchers to accept and return NumPy arrays
py::array_t<float> py_matmul_naive(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    if (buf_A.ndim != 2 || buf_B.ndim != 2) {
        throw std::runtime_error("Input matrices must be 2D");
    }

    int M = buf_A.shape[0];
    int K = buf_A.shape[1];
    int N = buf_B.shape[1];

    if (buf_B.shape[0] != K) {
        throw std::runtime_error("Matrix inner dimensions must match for multiplication");
    }

    // Allocate result array
    auto result = py::array_t<float>({M, N});
    py::buffer_info buf_C = result.request();

    // Call raw CUDA launch wrapper
    matmul_naive(
        static_cast<const float*>(buf_A.ptr),
        static_cast<const float*>(buf_B.ptr),
        static_cast<float*>(buf_C.ptr),
        M, N, K
    );

    return result;
}

py::array_t<float> py_matmul_shared(py::array_t<float> A, py::array_t<float> B) {
    py::buffer_info buf_A = A.request();
    py::buffer_info buf_B = B.request();

    if (buf_A.ndim != 2 || buf_B.ndim != 2) {
        throw std::runtime_error("Input matrices must be 2D");
    }

    int M = buf_A.shape[0];
    int K = buf_A.shape[1];
    int N = buf_B.shape[1];

    if (buf_B.shape[0] != K) {
        throw std::runtime_error("Matrix inner dimensions must match for multiplication");
    }

    auto result = py::array_t<float>({M, N});
    py::buffer_info buf_C = result.request();

    matmul_shared(
        static_cast<const float*>(buf_A.ptr),
        static_cast<const float*>(buf_B.ptr),
        static_cast<float*>(buf_C.ptr),
        M, N, K
    );

    return result;
}

PYBIND11_MODULE(cuda_matmul_backend, m) {
    m.doc() = "CUDA Matrix Multiplication High-Performance Kernels";
    m.def("matmul_naive", &py_matmul_naive, "Naive CUDA Matrix Multiplication");
    m.def("matmul_shared", &py_matmul_shared, "Shared Memory Tiled CUDA Matrix Multiplication");
}

