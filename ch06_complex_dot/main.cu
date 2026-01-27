#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define N 3

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

// Check for cuBLAS errors
inline void checkcuBlasErr(cublasStatus_t err, const char* file, int line) {
    if (err != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS error in " << file << " at line " << line << ": "
            << cublasGetStatusName(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkcuBlasError(err) (checkcuBlasErr((err), __FILE__, __LINE__))

int main() {

    // Allocate space for vectors
    cuFloatComplex *xMem, *yMem;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&xMem),
        N * sizeof(cuFloatComplex)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&yMem),
        N * sizeof(cuFloatComplex)));

    // Transfer data to vectors
    float xData[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float yData[] = { 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
    checkcuBlasError(cublasSetVector(N, sizeof(cuFloatComplex),
        xData, 1, xMem, 1));
    checkcuBlasError(cublasSetVector(N, sizeof(cuFloatComplex),
        yData, 1, yMem, 1));

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkcuBlasError(cublasCreate(&handle));

    // Perform computations
    cuFloatComplex z1, z2;
    checkcuBlasError(cublasCdotc(handle, N, xMem, 1, yMem, 1, &z1));
    checkcuBlasError(cublasCdotu(handle, N, xMem, 1, yMem, 1, &z2));

    // Display results
    std::cout << "Conjugated dot product = " << cuCrealf(z1)
        << ", " << cuCimagf(z1) << "i" << std::endl;
    std::cout << "Unconjugated dot product = " << cuCrealf(z2)
        << ", " << cuCimagf(z2) << "i" << std::endl;

    // Free resources
    checkcuBlasError(cublasDestroy(handle));
    cudaFree(xMem);
    cudaFree(yMem);
    return 0;
}