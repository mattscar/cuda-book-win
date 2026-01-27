#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>

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
    const int m = 4;  // Number of rows
    const int n = 4;  // Number of columns

    // Allocate device memory
    float *aMem, *xMem, *yMem;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&aMem), m * n * sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&xMem), n * sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&yMem), m * sizeof(float)));

    // Define the A matrix data
    float aData[m * n] = {
        1.0f, 2.0f, 4.0f, 6.0f,     // Col 0
        3.0f, 6.0f, 9.0f, 12.0f,    // Col 1
        5.0f, 10.0f, 15.0f, 20.0f,  // Col 2
        7.0f, 14.0f, 21.0f, 28.0f,  // Col 3
    };

    // Set the x vector data
    float xData[n] = { 3.0f, 7.0f, 11.0f, 17.0f };

    // Transfer data to device
    checkCudaErrors(cudaMemcpy(aMem, aData, m*n*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(xMem, xData, n*sizeof(float), cudaMemcpyHostToDevice));

    // Create cuBLAS handle
    cublasHandle_t handle;
    checkcuBlasError(cublasCreate(&handle));

    // Perform y = alpha * A * x + beta * y
    // A is m x n, so lda = m
    float alpha = 1.0f, beta = 1.0f;
    checkcuBlasError(cublasSgemv(
        handle,
        CUBLAS_OP_N,   // No operation
        m,             // Number of rows
        n,             // Number of columns
        &alpha,        // Scaling factor for A
        aMem,          // A matrix
        m,             // Leading dimension of A
        xMem,          // x vector
        1,             // Stride for x
        &beta,         // Scaling factor for x
        yMem,          // y vector
        1              // Stride for y
    ));

    // Copy y vector to host
    float yData[m];
    checkCudaErrors(cudaMemcpy(yData, yMem, m * sizeof(float), cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Result y = [ ";
    for (int i = 0; i < m; ++i) {
        std::cout << yData[i] << " ";
    }
    std::cout << "]" << std::endl;

    // Destroy handle
    checkcuBlasError(cublasDestroy(handle));

    // Free resources
    cudaFree(aMem);
    cudaFree(xMem);
    cudaFree(yMem);
    return 0;
}
