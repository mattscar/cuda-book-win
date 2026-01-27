#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>

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
inline void checkcuSolverErr(cusolverStatus_t status, const char* file, int line) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "cuSOLVER error in " << file << " at line " << line << ": ";
        switch (status) {
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            std::cerr << "not initialized" << std::endl; break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            std::cerr << "allocation failed" << std::endl; break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            std::cerr << "invalid value" << std::endl; break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            std::cerr << "architecture mismatch" << std::endl; break;
        case CUSOLVER_STATUS_MAPPING_ERROR:
            std::cerr << "mapping error" << std::endl; break;
        case CUSOLVER_STATUS_EXECUTION_FAILED:
            std::cerr << "execution failed" << std::endl; break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            std::cerr << "internal error" << std::endl; break;
        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            std::cerr << "matrix type not supported" << std::endl; break;
        case CUSOLVER_STATUS_NOT_SUPPORTED:
            std::cerr << "not supported" << std::endl; break;
        case CUSOLVER_STATUS_ZERO_PIVOT:
            std::cerr << "zero pivot" << std::endl; break;
        case CUSOLVER_STATUS_INVALID_LICENSE:
            std::cerr << "invalid license" << std::endl; break;
        default:
            std::cerr << "unknown error" << std::endl; break;
        }
        std::exit(EXIT_FAILURE);
    }
}
#define checkcuSolverError(err) (checkcuSolverErr((err), __FILE__, __LINE__))

int main() {

    // Create cuSOLVER handle
    cusolverDnHandle_t handle;
    checkcuSolverError(cusolverDnCreate(&handle));

    // Allocate device memory
    const int dim = 3;
    double *aMem, *bMem;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&aMem), dim * dim * sizeof(double)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&bMem), dim * sizeof(double)));

    // Define the A matrix data
    double aData[dim * dim] = { 
        9.0, 2.0, 4.0, // Column 0
        4.0, 6.0, 1.0, // Column 1
        2.0, 5.0, 7.0  // Column 2
    };

    // Set the b vector data
    double bData[dim] = { 3.0, 5.0, 7.0 };

    // Transfer data to device
    checkCudaErrors(cudaMemcpy(aMem, aData, dim * dim * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(bMem, bData, dim * sizeof(double), cudaMemcpyHostToDevice));

    // Determine workspace size
    int workspace_size = 0;
    checkcuSolverError(cusolverDnDgetrf_bufferSize(handle, dim, dim, aMem, dim, &workspace_size));

    // Allocate memory for workspace
    double *workspaceMem;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&workspaceMem), workspace_size * sizeof(double)));

    // Allocate memory for pivot info and error info
    int *pivotMem, *errMem;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&pivotMem), dim * sizeof(int)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&errMem), sizeof(int)));

    // Perform LU decomposition
    checkcuSolverError(cusolverDnDgetrf(handle, dim, dim, aMem, dim, workspaceMem, pivotMem, errMem));

    // Solve matrix system
    checkcuSolverError(cusolverDnDgetrs(handle, CUBLAS_OP_N, dim, 1, aMem, dim, pivotMem, bMem, dim, errMem));

    // Copy solution to host
    double xData[dim];
    checkCudaErrors(cudaMemcpy(xData, bMem, dim * sizeof(double), cudaMemcpyDeviceToHost));

    // Display solution
    std::cout << "Solution: x = " << xData[0] << ", " << xData[1] << ", " << xData[2] << std::endl;

    // Free resources
    cudaFree(aMem);
    cudaFree(bMem);
    cudaFree(workspaceMem);
    cudaFree(pivotMem);
    cudaFree(errMem);
    cusolverDnDestroy(handle);
    return 0;
}