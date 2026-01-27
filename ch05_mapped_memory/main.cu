#include <iostream>

#include "cuda_runtime.h"

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

// Scales elements by N
__global__ void scaleArray(float* data, float N) {
    data[threadIdx.x] *= N;
}

int main() {
    const int numVals = 16;
    const size_t size = numVals * sizeof(int);

    // Enable mapped memory
    cudaSetDeviceFlags(cudaDeviceMapHost);

    // Allocate mappable pinned memory array on host
    float *hostMem;
    checkCudaErrors(cudaHostAlloc(reinterpret_cast<void**>(&hostMem), 
        size, cudaHostAllocMapped));

    // Initialize host memory
    for (int i=0; i<numVals; i++) {
        hostMem[i] = 1.0f * i;
    }

    // Obtain pointer to device memory mapped to host memory
    float* deviceMem;
    checkCudaErrors(cudaHostGetDevicePointer(
        reinterpret_cast<void**>(&deviceMem), 
        reinterpret_cast<void*>(hostMem), 0));

    // Launch kernel
    const float scalingFactor = 7.0f;
    scaleArray <<<1, numVals >>>(deviceMem, scalingFactor);

    // Synchronize to make sure kernel finished
    checkCudaErrors(cudaDeviceSynchronize());

    // Display processed data
    std::cout << "Results:" << std::endl;
    for (int i=0; i<numVals; i++) {
        std::cout << hostMem[i] << " ";
    }
    std::cout << std::endl;

    // Free host memory
    checkCudaErrors(cudaFreeHost(hostMem));
    return 0;
}
