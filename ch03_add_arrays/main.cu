
#include <iostream>

#include "cuda_runtime.h"

// The kernel function to be executed on the GPU
__global__ void addArrays(const float* a, const float* b, float* c) {
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main() {
    const int N = 6;
    float* arrayA, * arrayB, * arrayC;
    cudaError_t cudaStatus;

    // Allocate buffer for the first input array
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&arrayA), N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to allocate first input array" << std::endl;
    }

    // Allocate buffer for the second input array
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&arrayB), N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to allocate second input array" << std::endl;
    }

    // Allocate buffer for the output array
    cudaStatus = cudaMalloc(reinterpret_cast<void**>(&arrayC), N * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to allocate output array" << std::endl;
    }

    // Set content of first input array
    float aData[N] = { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f };
    cudaStatus = cudaMemcpy(arrayA, aData, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to transfer data to first input array" << std::endl;
    }

    // Set content of second input array
    float bData[N] = { 3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f };
    cudaStatus = cudaMemcpy(arrayB, bData, N * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to transfer data to second input array" << std::endl;
    }

    // Execute kernel
    addArrays << <1, N >> > (arrayA, arrayB, arrayC);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to execute kernel" << std::endl;
        return 1;
    }

    // Access processed output
    float cData[N];
    cudaStatus = cudaMemcpy(cData, arrayC, N * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Failed to transfer output array to host" << std::endl;
    }
    cudaFree(arrayA);
    cudaFree(arrayB);
    cudaFree(arrayC);

    // Display result
    for (int i = 0; i < N; i++) {
        std::cout << aData[i] << " plus " << bData[i] << " equals " << cData[i] << std::endl;
    }
    return 0;
}
