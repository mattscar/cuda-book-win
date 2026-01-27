
#include <iostream>

#include "cuda_runtime.h"

inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

#define NUM_VALS 7

// The kernel function to be executed on the GPU
__global__ void printBuiltin(int* output) {
    if ((threadIdx.x == 0) && (threadIdx.y == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
        output[0] = blockDim.x;
        output[1] = blockDim.y;
        output[2] = blockDim.z;
        output[3] = gridDim.x;
        output[4] = gridDim.y;
        output[5] = gridDim.z;
        output[6] = warpSize;
    }
}

int main() {

    // Allocate buffer for the output array
    int* output;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&output), NUM_VALS * sizeof(int)));

    // Set the dimensions of the grid and thread blocks
    dim3 grid_dims(2, 5);
    dim3 block_dims(3, 4);

    // Invoke the kernel
    printBuiltin<<<grid_dims, block_dims>>>(output);

    // Access processed output
    int outputData[NUM_VALS];
    checkCudaErrors(cudaMemcpy(outputData, output, NUM_VALS * sizeof(int), cudaMemcpyDeviceToHost));

    // Display result
    std::cout << "blockDim.x: " << outputData[0] << std::endl;
    std::cout << "blockDim.y: " << outputData[1] << std::endl;
    std::cout << "blockDim.z: " << outputData[2] << std::endl;
    std::cout << "gridDim.x: " << outputData[3] << std::endl;
    std::cout << "gridDim.y: " << outputData[4] << std::endl;
    std::cout << "gridDim.z: " << outputData[5] << std::endl;
    std::cout << "warpSize: " << outputData[6] << std::endl;

    // Deallocate device memory
    cudaFree(output);

    return 0;
}
