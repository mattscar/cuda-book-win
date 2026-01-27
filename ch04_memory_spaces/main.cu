#include <cuda_runtime.h>
#include <iostream>

inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

#define NUM_VALS 10

// Store variable in global memory
__device__ int sum[NUM_VALS];

// Store variable in constant memory
__constant__ int a[NUM_VALS];

// Kernel function
__global__ void addVals() {
    int i = threadIdx.x;
    sum[i] = a[i] + i;
}

int main() {

    // Initialize content of a array
    int aData[NUM_VALS] = { 3, 3, 3, 3, 3, 3, 3, 3, 3, 3 };
    checkCudaErrors(cudaMemcpyToSymbol(a, aData, NUM_VALS * sizeof(int)));

    // Execute kernel
    addVals<<<1, NUM_VALS>>>();

    // Access processed output
    int sumData[NUM_VALS];
    checkCudaErrors(cudaMemcpyFromSymbol(sumData, sum, NUM_VALS * sizeof(int)));

    // Display results
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < NUM_VALS; i++) {
        std::cout << sumData[i] << std::endl;
    }
    return 0;
}