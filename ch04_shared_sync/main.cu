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

__global__ void blockReduce(float* input, float* output) {
    extern __shared__ float shared_data[];

    int local_id = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    shared_data[local_id] = input[global_id];

    // Wait until input has been loaded
    __syncthreads();  

    // Perform reduction
    for (int i = blockDim.x/2; i > 0; i >>= 1) {
        if (local_id < i) {
            shared_data[local_id] += shared_data[local_id + i];
        }
        __syncthreads();
    }

    // Transfer result to global memory
    if (local_id == 0) {
        output[0] = shared_data[0];
    }
}

int main() {

    // Determine the max number of threads per block
    int dev, MAX_THREADS;
    cudaGetDevice(&dev);
    checkCudaErrors(cudaDeviceGetAttribute(&MAX_THREADS,
      cudaDevAttrMaxThreadsPerBlock, dev));
    MAX_THREADS /= 2;

    // Allocate memory for input/output memory blocks
    float *inputMem, *outputMem;
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&inputMem), MAX_THREADS * sizeof(float)));
    checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&outputMem), sizeof(float)));

    // Initialize input memory
    float* inputData = new float[MAX_THREADS];
    for (int i = 0; i < MAX_THREADS; i++) {
        inputData[i] = i * 1.0f;
    }
    checkCudaErrors(cudaMemcpy(inputMem, inputData, MAX_THREADS * sizeof(float), cudaMemcpyHostToDevice));

    // Invoke kernel
    blockReduce<<<1, MAX_THREADS, MAX_THREADS * sizeof(float)>>>(inputMem, outputMem);

    // Read computed sum
    float computedSum;
    checkCudaErrors(cudaMemcpy(&computedSum, outputMem, sizeof(float), cudaMemcpyDeviceToHost));

    // Display and check results
    std::cout << "Computed sum: " << computedSum << std::endl;
    std::cout << "Actual sum: " << ((MAX_THREADS / 2) * (MAX_THREADS - 1)) << std::endl;

    cudaFree(inputMem);
    cudaFree(outputMem);
    delete[] inputData;
    return 0;
}
