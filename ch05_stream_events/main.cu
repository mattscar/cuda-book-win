#include <cstdio>
#include <iostream>
#include <cuda_runtime.h>

inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

__global__ void kernel1() {
    printf("Kernel 1 running on Stream 1\n");
}

__global__ void kernel2() {
    printf("Kernel 2 running on Stream 2\n");
}

int main() {
    
    // Create two streams
    cudaStream_t stream1, stream2;
    checkCudaErrors(cudaStreamCreate(&stream1));
    checkCudaErrors(cudaStreamCreate(&stream2));

    // Create event
    cudaEvent_t event;
    checkCudaErrors(cudaEventCreate(&event));

    // Launch Kernel 1 in Stream 1
    kernel1<<<1, 1, 0, stream1>>>();

    // Record event in Stream 1 after Kernel 1
    checkCudaErrors(cudaEventRecord(event, stream1));

    // Make Stream 2 wait until event completes
    checkCudaErrors(cudaStreamWaitEvent(stream2, event, 0));

    // Launch Stream 2 in Stream 2
    kernel2<<<1, 1, 0, stream2>>>();

    // Wait until streams complete commands
    checkCudaErrors(cudaStreamSynchronize(stream1));
    checkCudaErrors(cudaStreamSynchronize(stream2));

    // Free event
    checkCudaErrors(cudaEventDestroy(event));

    // Free streams
    checkCudaErrors(cudaStreamDestroy(stream1));
    checkCudaErrors(cudaStreamDestroy(stream2));
    return 0;
}
