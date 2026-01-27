#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

// Kernel function
__global__ void processHalf(const __half a, const float b,
    __half* c) {

    c[0] = a * (__half)b;
}

int main() {

    __half halfVal = (__half)7.0f;
    float floatVal = 3.0f;

    // Allocate buffer for the output half array
    __half* outputMem;
    checkCudaErrors(cudaMalloc(
        reinterpret_cast<void**>(&outputMem), sizeof(__half)));

    // Execute kernel
    processHalf << <1, 1 >> > (halfVal, floatVal, outputMem);

    // Access processed output
    __half result;
    checkCudaErrors(cudaMemcpy(&result, outputMem,
        sizeof(__half), cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "Output: " << (float)result << std::endl;

    // Free memory blocks
    cudaFree(outputMem);
    return 0;
}