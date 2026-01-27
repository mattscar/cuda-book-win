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

int main() {

    int deviceCount = 0, maxThreads = -1, maxIndex = -1;
    cudaDeviceProp prop;

    // Determine the number of connected compatible devices
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    // Iterate through devices
    for (int i = 0; i < deviceCount; i++) {

        // Get properties of device
        checkCudaErrors(cudaGetDeviceProperties(&prop, i));

        // Display name, major, minor properties:
        std::cout << "Device " << prop.name << ": " 
            << prop.major << "." << prop.minor << std::endl;

        // Check maximum threads per block
        if (prop.maxThreadsPerBlock > maxThreads) {
            maxThreads = prop.maxThreadsPerBlock;
            maxIndex = i;
        }
    }

    // Set the current device to the one with the most threads per block
    checkCudaErrors(cudaSetDevice(maxIndex));

    return 0;
}