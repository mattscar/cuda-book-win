#include <iostream>
#include <cuda.h>

// Check for errors
inline void checkErr(CUresult err, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* errorMsg;
        cuGetErrorString(err, &errorMsg);
        std::cerr << "CUDA error in " << file << 
            " at line " << line << ": " << errorMsg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

int main() {

    // Initialize the driver
    checkCudaErrors(cuInit(0));

    // Select the first connected device
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    // Access major version number of compute capability
    int ccMajor;
    checkCudaErrors(cuDeviceGetAttribute(&ccMajor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));

    // Access minor version number of compute capability
    int ccMinor;
    checkCudaErrors(cuDeviceGetAttribute(&ccMinor,
        CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));

    // Access max number of threads per thread block
    int maxThreads;
    checkCudaErrors(cuDeviceGetAttribute(&maxThreads,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));

    // Print results
    std::cout << "Compute capability: " << ccMajor << "." << ccMinor << std::endl;
    std::cout << "Maximum number of threads per block: " << maxThreads << std::endl;
    return 0;
}