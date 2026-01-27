#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include <cuda.h>

// Check for Driver API errors
inline void checkDriverErr(CUresult err, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* errorMsg;
        cuGetErrorString(err, &errorMsg);
        std::cerr << "CUDA error in " << file <<
            " at line " << line << ": " << errorMsg << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkDriverErr((err), __FILE__, __LINE__))

int main() {

    // Initialize the driver
    checkCudaErrors(cuInit(0));

    // Access the device
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    // Create the context
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, nullptr, 0, device));

    // Allocate buffer
    CUdeviceptr devPtr;
    checkCudaErrors(cuMemAlloc(&devPtr, sizeof(float)));

    // Load module
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, "simple.ptx"));

    // Access the kernel
    CUfunction kernel;
    checkCudaErrors(cuModuleGetFunction(&kernel, module, "simple"));

    // Create array of kernel parameters
    void* params[] = { &devPtr };

    // Execute kernel
    checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0,
        nullptr, params, nullptr));

    // Wait for kernel execution to complete
    checkCudaErrors(cuCtxSynchronize());

    // Copy result to host
    float res;
    checkCudaErrors(cuMemcpyDtoH(&res, devPtr, sizeof(float)));

    // Display result
    std::cout << "The result is " << res << std::endl;

    // Free resources
    checkCudaErrors(cuMemFree(devPtr));
    checkCudaErrors(cuModuleUnload(module));
    checkCudaErrors(cuCtxDestroy(context));
    return 0;
}