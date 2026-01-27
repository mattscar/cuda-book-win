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

    // Allocate buffers
    const int N = 6;
    CUdeviceptr inPtr, outPtr;
    checkCudaErrors(cuMemAlloc(&inPtr, N * sizeof(float)));
    checkCudaErrors(cuMemAlloc(&outPtr, N * sizeof(float)));

    // Copy input data to input buffer
    float inData[N] = { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f };
    checkCudaErrors(cuMemcpyHtoD(inPtr, inData, N * sizeof(float)));

    // Load module
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, "array_reverse.ptx"));

    // Access the kernel
    CUfunction kernel;
    checkCudaErrors(cuModuleGetFunction(&kernel, module, "array_reverse"));

    // Create array of kernel parameters
    void* params[] = { &inPtr, &outPtr, (void*)&N};

    // Execute kernel
    checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0,
        nullptr, params, nullptr));

    // Wait for kernel execution to complete
    checkCudaErrors(cuCtxSynchronize());

    // Copy result to host
    float outData[N];
    checkCudaErrors(cuMemcpyDtoH(outData, outPtr, N * sizeof(float)));

    // Display input data
    std::cout << "Input data: ";
    for (int i = 0; i < N; i++) {
        std::cout << inData[i] << " ";
    }
    std::cout << std::endl;

    // Display output data
    std::cout << "Output data: ";
        for (int i = 0; i < N; i++) {
            std::cout << outData[i] << " ";
        }
    std::cout << std::endl;

    // Free resources
    checkCudaErrors(cuMemFree(inPtr));
    checkCudaErrors(cuMemFree(outPtr));
    checkCudaErrors(cuModuleUnload(module));
    checkCudaErrors(cuCtxDestroy(context));
    return 0;
}