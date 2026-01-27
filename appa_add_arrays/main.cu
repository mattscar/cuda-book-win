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
    const int N = 6;

    // Initialize the driver
    checkCudaErrors(cuInit(0));

    // Access the device
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));

    // Create the context
    CUcontext context;
    checkCudaErrors(cuCtxCreate(&context, nullptr, 0, device));

    // Allocate buffers
    CUdeviceptr arrayA, arrayB, arrayC;
    checkCudaErrors(cuMemAlloc(&arrayA, N * sizeof(float)));
    checkCudaErrors(cuMemAlloc(&arrayB, N * sizeof(float)));
    checkCudaErrors(cuMemAlloc(&arrayC, N * sizeof(float)));

    // Copy values of first input array
    float aData[N] = { 2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f };
    checkCudaErrors(cuMemcpyHtoD(arrayA, aData, N * sizeof(float)));

    // Copy values of second input array
    float bData[N] = { 3.0f, 6.0f, 9.0f, 12.0f, 15.0f, 18.0f };
    checkCudaErrors(cuMemcpyHtoD(arrayB, bData, N * sizeof(float)));

    // Load module
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, "add_arrays.ptx"));

    // Access the kernel
    CUfunction kernel;
    checkCudaErrors(cuModuleGetFunction(&kernel, module, "add_arrays"));

    // Create array of kernel parameters
    void* params[] = { &arrayA, &arrayB, &arrayC };

    // Execute kernel
    checkCudaErrors(cuLaunchKernel(kernel, 1, 1, 1, N, 1, 1, 0,
        nullptr, params, nullptr));

    // Wait for kernel execution to complete
    checkCudaErrors(cuCtxSynchronize());

    // Copy result to host
    float cData[N];
    checkCudaErrors(cuMemcpyDtoH(cData, arrayC, N * sizeof(float)));

    // Display result
    for (int i = 0; i < N; i++) {
        std::cout << aData[i] << " plus " << bData[i] << " equals " << cData[i] << std::endl;
    }

    // Free resources
    checkCudaErrors(cuMemFree(arrayA));
    checkCudaErrors(cuMemFree(arrayB));
    checkCudaErrors(cuMemFree(arrayC));
    checkCudaErrors(cuModuleUnload(module));
    checkCudaErrors(cuCtxDestroy(context));
    return 0;
}
