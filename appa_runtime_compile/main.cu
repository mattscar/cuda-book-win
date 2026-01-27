#include <iostream>
#include <nvrtc.h>
#include <cuda.h>

// Check for NVRTC errors
inline void checkNvrtcErr(nvrtcResult err, const char* file, int line) {
    if (err != NVRTC_SUCCESS) {
        std::cerr << "NVRTC error in " << file <<
            " at line " << line << ": " << nvrtcGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkNvrtcErrors(err) (checkNvrtcErr((err), __FILE__, __LINE__))

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

// Kernel code
const char* kernelSrc = R"(
#include "header.h"
extern "C" __global__
void add_arrays(float *a, float *b, float *c) {
    int i = threadIdx.x;
    c[i] = foo(a[i], b[i]);
}
)";

// Header code
const char* headerSrc = R"(
__device__ float foo(float x, float y) {
    return x + y;
}
)";

int main() {

    // Prepare header arrays
    const char* headers[] = { headerSrc };
    const char* includeNames[] = { "header.h" };

    // Create the program
    nvrtcProgram prog;
    checkNvrtcErrors(nvrtcCreateProgram(&prog, kernelSrc,
        "runtime_compile.cu", 1, headers, includeNames));

    // Compile the program
    const char* options[] = { "-arch=compute_75", "--time=-" };
    nvrtcResult compileResult = nvrtcCompileProgram(prog, 2, options);

    // Get the size of the compilation log
    size_t logSize;
    checkNvrtcErrors(nvrtcGetProgramLogSize(prog, &logSize));

    // Print the log if not empty
    if (logSize < 2) {
        std::cout << "Log empty" << std::endl;
    }
    else {
        std::string log(logSize, '\0');
        checkNvrtcErrors(nvrtcGetProgramLog(prog, &log[0]));
        std::cout << "Compiler log:" << log << std::endl;
    }

    // Destroy the program if compilation failed
    if (compileResult != NVRTC_SUCCESS) {
        std::cerr << "Compilation failed." << std::endl;
        nvrtcDestroyProgram(&prog);
        return 1;
    }

    // Access the compiled PTX
    size_t ptxSize;
    checkNvrtcErrors(nvrtcGetPTXSize(prog, &ptxSize));
    std::string ptxCode(ptxSize, '\0');
    checkNvrtcErrors(nvrtcGetPTX(prog, &ptxCode[0]));

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
    checkCudaErrors(cuModuleLoadData(&module, ptxCode.data()));

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
    checkNvrtcErrors(nvrtcDestroyProgram(&prog));
    checkCudaErrors(cuMemFree(arrayA));
    checkCudaErrors(cuMemFree(arrayB));
    checkCudaErrors(cuMemFree(arrayC));
    checkCudaErrors(cuModuleUnload(module));
    checkCudaErrors(cuCtxDestroy(context));
    return 0;
}
