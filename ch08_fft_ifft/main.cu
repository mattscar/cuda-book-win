#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <cufft.h>
#include <cuda_runtime.h>

#define M_PI 3.14159265358979323846

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

// Check for cuFFT errors
inline void checkcuFftErr(cufftResult_t err, const char* file, int line) {
    if (err != CUFFT_SUCCESS) {
        std::cerr << "cuFFT error in " << file << " at line " << line << ": ";
        switch (err) {
            case CUFFT_INVALID_PLAN:
                std::cerr << "CUFFT_INVALID_PLAN: The cuFFT plan is invalid." << std::endl;
            case CUFFT_ALLOC_FAILED:
                std::cerr << "CUFFT_ALLOC_FAILED: Failed to allocate memory." << std::endl;
            case CUFFT_INVALID_TYPE:
                std::cerr << "CUFFT_INVALID_TYPE: The requested transform type is not supported." << std::endl;
            case CUFFT_INVALID_VALUE:
                std::cerr << "CUFFT_INVALID_VALUE: Invalid value for a parameter." << std::endl;
            case CUFFT_INTERNAL_ERROR:
                std::cerr << "CUFFT_INTERNAL_ERROR: An internal driver error occurred." << std::endl;
            case CUFFT_EXEC_FAILED:
                std::cerr << "CUFFT_EXEC_FAILED: Failed to execute an FFT on the GPU." << std::endl;
            case CUFFT_SETUP_FAILED:
                std::cerr << "CUFFT_SETUP_FAILED: CUFFT library failed to initialize." << std::endl;
            case CUFFT_INVALID_SIZE:
                std::cerr << "CUFFT_INVALID_SIZE: The transform size is not supported." << std::endl;
            case CUFFT_UNALIGNED_DATA:
                std::cerr << "CUFFT_UNALIGNED_DATA: The input/output data is not aligned properly." << std::endl;
            case CUFFT_INVALID_DEVICE:
                std::cerr << "CUFFT_INVALID_DEVICE: Execution of a plan was on different GPU than plan creation." << std::endl;
            case CUFFT_NO_WORKSPACE:
                std::cerr << "CUFFT_NO_WORKSPACE: No workspace has been provided." << std::endl;
            case CUFFT_NOT_IMPLEMENTED:
                std::cerr << "CUFFT_NOT_IMPLEMENTED: The requested feature is not implemented." << std::endl;
            case CUFFT_NOT_SUPPORTED:
                std::cerr << "CUFFT_NOT_SUPPORTED: The requested feature is not supported." << std::endl;
            case CUFFT_MISSING_DEPENDENCY:
                std::cerr << "CUFFT_MISSING_DEPENDENCY: cuFFT is unable to find a dependency." << std::endl;
            case CUFFT_NVRTC_FAILURE:
                std::cerr << "CUFFT_NVRTC_FAILURE: An NVRTC failure was encountered." << std::endl;
            case CUFFT_NVJITLINK_FAILURE:
                std::cerr << "CUFFT_NVJITLINK_FAILURE: An nvJitLink failure was encountered." << std::endl;
            case CUFFT_NVSHMEM_FAILURE:
                std::cerr << "CUFFT_NVSHMEM_FAILURE: An NVSHMEM failure was encountered." << std::endl;
            default:
                std::cerr << "Unknown cuFFT error." << std::endl;
        }
    }
}
#define checkcuFftError(err) (checkcuFftErr((err), __FILE__, __LINE__))

int main() {
    const int N = 128;

    // Allocate device memory
    int complexSize = N / 2 + 1;
    float *inputMem, *ifftOutputMem;
    cufftComplex* fftOutputMem;
    checkCudaErrors(cudaMalloc(&inputMem, N * sizeof(float)));
    checkCudaErrors(cudaMalloc(&fftOutputMem, complexSize * sizeof(cufftComplex)));
    checkCudaErrors(cudaMalloc(&ifftOutputMem, sizeof(float) * N));

    // Fill input with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0f, 100.0f);
    std::vector<float> inputData(N);
    std::generate(inputData.begin(), inputData.end(), [&]() { return dist(gen); });
    checkCudaErrors(cudaMemcpy(inputMem, inputData.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Create plan for forward FFT
    cufftHandle fwdHandle;
    checkcuFftError(cufftPlan1d(&fwdHandle, N, CUFFT_R2C, 1));

    // Execute forward FFT
    checkcuFftError(cufftExecR2C(fwdHandle, inputMem, fftOutputMem));

    // Create plan for inverse IFFT
    cufftHandle invHandle;
    checkcuFftError(cufftPlan1d(&invHandle, N, CUFFT_C2R, 1));

    // Execute inverse FFT
    checkcuFftError(cufftExecC2R(invHandle, fftOutputMem, ifftOutputMem));

    // Copy result to host
    std::vector<float> outputData(N);
    checkCudaErrors(cudaMemcpy(outputData.data(), ifftOutputMem, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Divide result by N
    for (int i = 0; i < N; i++) {
        outputData[i] /= N;
    }

    // Check error
    float diff, error = 0.0;
    for (size_t i = 0; i<N; i++) {
        diff = inputData[i] - outputData[i];
        error += diff * diff;
    }
    std::cout << "Error: " << error << std::endl;

    // Free resources
    cufftDestroy(fwdHandle);
    cufftDestroy(invHandle);
    cudaFree(inputMem);
    cudaFree(fftOutputMem);
    cudaFree(ifftOutputMem);
    return 0;
}