#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <nppi.h>
#include <nppi_arithmetic_and_logical_operations.h>
#include <cuda_runtime.h>

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

// Check for NPP errors
inline void checkNppErr(NppStatus err, const char* file, int line) {
    if (err != NPP_NO_ERROR) {
        std::cerr << "NPP error in " << file << " at line " << line << ": "
            << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkNppError(err) (checkNppErr((err), __FILE__, __LINE__))

int main() {

    // Get device identifier
    int device;
    checkCudaErrors(cudaGetDevice(&device));

    // Get device properties
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));

    // Create stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Define NPPI stream
    NppStreamContext ctx;
    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = deviceProp.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = deviceProp.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = deviceProp.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = deviceProp.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = deviceProp.minor;

    // Read data from image file
    int width, height, channels;
    unsigned char* imgData = stbi_load("smiley.png", &width, &height, &channels, 1);
    if (!imgData) {
        std::cerr << "Error: failed to load image" << std::endl;
        return -1;
    }

    // Allocate memory on the device
    unsigned char* imgMem;
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&imgMem), width * height, ctx.hStream));

    // Copy image data to the device
    checkCudaErrors(cudaMemcpyAsync(imgMem, imgData, width * height, cudaMemcpyHostToDevice, ctx.hStream));

    // Multiply by scaled contrast
    int step = width;
    NppiSize roi = { width, height };
    checkNppError(nppiMulC_8u_C1IRSfs_Ctx(6, imgMem, step, roi, 2, ctx));

    // Subtract 64 from each pixel
    checkNppError(nppiSubC_8u_C1IRSfs_Ctx(64, imgMem, step, roi, 0, ctx));

    // Copy image data to the host
    checkCudaErrors(cudaMemcpyAsync(imgData, imgMem, width * height, cudaMemcpyDeviceToHost, ctx.hStream));

    // Wait for stream to finish
    cudaStreamSynchronize(ctx.hStream);

    // Write data to new image file
    stbi_write_png("new_smiley.png", width, height, 1, imgData, step);

    // Free resources
    cudaFreeAsync(imgMem, ctx.hStream);
    cudaStreamDestroy(ctx.hStream);
    stbi_image_free(imgData);
    return 0;
}
