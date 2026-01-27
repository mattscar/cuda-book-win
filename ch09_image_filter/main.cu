#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <nppi.h>
#include <nppi_filtering_functions.h>
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
    unsigned char* inMem, * outMem;
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&inMem), width * height, ctx.hStream));
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&outMem), width * height, ctx.hStream));

    // Copy image data to the device
    checkCudaErrors(cudaMemcpyAsync(inMem, imgData, width * height, cudaMemcpyHostToDevice, ctx.hStream));

    // Define kernel of sharpening filter
    Npp32s kernel[9] = {
        0, -1, 0,
        -1, 5, -1,
        0, -1, 0
    };
    NppiSize kernelSize = { 3, 3 };
    NppiPoint anchor = { 1, 1 };
    int divisor = 1;

    // Allocate device memory for the kernel
    Npp32s* kernelMem;
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&kernelMem), 9 * sizeof(Npp32s), ctx.hStream));

    // Copy the kernel to the device
    checkCudaErrors(cudaMemcpyAsync(kernelMem, kernel, 9 * sizeof(Npp32s), cudaMemcpyHostToDevice, ctx.hStream));

    // Filter image
    int step = width;
    NppiSize size = { width, height };
    NppiPoint offset = { 0, 0 };
    checkNppError(nppiFilterBorder_8u_C1R_Ctx(inMem, step, size, offset, outMem, step, size, kernelMem,
        kernelSize, anchor, divisor, NPP_BORDER_REPLICATE, ctx));

    // Copy filtered image to the host
    checkCudaErrors(cudaMemcpyAsync(imgData, outMem, width * height, cudaMemcpyDeviceToHost, ctx.hStream));

    // Wait for stream to finish
    cudaStreamSynchronize(ctx.hStream);

    // Write data to new image file
    stbi_write_png("filtered_smiley.png", width, height, 1, imgData, step);

    // Free resources
    cudaFreeAsync(kernelMem, ctx.hStream);
    cudaFreeAsync(inMem, ctx.hStream);
    cudaFreeAsync(outMem, ctx.hStream);
    cudaStreamDestroy(ctx.hStream);
    stbi_image_free(imgData);
    return 0;
}
