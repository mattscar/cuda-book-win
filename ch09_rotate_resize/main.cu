#pragma nv_diag_suppress 550
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <nppi.h>
#include <nppi_geometry_transforms.h>
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
    unsigned char* inData = stbi_load("smiley.png", &width, &height, &channels, 1);
    if (!inData) {
        std::cerr << "Error: failed to load image" << std::endl;
        return -1;
    }

    // Allocate memory on the device
    unsigned char *imgMem1;
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&imgMem1), width * height, ctx.hStream));

    // Copy image data to the device
    checkCudaErrors(cudaMemcpyAsync(imgMem1, inData, width * height, cudaMemcpyHostToDevice, ctx.hStream));

    // Get the bounds of rotation
    double boundingBox[2][2];
    NppiRect srcRoi = { 0, 0, width, height };
    float shiftX = width / 2.0f, shiftY = height / 2.0f;
    checkNppError(nppiGetRotateBound(srcRoi, boundingBox, 45.0, shiftX, shiftY));

    // Allocate memory for rotated image
    unsigned char* imgMem2;
    int dstWidth = static_cast<int>(ceil(boundingBox[1][0] - boundingBox[0][0]));
    int dstHeight = static_cast<int>(ceil(boundingBox[1][1] - boundingBox[0][1]));
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&imgMem2), dstWidth * dstHeight, ctx.hStream));

    // Rotate image
    int srcStep = width, dstStep = dstWidth;
    double sy = ceil(dstHeight / 2.0);
    NppiSize size = { width, height };
    NppiRect dstRoi = { 0, 0, dstWidth, dstHeight };
    checkNppError(nppiRotate_8u_C1R_Ctx(imgMem1, size, srcStep, srcRoi, imgMem2, 
        dstStep, dstRoi, 45.0, 0.0, sy, NPPI_INTER_CUBIC, ctx));

    // Allocate memory for rotated image
    unsigned char* imgMem3;
    double xFactor = 2.0, yFactor = 2.0;
    int outWidth = dstWidth * static_cast<int>(xFactor);
    int outHeight = dstHeight * static_cast<int>(yFactor);
    checkCudaErrors(cudaMallocAsync(reinterpret_cast<void**>(&imgMem3), outWidth * outHeight, ctx.hStream));

    // Resize image
    size = { dstWidth, dstHeight };
    int outStep = outWidth;
    NppiRect outRoi = { 0, 0, outWidth, outHeight };
    checkNppError(nppiResizeSqrPixel_8u_C1R_Ctx(imgMem2, size, dstStep, dstRoi, imgMem3, outStep, outRoi, 
        xFactor, yFactor, 0, 2, NPPI_INTER_CUBIC, ctx));

    // Copy image data to the host
    unsigned char* outData = (unsigned char*)malloc(outWidth * outHeight);
    checkCudaErrors(cudaMemcpyAsync(outData, imgMem3, outWidth * outHeight, cudaMemcpyDeviceToHost, ctx.hStream));

    // Wait for stream to finish
    cudaStreamSynchronize(ctx.hStream);

    // Write data to new image file
    stbi_write_png("new_smiley.png", outWidth, outHeight, 1, outData, outStep);

    // Free resources
    cudaFreeAsync(imgMem1, ctx.hStream);
    cudaFreeAsync(imgMem2, ctx.hStream);
    cudaFreeAsync(imgMem3, ctx.hStream);
    cudaStreamDestroy(ctx.hStream);
    stbi_image_free(inData);
    free(outData);
    return 0;
}
