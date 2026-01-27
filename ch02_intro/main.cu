
#include <iostream>

#include "cuda_runtime.h"

// The kernel function to be executed on the GPU
__global__ void timesTwo(const float *num, float* twice_num) {
  twice_num[0] = num[0] * 2.0;
}

int main() {
  float *input, *output;
  float inputVal = 7.0, outputVal;

  // Allocate buffer for input value
  cudaError_t cudaStatus = cudaMalloc(reinterpret_cast<void**>(&input), sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed");
  }

  // Allocate buffer for output value
  cudaStatus = cudaMalloc(reinterpret_cast<void**>(&output), sizeof(float));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed");
  }

  // Set input value to 7.0
  cudaStatus = cudaMemcpy(input, &inputVal, sizeof(float), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed");
  }

  // Multiply input by 2
  timesTwo <<<1, 1>>> (input, output);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "kernel failed");
    return 1;
  }

  // Access processed output
  cudaStatus = cudaMemcpy(&outputVal, output, sizeof(float), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
  }

  // Display result
  std::cout << inputVal << " times two is " << outputVal << std::endl;
  return 0;
}
