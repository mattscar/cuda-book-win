#include <iostream>
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// Tile dimensions
const int TILE_M = 16;
const int TILE_N = 16;
const int TILE_K = 16;

// Matrix dimensions (must be multiples of tile dimensions)
const int M = 128;
const int N = 128;
const int K = 128;

inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

__global__ void wmma_demo(const half* A, const half* B, float* C) {

    // Warp responsible for tile at (tileRow, tileCol)
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    // Initialize Matrix C (the accumulator)
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_K, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    // Loop over the K dimension in tiles
    for (int tileK = 0; tileK < K / TILE_K; tileK++) {

        // Offset pointers
        const half* tileA = A + (tileRow * TILE_M) * K + tileK * TILE_K;
        const half* tileB = B + (tileK * TILE_K) * N + tileCol * TILE_N;

        // Create fragment A
        wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_K,
            half, wmma::row_major> frag_a;

        // Create fragment B
        wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_K,
            half, wmma::col_major> frag_b;

        // Load matrix data
        wmma::load_matrix_sync(frag_a, tileA, K);
        wmma::load_matrix_sync(frag_b, tileB, N);

        // Perform MMA operation
        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    // Store the output C tile
    float* tileC = C + (tileRow * TILE_M) * N + tileCol * TILE_N;
    wmma::store_matrix_sync(tileC, frag_c, N, wmma::mem_row_major);
}

int main() {

    // Host matrix A
    half *hostMatrixA = new half[M * K];
    for (int i = 0; i < M * K; i++) {
        hostMatrixA[i] = __float2half(1.0f);
    }

    // Host matrix B
    half* hostMatrixB = new half[K * N];
    for (int i = 0; i < K * N; i++) {
        hostMatrixB[i] = __float2half(1.0f);
    }

    // Device matrix A
    half* devMatrixA;
    checkCudaErrors(cudaMalloc(&devMatrixA, sizeof(half) * M * K));
    checkCudaErrors(cudaMemcpy(devMatrixA, hostMatrixA, sizeof(half) * M * K, cudaMemcpyHostToDevice));

    // Device matrix B
    half* devMatrixB;
    checkCudaErrors(cudaMalloc(&devMatrixB, sizeof(half) * K * N));
    checkCudaErrors(cudaMemcpy(devMatrixB, hostMatrixB, sizeof(half) * K * N, cudaMemcpyHostToDevice));

    // Host and device Matrix C
    float* hostMatrixC = new float[M * N];
    float* devMatrixC;
    checkCudaErrors(cudaMalloc(&devMatrixC, sizeof(float) * M * N));

    // One warp per thread block
    dim3 blockDim(32, 1, 1);

    // M/16 by N/16 tiles in each matrix
    dim3 gridDim(N / 16, M / 16);

    // Execute kernel
    wmma_demo<<<gridDim, blockDim>>>(devMatrixA, devMatrixB, devMatrixC);

    // Wait for kernel to execute
    cudaDeviceSynchronize();

    // Copy data from device to host
    checkCudaErrors(cudaMemcpy(hostMatrixC, devMatrixC, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

    // Display results
    std::cout << "C[0] = " << hostMatrixC[0] << std::endl;
    std::cout << "Expected value: 128\n";

    // Free device memory
    checkCudaErrors(cudaFree(devMatrixA));
    checkCudaErrors(cudaFree(devMatrixB));
    checkCudaErrors(cudaFree(devMatrixC));

    // Free host memory
    delete[] hostMatrixA;
    delete[] hostMatrixB;
    delete[] hostMatrixC;
}