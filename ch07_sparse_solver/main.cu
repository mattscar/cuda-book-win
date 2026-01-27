#include <algorithm>
#include <cuda_runtime.h>
#include <cudss.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

// Check for CUDA errors
inline void checkErr(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << file << " at line " << line << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}
#define checkCudaErrors(err) (checkErr((err), __FILE__, __LINE__))

// Check for cuDSS errors
inline void checkcudssErr(cudssStatus_t status, const char* file, int line) {
    if (status != CUDSS_STATUS_SUCCESS) {
        std::cerr << "cuDSS error in " << file << " at line " << line << ": ";
        switch (status) {
        case CUDSS_STATUS_NOT_INITIALIZED:
            std::cerr << "not initialized" << std::endl; break;
        case CUDSS_STATUS_ALLOC_FAILED:
            std::cerr << "allocation failed" << std::endl; break;
        case CUDSS_STATUS_INVALID_VALUE:
            std::cerr << "invalid value" << std::endl; break;
        case CUDSS_STATUS_EXECUTION_FAILED:
            std::cerr << "execution failed" << std::endl; break;
        case CUDSS_STATUS_INTERNAL_ERROR:
            std::cerr << "internal error" << std::endl; break;
        case CUDSS_STATUS_NOT_SUPPORTED:
            std::cerr << "not supported" << std::endl; break;
        default:
            std::cerr << "unknown error" << std::endl; break;
        }
        std::exit(EXIT_FAILURE);
    }
}
#define checkcudssError(err) (checkcudssErr((err), __FILE__, __LINE__))

int main() {

    // Load MTX file
    std::ifstream infile("mcca.mtx");
    if (!infile) {
        std::cerr << "Error: could not open matrix file" << std::endl;
        return 1;
    }

    // Skip comments
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '%') continue;
        break;
    }

    // Read matrix dimensions
    std::stringstream ss(line);
    int rows, cols, nnz;
    ss >> rows >> cols >> nnz;

    // Create data containers on host
    std::vector<int> rowStarts(rows + 1, 0);
    std::vector<int> colIndices(nnz);
    std::vector<double> values(nnz);
    std::vector<double> rhs(rows, 0.0);
    std::vector<std::tuple<int, int, double>> entries(nnz);

    // Create vector entries
    int rowIndex, colIndex;
    double value;
    for (int i = 0; i < nnz; i++) {
        infile >> rowIndex >> colIndex >> value;
        entries[i] = std::make_tuple(rowIndex - 1, colIndex - 1, value);
        rowStarts[rowIndex]++;
        rhs[rowIndex - 1] += value;
    }
    infile.close();

    // Update rowStarts array
    for (int i = 1; i < rowStarts.size(); ++i) {
        rowStarts[i] += rowStarts[i - 1];
    }

    // Sort file entries
    std::sort(entries.begin(), entries.end(),
        [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });

    // Get column indices and values
    for (int i = 0; i < nnz; i++) {
        colIndices[i] = std::get<1>(entries[i]);
        values[i] = std::get<2>(entries[i]);
    }

    // Allocate device memory
    int *rowMem, *colMem;
    double *valMem, *xMem, *bMem;
    checkCudaErrors(cudaMalloc((void**)&rowMem, (rows + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&colMem, nnz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void**)&valMem, nnz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&xMem, cols * sizeof(double)));
    checkCudaErrors(cudaMalloc((void**)&bMem, rows * sizeof(double)));

    // Copy data to device
    checkCudaErrors(cudaMemcpy(rowMem, rowStarts.data(), (rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(colMem, colIndices.data(), nnz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(valMem, values.data(), nnz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(bMem, rhs.data(), rows * sizeof(double), cudaMemcpyHostToDevice));

    // Initialize cuDSS data structures
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t solver_data;
    checkcudssError(cudssCreate(&handle));
    checkcudssError(cudssConfigCreate(&config));
    checkcudssError(cudssDataCreate(handle, &solver_data));

    // Create matrices
    cudssMatrix_t aMat, xMat, bMat;
    checkcudssError(cudssMatrixCreateCsr(&aMat, rows, cols, nnz, rowMem, NULL, colMem, valMem,
        CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));

    // Create Vector x
    checkcudssError(cudssMatrixCreateDn(&xMat, rows, 1, rows, xMem, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // Create Vector b
    checkcudssError(cudssMatrixCreateDn(&bMat, rows, 1, rows, bMem, CUDA_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    // Perform reordering and symbolic factorization
    checkcudssError(cudssExecute(handle, CUDSS_PHASE_ANALYSIS, config, solver_data, aMat, xMat, bMat));

    // Perform factorization
    checkcudssError(cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, config, solver_data, aMat, xMat, bMat));

    // Solve the system
    checkcudssError(cudssExecute(handle, CUDSS_PHASE_SOLVE, config, solver_data, aMat, xMat, bMat));

    // Copy results to host
    std::vector<double> xData(rows);
    checkCudaErrors(cudaMemcpy(xData.data(), xMem, rows * sizeof(double), cudaMemcpyDeviceToHost));

    // Check results
    bool pass = true;
    for (int i = 0; i < rows; i++) {
        if (std::abs(xData[i] - 1.0) > 0.01) {
            pass = false;
            break;
        }
    }
    std::cout << (pass ? "Solution check passed" : "Solution check failed") << std::endl;

    // Destroy data structures
    cudssMatrixDestroy(aMat);
    cudssMatrixDestroy(xMat);
    cudssMatrixDestroy(bMat);
    cudssDataDestroy(handle, solver_data);
    cudssConfigDestroy(config);
    cudssDestroy(handle);

    // Free device memory
    cudaFree(rowMem);
    cudaFree(colMem);
    cudaFree(valMem);
    cudaFree(xMem);
    cudaFree(bMem);
    return 0;
}