#include <cstdio>

#include <curand.h>
#include <cublas_v2.h>

#include "kernelFunc.cuh"
#include "hostFunc.hpp"
#include "cudaErrorCheck.cuh"
#include "matrixSetting.hpp"

int main() {
    float *aFp32;
    float *bFp32;

    half *aFp16;
    half *bFp16;

    float *cMmaExampleCommon;
    float *cCublasGemmEx;
    float *cWmmaExample1DGrid;
    float *cWmmaExample2DGrid;
    float *cWmmaExample2DGrid2;
    float *cWmmaExample2DGrid3;

    const float alpha = 2.0f;
    const float beta = 2.0f;

    // Allocated memory in the global memory of the GPU
    {
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&aFp32), MATRIX_A_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&bFp32), MATRIX_B_SIZE * sizeof(float)));

        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&aFp16), MATRIX_A_SIZE * sizeof(half)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&bFp16), MATRIX_B_SIZE * sizeof(half)));

        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cMmaExampleCommon), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cCublasGemmEx), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmmaExample1DGrid), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmmaExample2DGrid), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmmaExample2DGrid2), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmmaExample2DGrid3), MATRIX_C_SIZE * sizeof(float)));
    }

    // using cuRAND to initialize
    {
        curandGenerator_t curandGen;

        curandErrCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL));

        curandErrCheck(curandGenerateUniform(curandGen, aFp32, MATRIX_A_SIZE));
        curandErrCheck(curandGenerateUniform(curandGen, bFp32, MATRIX_B_SIZE));

        const int numThreadPerBlock = 256;
        convertFp32ToFp16<<< (MATRIX_A_SIZE + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            MATRIX_A_SIZE, aFp32, aFp16);
        convertFp32ToFp16<<< (MATRIX_B_SIZE + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            MATRIX_B_SIZE, bFp32, bFp16);

        float *c;
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&c), MATRIX_C_SIZE * sizeof(float)));
        curandErrCheck(curandGenerateUniform(curandGen, c, MATRIX_C_SIZE));

        cudaErrCheck(cudaMemcpy(cMmaExampleCommon, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cCublasGemmEx, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample1DGrid, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid2, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid3, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));

        curandErrCheck(curandDestroyGenerator(curandGen));
    }

    // using mmaExampleCommon computation
    {
        const int numThreadPerBlocks = 1024;
        const int numBlocks = (MATRIX_C_SIZE + numThreadPerBlocks - 1) / numThreadPerBlocks;
        mmaExampleCommon<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                            alpha, beta,
                                                            aFp16, bFp16, cMmaExampleCommon);
    }

    // using cuBLAS computation
    {
        printf("---------------------------\n");
        printf("Running with cuBLAS...\n");

        cudaEvent_t startCublas;
        cudaEvent_t stopCublas;

        cudaErrCheck(cudaEventCreate(&startCublas));
        cudaErrCheck(cudaEventCreate(&stopCublas));

        cublasHandle_t cublasHandle;
        cublasErrCheck(cublasCreate(&cublasHandle));

        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cudaErrCheck(cudaEventRecord(startCublas));
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_T,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    aFp16, CUDA_R_16F, MATRIX_M,
                                    bFp16, CUDA_R_16F, MATRIX_K,
                                    &beta,
                                    cCublasGemmEx, CUDA_R_32F, MATRIX_M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaErrCheck(cudaEventRecord(stopCublas));
        cudaErrCheck(cudaEventSynchronize(stopCublas));

        float cublasTime;
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startCublas, stopCublas));
        printf("cublasGemmEx time : %fms\n", cublasTime);

        cublasErrCheck(cublasDestroy(cublasHandle));

        cudaErrCheck(cudaEventDestroy(startCublas));
        cudaErrCheck(cudaEventDestroy(stopCublas));
    }

    // using wmmaExample1DGrid computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample1DGrid...\n");

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        const int wmmaCalculatesOneResultTileSize = WMMA_M * WMMA_N;
        int numThreadPerBlocks = WARP_SIZE * 1;
        int numBlocks = (MATRIX_C_SIZE / wmmaCalculatesOneResultTileSize * WARP_SIZE + numThreadPerBlocks - 1)
            / numThreadPerBlocks;

        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmmaExample1DGrid<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                             alpha, beta,
                                                             aFp16, bFp16, cWmmaExample1DGrid);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample1DGrid time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    // using wmmaExample2DGrid computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample2DGrid...\n");

        cudaEvent_t start;
        cudaEvent_t stop;

        cudaErrCheck(cudaEventCreate(&start));
        cudaErrCheck(cudaEventCreate(&stop));

        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = WARP_SIZE;
        blockDim.y = WARP_SIZE;

        const int numCountRowOfOutputMatrixPerBlock = (int) (WMMA_M * blockDim.x / 32);
        const int numCountColOfOutputMatrixPerBlock = (int) (WMMA_N * blockDim.y);
        gridDim.x = (MATRIX_M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
        gridDim.y = (MATRIX_N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;

        cudaErrCheck(cudaEventRecord(start));
        wmmaExample2DGrid<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                 alpha, beta,
                                                 aFp16, bFp16, cWmmaExample2DGrid);
        cudaErrCheck(cudaEventRecord(stop));
        cudaErrCheck(cudaEventSynchronize(stop));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, start, stop));
        printf("wmmaExample2DGrid time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(start));
        cudaErrCheck(cudaEventDestroy(stop));
    }

    // using wmmaExample2DGrid2 computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample2DGrid2...\n");

        cudaEvent_t start2;
        cudaEvent_t stop2;

        cudaErrCheck(cudaEventCreate(&start2));
        cudaErrCheck(cudaEventCreate(&stop2));

        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = WARP_SIZE;
        blockDim.y = WARP_SIZE;

        const int numCountRowOfOutputMatrixPerBlock = (int) (WMMA_M * blockDim.x / 32);
        const int numCountColOfOutputMatrixPerBlock = (int) (WMMA_N * blockDim.y);
        gridDim.x = (MATRIX_M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
        gridDim.y = (MATRIX_N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;

        cudaErrCheck(cudaEventRecord(start2));
        wmmaExample2DGrid2<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                  alpha, beta,
                                                  aFp16, bFp16, cWmmaExample2DGrid2);

        cudaErrCheck(cudaEventRecord(stop2));
        cudaErrCheck(cudaEventSynchronize(stop2));
        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, start2, stop2));
        printf("wmmaExample2DGrid2 time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(start2));
        cudaErrCheck(cudaEventDestroy(stop2));
    }

    // using wmmaExample2DGrid3 computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample2DGrid3...\n");

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multiple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);
        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmmaExample2DGrid3<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                  alpha, beta,
                                                  aFp16, bFp16, cWmmaExample2DGrid3);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample2DGrid3 time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    if (!checkDevData(MATRIX_C_SIZE, cCublasGemmEx, cWmmaExample2DGrid3)) {
        printf("Error! Function cublasGemmEx, wmmaExample2DGrid3 Check no passes!\n");
    } else {
        printf("Function cublasGemmEx, wmmaExample2DGrid3 Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cMmaExampleCommon, cWmmaExample1DGrid)) {
        printf("Error! Function mmaExampleCommon, wmmaExample1DGrid Check no passes!\n");
    } else {
        printf("Function mmaExampleCommon, wmmaExample1DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cMmaExampleCommon, cWmmaExample2DGrid)) {
        printf("Error! Function mmaExampleCommon, wmmaExample2DGrid Check no passes!\n");
    } else {
        printf("Function mmaExampleCommon, wmmaExample2DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cWmmaExample1DGrid, cWmmaExample2DGrid)) {
        printf("Error! Function wmmaExample1DGrid, wmmaExample2DGrid Check no passes!\n");
    } else {
        printf("Function wmmaExample1DGrid, wmmaExample2DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cCublasGemmEx, cWmmaExample2DGrid2)) {
        printf("Error! Function cublasGemmEx, wmmaExample2DGrid2 Check no passes!\n");
    } else {
        printf("Function cublasGemmEx, wmmaExample2DGrid2 Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cWmmaExample2DGrid, cWmmaExample2DGrid2)) {
        printf("Error! Function wmmaExample2DGrid, wmmaExample2DGrid2 Check no passes!\n");
    } else {
        printf("Function wmmaExample2DGrid, wmmaExample2DGrid2 Check passes!\n");
    }

    cudaErrCheck(cudaFree(aFp32));
    cudaErrCheck(cudaFree(bFp32));
    cudaErrCheck(cudaFree(aFp16));
    cudaErrCheck(cudaFree(bFp16));
    cudaErrCheck(cudaFree(cCublasGemmEx));
    cudaErrCheck(cudaFree(cWmmaExample2DGrid3));
    cudaErrCheck(cudaFree(cWmmaExample1DGrid));
    cudaErrCheck(cudaFree(cWmmaExample2DGrid));
    cudaErrCheck(cudaFree(cWmmaExample2DGrid2));

    return 0;
}