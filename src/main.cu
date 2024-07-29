#include <cstdio>

#include <curand.h>
#include <cublas_v2.h>

#include "kernelFunc.cuh"
#include "hostFunc.hpp"
#include "cudaErrorCheck.hpp"
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
    float *cWmma_example;

    const float alpha = 2.0f;
    const float beta = 2.0f;

    /* Allocated memory in the global memory of the GPU */
    {
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&aFp32), MATRIX_A_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&bFp32), MATRIX_B_SIZE * sizeof(float)));

        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&aFp16), MATRIX_A_SIZE * sizeof(half)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&bFp16), MATRIX_B_SIZE * sizeof(half)));

        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cMmaExampleCommon), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cCublasGemmEx), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmmaExample1DGrid), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmmaExample2DGrid), MATRIX_C_SIZE * sizeof(float)));
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&cWmma_example), MATRIX_C_SIZE * sizeof(float)));
    }

    /* using curand to initialize */
    {
        curandGenerator_t curandGen;

        curandErrCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL));

        curandErrCheck(curandGenerateUniform(curandGen, aFp32, MATRIX_A_SIZE));
        curandErrCheck(curandGenerateUniform(curandGen, bFp32, MATRIX_B_SIZE));

        const int numThreadPerBlock = 256;
        convertFp32ToFp16<<< (MATRIX_A_SIZE + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            aFp16, aFp32, MATRIX_A_SIZE);
        convertFp32ToFp16<<< (MATRIX_B_SIZE + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            bFp16, bFp32, MATRIX_B_SIZE);

        float *c;
        cudaErrCheck(cudaMalloc(reinterpret_cast<void **>(&c), MATRIX_C_SIZE * sizeof(float)));
        curandErrCheck(curandGenerateUniform(curandGen, c, MATRIX_C_SIZE));

        cudaErrCheck(cudaMemcpy(cMmaExampleCommon, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cCublasGemmEx, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample1DGrid, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmma_example, c, MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));

        curandErrCheck(curandDestroyGenerator(curandGen));
    }

//    std::vector<float> aHost(MATRIX_A_SIZE);
//    std::vector<float> bHost(MATRIX_B_SIZE);
//    std::vector<float> cHost(MATRIX_C_SIZE);
//
//    cudaMemcpy(aHost.data(), aFp16, MATRIX_A_SIZE, cudaMemcpyDeviceToHost);
//    cudaMemcpy(bHost.data(), bFp16, MATRIX_B_SIZE, cudaMemcpyDeviceToHost);
//    cudaMemcpy(cHost.data(), cWmmaExample1DGrid, MATRIX_C_SIZE, cudaMemcpyDeviceToHost);
//
//    mmaHost(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aHost, bHost, cHost);

    /* using mmaExampleCommon computation  */
    {
        const int numThreadPerBlocks = 1024;
        const int numBlocks = (MATRIX_C_SIZE + numThreadPerBlocks - 1) / numThreadPerBlocks;
        mmaExampleCommon<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                            alpha, beta,
                                                            aFp16, bFp16, cMmaExampleCommon);
    }

    /* using cuBLAS computation */
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
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
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

    /* using wmma-example computation */
    {
        printf("---------------------------\n");
        printf("Running with wmma-example...\n");

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
        printf("gridDim.x = %d gridDim.y = %d blockDim.x = %d blockDim.y = %d\n",
               gridDim.x, gridDim.y,
               blockDim.x, blockDim.y);
        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmma_example<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aFp16, bFp16, cWmma_example);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmma_example time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    /* using wmmaExample1DGrid computation */
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
        printf("numBlocks = %d numThreadPerBlocks = %d\n", numBlocks, numThreadPerBlocks);
        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmmaExample1DGrid<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                             alpha, beta,
                                                             aFp16, bFp16, cWmmaExample1DGrid);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample1DGrid time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    /* using wmmaExample2DGrid computation */
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

        const int numCountRowOfOutputMatrixPerBlock = (WMMA_M * blockDim.x / 32);
        const int numCountColOfOutputMatrixPerBlock = (WMMA_N * blockDim.y);
        gridDim.x = (MATRIX_M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
        gridDim.y = (MATRIX_N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;

        cudaErrCheck(cudaEventRecord(start));
        wmmaExample2DGrid<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                 alpha, beta,
                                                 aFp16, bFp16, cWmmaExample2DGrid);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaErrCheck(cudaEventRecord(stop));
        cudaErrCheck(cudaEventSynchronize(stop));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, start, stop));
        printf("wmmaExample2DGrid time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(start));
        cudaErrCheck(cudaEventDestroy(stop));
    }

    if (!checkDevData(MATRIX_C_SIZE, cCublasGemmEx, cWmma_example)) {
        printf("Error! cublasGemmEx, wmma_example Check no passes!\n");
    } else {
        printf("cublasGemmEx, wmma_example Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cMmaExampleCommon, cWmmaExample1DGrid)) {
        printf("Error! mmaExampleCommon, wmmaExample1DGrid Check no passes!\n");
    } else {
        printf("mmaExampleCommon, wmmaExample1DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cMmaExampleCommon, cWmmaExample2DGrid)) {
        printf("Error! mmaExampleCommon, wmmaExample2DGrid Check no passes!\n");
    } else {
        printf("mmaExampleCommon, wmmaExample2DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cWmmaExample1DGrid, cWmmaExample2DGrid)) {
        printf("Error! wmmaExample1DGrid, wmmaExample2DGrid Check no passes!\n");
    } else {
        printf("wmmaExample1DGrid, wmmaExample2DGrid Check passes!\n");
    }

    cudaErrCheck(cudaFree(aFp32));
    cudaErrCheck(cudaFree(bFp32));
    cudaErrCheck(cudaFree(aFp16));
    cudaErrCheck(cudaFree(bFp16));
    cudaErrCheck(cudaFree(cCublasGemmEx));
    cudaErrCheck(cudaFree(cWmmaExample1DGrid));
    cudaErrCheck(cudaFree(cWmma_example));

    return 0;
}