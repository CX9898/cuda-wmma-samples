#include <cstdio>

#include <curand.h>
#include <cublas_v2.h>

#include "devVector.cuh"
#include "kernelFunc.cuh"
#include "hostFunc.hpp"
#include "cudaErrorCheck.cuh"
#include "matrixSetting.hpp"
#include "cudaTimeCalculator.cuh"

int main() {
    dev::vector<float> aFp32(MATRIX_A_SIZE);
    dev::vector<float> bFp32(MATRIX_B_SIZE);

    dev::vector<half> aFp16(MATRIX_A_SIZE);
    dev::vector<half> bFp16(MATRIX_B_SIZE);

    dev::vector<float> cMmaExampleCommon(MATRIX_C_SIZE);
    dev::vector<float> cCublasGemmEx(MATRIX_C_SIZE);
    dev::vector<float> cWmmaExample1DGrid(MATRIX_C_SIZE);
    dev::vector<float> cWmmaExample2DGrid(MATRIX_C_SIZE);
    dev::vector<float> cWmmaExample2DGrid2(MATRIX_C_SIZE);
    dev::vector<float> cWmmaExample2DGrid3(MATRIX_C_SIZE);

    const float alpha = 2.0f;
    const float beta = 2.0f;

    // using cuRAND to initialize
    {
        curandGenerator_t curandGen;

        curandErrCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL));

        curandErrCheck(curandGenerateUniform(curandGen, aFp32.data(), MATRIX_A_SIZE));
        curandErrCheck(curandGenerateUniform(curandGen, bFp32.data(), MATRIX_B_SIZE));

        const int numThreadPerBlock = 256;
        convertFp32ToFp16<<< (MATRIX_A_SIZE + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            MATRIX_A_SIZE, aFp32.data(), aFp16.data());
        convertFp32ToFp16<<< (MATRIX_B_SIZE + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            MATRIX_B_SIZE, bFp32.data(), bFp16.data());

        dev::vector<float> c(MATRIX_C_SIZE);
        curandErrCheck(curandGenerateUniform(curandGen, c.data(), c.size()));

        // TODO : implement copy function
        cudaErrCheck(cudaMemcpy(cMmaExampleCommon.data(), c.data(), MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cCublasGemmEx.data(), c.data(), MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample1DGrid.data(), c.data(), MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid.data(), c.data(), MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid2.data(), c.data(), MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaExample2DGrid3.data(), c.data(), MATRIX_C_SIZE, cudaMemcpyDeviceToDevice));

        curandErrCheck(curandDestroyGenerator(curandGen));
    }

    // using mmaExampleCommon computation
    {
        const int numThreadPerBlocks = 1024;
        const int numBlocks = (MATRIX_C_SIZE + numThreadPerBlocks - 1) / numThreadPerBlocks;
        mmaExampleCommon<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                            alpha, beta,
                                                            aFp16.data(), bFp16.data(), cMmaExampleCommon.data());
    }

    // using cuBLAS computation
    {
        printf("---------------------------\n");
        printf("Running with cuBLAS...\n");

        cublasHandle_t cublasHandle;
        cublasErrCheck(cublasCreate(&cublasHandle));

        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cudaTimeCalculator timeCalculator;

        timeCalculator.startClock();
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    aFp16.data(), CUDA_R_16F, MATRIX_M,
                                    bFp16.data(), CUDA_R_16F, MATRIX_K,
                                    &beta,
                                    cCublasGemmEx.data(), CUDA_R_32F, MATRIX_M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        timeCalculator.endClock();

        printf("cublasGemmEx time : %fms\n", timeCalculator.getTime());

        cublasErrCheck(cublasDestroy(cublasHandle));
    }

    // using wmmaExample1DGrid computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample1DGrid...\n");

        const int wmmaCalculatesOneResultTileSize = WMMA_M * WMMA_N;
        int numThreadPerBlocks = WARP_SIZE * 1;
        int numBlocks = (MATRIX_C_SIZE / wmmaCalculatesOneResultTileSize * WARP_SIZE + numThreadPerBlocks - 1)
            / numThreadPerBlocks;

        cudaTimeCalculator timeCalculator;

        timeCalculator.startClock();
        wmmaExample1DGrid<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                             alpha, beta,
                                                             aFp16.data(), bFp16.data(), cWmmaExample1DGrid.data());
        timeCalculator.endClock();

        printf("wmmaExample1DGrid time : %fms\n", timeCalculator.getTime());
    }

    // using wmmaExample2DGrid computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample2DGrid...\n");

        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = WARP_SIZE;
        blockDim.y = WARP_SIZE;

        const int numCountRowOfOutputMatrixPerBlock = (int) (WMMA_M * blockDim.x / 32);
        const int numCountColOfOutputMatrixPerBlock = (int) (WMMA_N * blockDim.y);
        gridDim.x = (MATRIX_M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
        gridDim.y = (MATRIX_N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;

        cudaTimeCalculator timeCalculator;

        timeCalculator.startClock();
        wmmaExample2DGrid<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                 alpha, beta,
                                                 aFp16.data(), bFp16.data(), cWmmaExample2DGrid.data());
        timeCalculator.endClock();

        printf("wmmaExample2DGrid time : %fms\n", timeCalculator.getTime());
    }

    // using wmmaExample2DGrid2 computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample2DGrid2...\n");

        dim3 gridDim;
        dim3 blockDim;

        blockDim.x = WARP_SIZE;
        blockDim.y = WARP_SIZE;

        const int numCountRowOfOutputMatrixPerBlock = (int) (WMMA_M * blockDim.x / 32);
        const int numCountColOfOutputMatrixPerBlock = (int) (WMMA_N * blockDim.y);
        gridDim.x = (MATRIX_M + numCountRowOfOutputMatrixPerBlock - 1) / numCountRowOfOutputMatrixPerBlock;
        gridDim.y = (MATRIX_N + numCountColOfOutputMatrixPerBlock - 1) / numCountColOfOutputMatrixPerBlock;

        cudaTimeCalculator timeCalculator;

        timeCalculator.startClock();
        wmmaExample2DGrid2<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                  alpha, beta,
                                                  aFp16.data(), bFp16.data(), cWmmaExample2DGrid2.data());
        timeCalculator.endClock();

        printf("wmmaExample2DGrid2 time : %fms\n", timeCalculator.getTime());
    }

    // using wmmaExample2DGrid3 computation
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample2DGrid3...\n");

        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multiple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        cudaTimeCalculator timeCalculator;

        timeCalculator.startClock();
        wmmaExample2DGrid3<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                  alpha, beta,
                                                  aFp16.data(), bFp16.data(), cWmmaExample2DGrid3.data());
        timeCalculator.endClock();

        printf("wmmaExample2DGrid3 time : %fms\n", timeCalculator.getTime());
    }

    if (!checkDevData(MATRIX_C_SIZE, cCublasGemmEx.data(), cWmmaExample2DGrid3.data())) {
        printf("Error! Function cublasGemmEx, wmmaExample2DGrid3 Check no passes!\n");
    } else {
        printf("Function cublasGemmEx, wmmaExample2DGrid3 Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cMmaExampleCommon.data(), cWmmaExample1DGrid.data())) {
        printf("Error! Function mmaExampleCommon, wmmaExample1DGrid Check no passes!\n");
    } else {
        printf("Function mmaExampleCommon, wmmaExample1DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cMmaExampleCommon.data(), cWmmaExample2DGrid.data())) {
        printf("Error! Function mmaExampleCommon, wmmaExample2DGrid Check no passes!\n");
    } else {
        printf("Function mmaExampleCommon, wmmaExample2DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cWmmaExample1DGrid.data(), cWmmaExample2DGrid.data())) {
        printf("Error! Function wmmaExample1DGrid, wmmaExample2DGrid Check no passes!\n");
    } else {
        printf("Function wmmaExample1DGrid, wmmaExample2DGrid Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cCublasGemmEx.data(), cWmmaExample2DGrid2.data())) {
        printf("Error! Function cublasGemmEx, wmmaExample2DGrid2 Check no passes!\n");
    } else {
        printf("Function cublasGemmEx, wmmaExample2DGrid2 Check passes!\n");
    }

    if (!checkDevData(MATRIX_C_SIZE, cWmmaExample2DGrid.data(), cWmmaExample2DGrid2.data())) {
        printf("Error! Function wmmaExample2DGrid, wmmaExample2DGrid2 Check no passes!\n");
    } else {
        printf("Function wmmaExample2DGrid, wmmaExample2DGrid2 Check passes!\n");
    }

//    cudaErrCheck(cudaFree(aFp32));
//    cudaErrCheck(cudaFree(bFp32));
//    cudaErrCheck(cudaFree(aFp16));
//    cudaErrCheck(cudaFree(bFp16));
//    cudaErrCheck(cudaFree(cCublasGemmEx));
//    cudaErrCheck(cudaFree(cWmmaExample2DGrid3));
//    cudaErrCheck(cudaFree(cWmmaExample1DGrid));
//    cudaErrCheck(cudaFree(cWmmaExample2DGrid));
//    cudaErrCheck(cudaFree(cWmmaExample2DGrid2));

    return 0;
}