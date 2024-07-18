#include "kernel.cuh"
#include "host.hpp"
#include "cudaErrorCheck.hpp"

int main() {
    float *aFp32;
    float *bFp32;

    half *aFp16;
    half *bFp16;

    float *cCublas;
    float *cWmmaEx;
    float *cWmmaEx2;

    const float alpha = 1.0f;
    const float beta = 1.0f;

    const int numMatrixADates = MATRIX_M * MATRIX_K;
    const int numMatrixBDates = MATRIX_K * MATRIX_N;
    const int numMatrixCDates = MATRIX_M * MATRIX_N;

    cudaErrCheck(cudaMalloc((void **) &aFp32, numMatrixADates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &bFp32, numMatrixBDates * sizeof(float)));

    cudaErrCheck(cudaMalloc((void **) &aFp16, numMatrixADates * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **) &bFp16, numMatrixBDates * sizeof(half)));

    cudaErrCheck(cudaMalloc((void **) &cCublas, numMatrixCDates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &cWmmaEx, numMatrixCDates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &cWmmaEx2, numMatrixCDates * sizeof(float)));

    /* using curand to initialize */
    {
        curandGenerator_t curandGen;

        curandErrCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL));

        curandErrCheck(curandGenerateUniform(curandGen, aFp32, numMatrixADates));
        curandErrCheck(curandGenerateUniform(curandGen, bFp32, numMatrixBDates));

        float *c;
        cudaErrCheck(cudaMalloc((void **) &c, numMatrixCDates * sizeof(float)));
        curandErrCheck(curandGenerateUniform(curandGen, c, numMatrixCDates));

        cudaErrCheck(cudaMemcpy(cCublas, c, numMatrixCDates, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaEx, c, numMatrixCDates, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaEx2, c, numMatrixCDates, cudaMemcpyDeviceToDevice));

        curandErrCheck(curandDestroyGenerator(curandGen));

        const int numThreadPerBlock = 256;
        convertFp32ToFp16<<< (numMatrixADates + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            aFp16, aFp32, numMatrixADates);
        convertFp32ToFp16<<< (numMatrixBDates + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            bFp16, bFp32, numMatrixBDates);
    }

    std::vector<float> aHost(numMatrixADates);
    std::vector<float> bHost(numMatrixBDates);
    std::vector<float> cHost(numMatrixCDates);

    cudaMemcpy(aHost.data(), aFp32, numMatrixADates, cudaMemcpyDeviceToHost);
    cudaMemcpy(bHost.data(), bFp32, numMatrixBDates, cudaMemcpyDeviceToHost);
    cudaMemcpy(cHost.data(), cWmmaEx, numMatrixCDates, cudaMemcpyDeviceToHost);

    mmaHost(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aHost, bHost, cHost);

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
                                    cCublas, CUDA_R_32F, MATRIX_M,
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
        wmma_example<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aFp16, bFp16, cWmmaEx2);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmma_example time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    /* using wmmaExample computation */
    {
        printf("---------------------------\n");
        printf("Running with wmmaExample...\n");

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        const int wmmaCalculatesOneResultTileSize = WMMA_M * WMMA_N;
        int numThreadPerBlocks = WARP_SIZE * 1;
        int numBlocks = (numMatrixCDates / wmmaCalculatesOneResultTileSize * WARP_SIZE + numThreadPerBlocks - 1)
            / numThreadPerBlocks;
        printf("numBlocks = %d numThreadPerBlocks = %d\n", numBlocks, numThreadPerBlocks);
        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmmaExample<<<numBlocks, numThreadPerBlocks>>>(MATRIX_M, MATRIX_N, MATRIX_K,
                                                       alpha, beta,
                                                       aFp16, bFp16, cWmmaEx);
        printf("%s\n", cudaGetErrorString(cudaGetLastError()));
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    if (!checkDevData(numMatrixCDates, cCublas, cWmmaEx)) {
        printf("Error! cublas, wmmaExample Check no passes!\n");
    } else {
        printf("cublas, wmmaExample Check passes!\n");
    }

    if (!checkDevData(numMatrixCDates, cCublas, cWmmaEx2)) {
        printf("Error! cublas, wmma_example Check no passes!\n");
    } else {
        printf("cublas, wmma_example Check passes!\n");
    }

    if (!checkData(numMatrixCDates, cHost, cWmmaEx)) {
        printf("Error! mmaHost, wmmaExample Check no passes!\n");
    } else {
        printf("mmaHost, wmmaExample Check passes!\n");
    }

    cudaErrCheck(cudaFree(aFp32));
    cudaErrCheck(cudaFree(bFp32));
    cudaErrCheck(cudaFree(aFp16));
    cudaErrCheck(cudaFree(bFp16));
    cudaErrCheck(cudaFree(cCublas));
    cudaErrCheck(cudaFree(cWmmaEx));
    cudaErrCheck(cudaFree(cWmmaEx2));

    return 0;
}