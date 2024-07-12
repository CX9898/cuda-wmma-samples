#include "kernel.cuh"

int main() {
    float *aFp32;
    float *bFp32;

    half *aFp16;
    half *bFp16;

    float *cCublas;
    float *cWmmaEx;
    float *cWmmaEx2;

    const float alpha = 2.0f;
    const float beta = 2.0f;

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

        curandErrCheck(curandDestroyGenerator(curandGen));

        const int numThreadPerBlock = 256;
        convertFp32ToFp16<<< (numMatrixADates + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            aFp16, aFp32, numMatrixADates);
        convertFp32ToFp16<<< (numMatrixBDates + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            bFp16, bFp32, numMatrixBDates);
    }

    /* using cuBLAS computation */
    {
        printf("Running with cuBLAS...\n");

        cudaEvent_t startcublas;
        cudaEvent_t stopcublas;

        cudaErrCheck(cudaEventCreate(&startcublas));
        cudaErrCheck(cudaEventCreate(&stopcublas));

        cublasHandle_t cublasHandle;
        cublasErrCheck(cublasCreate(&cublasHandle));

        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cudaErrCheck(cudaEventRecord(startcublas));
        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    aFp16, CUDA_R_16F, MATRIX_M,
                                    bFp16, CUDA_R_16F, MATRIX_K,
                                    &beta,
                                    cCublas, CUDA_R_32F, MATRIX_M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        cudaErrCheck(cudaEventRecord(stopcublas));
        cudaErrCheck(cudaEventSynchronize(stopcublas));

        float cublasTime;
        cudaErrCheck(cudaEventElapsedTime(&cublasTime, startcublas, stopcublas));
        printf("cublasGemmEx time : %fms\n", cublasTime);

        cublasErrCheck(cublasDestroy(cublasHandle));

        cudaErrCheck(cudaEventDestroy(startcublas));
        cudaErrCheck(cudaEventDestroy(stopcublas));
    }

    /* using wmma-example computation */
    {
        printf("Running with wmma-example...\n");

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmma_example<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aFp16, bFp16, cWmmaEx2);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    /* using wmmaExample computation */
    {
        printf("Running with wmmaExample...\n");

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        dim3 gridDim;
        dim3 blockDim;

        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmmaExample<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aFp16, bFp16, cWmmaEx);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

//    if (!checkData(numMatrixCDates, cCublas, cWmmaEx)) {
//        fprintf(stderr,"The results of cublas and wmmaEx are inconsistent\n");
//    }
    if (!checkData(numMatrixCDates, cCublas, cWmmaEx2)) {
        fprintf(stderr,"The results of cublas and wmmaEx2 are inconsistent\n");
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