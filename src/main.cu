#include <cstdio>
#include <curand.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

// Must be multiples of 16 for wmma code to work
#define MATRIX_M 16384
#define MATRIX_N 16384
#define MATRIX_K 16384

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void wmmaExample(const int M, const int N, const int K,
                            const float alpha, const float beta,
                            const half *mtrA, const half *mtrB, float *mtrC) {
    const int warpID = (int) (blockDim.x * blockIdx.x + threadIdx.x) / warpSize;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

}

int main() {
    float *aFp32;
    float *bFp32;

    half *aFp16;
    half *bFp16;

    float *cWmmaEx;
    float *cCublas;

    const float alpha = 2.0f;
    const float beta = 2.0f;

    const int numMatrixADates = MATRIX_M * MATRIX_K;
    const int numMatrixBDates = MATRIX_K * MATRIX_N;
    const int numMatrixCDates = MATRIX_M * MATRIX_N;

    cudaErrCheck(cudaMalloc((void **) &aFp32, numMatrixADates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &bFp32, numMatrixBDates * sizeof(float)));

    cudaErrCheck(cudaMalloc((void **) &aFp16, numMatrixADates * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **) &bFp16, numMatrixBDates * sizeof(half)));

    cudaErrCheck(cudaMalloc((void **) &cWmmaEx, numMatrixCDates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &cCublas, numMatrixCDates * sizeof(float)));

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

    /* error checking */
    {
        printf("\nChecking results...\n");

        float *cCublasHost = (float *) malloc(numMatrixCDates * sizeof(float));
        float *cWmmaExHost = (float *) malloc(numMatrixCDates * sizeof(float));

        cudaErrCheck(cudaMemcpy(cWmmaExHost, cWmmaEx, numMatrixCDates * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(cCublasHost, cCublas, numMatrixCDates * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int idx = 0; idx < numMatrixCDates; ++idx) {
            float cublasRes = cCublasHost[idx];
            float wmmaExRes = cWmmaEx[idx];
            float diffDats = fabs(cublasRes - wmmaExRes);

            float relativeErr = diffDats / cublasRes;
            float eps = 1e-4;
            if (relativeErr >= eps) {
                ++errors;
                if (errors < 10) {
                    printf("error : cublasRes = %f, wmmaExRes = %f\n", cublasRes, wmmaExRes);
                }
            }
        }

        free(cCublasHost);
        free(cWmmaExHost);
    }

    cudaErrCheck(cudaFree(aFp32));
    cudaErrCheck(cudaFree(bFp32));
    cudaErrCheck(cudaFree(aFp16));
    cudaErrCheck(cudaFree(bFp16));
    cudaErrCheck(cudaFree(cWmmaEx));
    cudaErrCheck(cudaFree(cCublas));

    return 0;
}