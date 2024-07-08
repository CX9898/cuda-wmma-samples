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
    float *a_fp32;
    float *b_fp32;

    half *a_fp16;
    half *b_fp16;

    float *c;
    float *c_cublas;

    const float alpha = 2.0f;
    const float beta = 2.0f;

    cudaErrCheck(cudaMalloc((void **) &a_fp32, MATRIX_M * MATRIX_K * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &b_fp32, MATRIX_K * MATRIX_N * sizeof(float)));

    cudaErrCheck(cudaMalloc((void **) &a_fp16, MATRIX_M * MATRIX_K * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **) &b_fp16, MATRIX_K * MATRIX_N * sizeof(half)));

    cudaErrCheck(cudaMalloc((void **) &c, MATRIX_M * MATRIX_N * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &c_cublas, MATRIX_M * MATRIX_N * sizeof(float)));

    /* using curand to initialize */
    {
        curandGenerator_t curandGen;

        curandErrCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL));

        curandErrCheck(curandGenerateUniform(curandGen, a_fp32, MATRIX_M * MATRIX_K));
        curandErrCheck(curandGenerateUniform(curandGen, b_fp32, MATRIX_K * MATRIX_N));
        curandErrCheck(curandGenerateUniform(curandGen, c, MATRIX_M * MATRIX_N));

        curandErrCheck(curandDestroyGenerator(curandGen));

        const int numThreadPerBlock = 256;
        const int numBlocks = (MATRIX_M * MATRIX_K + 255) / 256;
        convertFp32ToFp16<<< numBlocks, numThreadPerBlock>>>(a_fp16, a_fp32, MATRIX_M * MATRIX_K);
        convertFp32ToFp16<<< numBlocks, numThreadPerBlock>>>(b_fp16, b_fp32, MATRIX_K * MATRIX_N);
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
                                    a_fp16, CUDA_R_16F, MATRIX_M,
                                    b_fp16, CUDA_R_16F, MATRIX_K,
                                    &beta,
                                    c_cublas, CUDA_R_32F, MATRIX_M,
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

        cudaEvent_t startWMMAEx;
        cudaEvent_t stopWMMAEx;

        cudaErrCheck(cudaEventCreate(&startWMMAEx));
        cudaErrCheck(cudaEventCreate(&stopWMMAEx));

        dim3 gridDim;
        dim3 blockDim;

        cudaErrCheck(cudaEventRecord(startWMMAEx));
        wmmaExample<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, a_fp16, b_fp16, c);
        cudaErrCheck(cudaEventRecord(stopWMMAEx));
        cudaErrCheck(cudaEventSynchronize(stopWMMAEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWMMAEx, stopWMMAEx));
        printf("wmmaExample time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWMMAEx));
        cudaErrCheck(cudaEventDestroy(stopWMMAEx));
    }

    cudaErrCheck(cudaFree(a_fp32));
    cudaErrCheck(cudaFree(b_fp32));
    cudaErrCheck(cudaFree(a_fp16));
    cudaErrCheck(cudaFree(b_fp16));
    cudaErrCheck(cudaFree(c));
    cudaErrCheck(cudaFree(c_cublas));

    return 0;
}