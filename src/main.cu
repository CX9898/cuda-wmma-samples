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

//void cudaErrCheck(cudaError_t stat) {
//    if (stat != cudaSuccess) {
//        printf("CUDA Error: %s %s %d\n", cudaGetErrorString(stat), __FILEW__, __LINE__);
//    }
//}
//
//void cublasErrCheck(cublasStatus_t stat) {
//    if (stat != CUBLAS_STATUS_SUCCESS) {
//        printf( "cuBLAS Error: %d %s %d\n", stat, __FILEW__, __LINE__);
//    }
//}

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

__global__ void wmmaExample(const int M, const int N, const int K,
                            const float alpha, const float beta,
                            const half *mtrA, const float *mtrB, float *mtrC) {
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


    /* using cuBLAS */
    {
        printf("Running with cuBLAS...\n");

        cublasHandle_t cublasHandle;
        cublasErrCheck(cublasCreate(&cublasHandle));

        // Use tensor cores
        cublasErrCheck(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

        cublasErrCheck(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                                    MATRIX_M, MATRIX_N, MATRIX_K,
                                    &alpha,
                                    a_fp16, CUDA_R_16F, MATRIX_M,
                                    b_fp16, CUDA_R_16F, MATRIX_K,
                                    &beta,
                                    c_cublas, CUDA_R_32F, MATRIX_M,
                                    CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    cudaErrCheck(cudaFree(a_fp16));

    return 0;
}