#pragma once

#include <cstdio>
#include <mma.h>
#include <curand.h>
#include <cublas_v2.h>
#include "matrixSetting.hpp"

#define WARP_SIZE 32

// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
    }
}

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
inline void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
    }
}

#define curandErrCheck(stat) { curandErrCheck_((stat), __FILE__, __LINE__); }
inline void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
    if (stat != CURAND_STATUS_SUCCESS) {
        fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
    }
}

__global__ void convertFp32ToFp16(half *out, float *in, int n);

__global__ void wmmaExample(const int M, const int N, const int K,
                            const float alpha, const float beta,
                            const half *mtrA, const half *mtrB, float *mtrC);

__global__ void wmma_example(int M, int N, int K, float alpha, float beta, half *a, half *b, float *c);

bool checkData(const int num, const float *dataDev1, const float *dataDev2);
