#pragma once

#include <cstdio>
#include <mma.h>
#include <curand.h>
#include <cublas_v2.h>

#include "matrixSetting.hpp"

#define WARP_SIZE 32

__global__ void convertFp32ToFp16(half *out, float *in, int n);

__global__ void wmmaExample(const int M, const int N, const int K,
                            const float alpha, const float beta,
                            const half *mtrA, const half *mtrB, float *mtrC);

__global__ void wmma_example(int M, int N, int K, float alpha, float beta, half *a, half *b, float *c);

bool checkDevData(const int num, const float *dataDev1, const float *dataDev2);
