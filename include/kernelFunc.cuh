#pragma once

#define WARP_SIZE 32

__global__ void convertFp32ToFp16(half *out, float *in, int n);

__global__ void wmma_example(int M, int N, int K,
                             float alpha, float beta,
                             half *a, half *b, float *c);

__global__ void mmaExampleCommon(const int M, const int N, const int K,
                                 const float alpha, const float beta,
                                 const half *mtrA, const half *mtrB, float *mtrC);

// Tile using a 1D grid
__global__ void wmmaExample1DGrid(const int M, const int N, const int K,
                                  const float alpha, const float beta,
                                  const half *mtrA, const half *mtrB, float *mtrC);

// Tile using a 2D grid
__global__ void wmmaExample2DGrid(const int M, const int N, const int K,
                                 const float alpha, const float beta,
                                 const half *mtrA, const half *mtrB, float *mtrC);
