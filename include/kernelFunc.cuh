#pragma once

#define WARP_SIZE 32

__global__ void convertFp32ToFp16(const int n, const float *in, half *out);

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(int M, int N, int K,
                             float alpha, float beta,
                             half *a, half *b, float *c);

/**
 * This is the most straightforward way to compute matrix multiplication and addition.
 * C = alpha * A * B + beta * C
 * All three matrices A,B and C are stored in row major order.
 * This is just for comparison purposes.
 **/
__global__ void mmaExampleCommon(const int M, const int N, const int K,
                                 const float alpha, const float beta,
                                 const half *mtrA, const half *mtrB, float *mtrC);

/**
 * This is an example of matrix multiplication and addition using the WMMA API.
 * C = alpha * A * B + beta * C
 * All three matrices A,B and C are stored in row major order.
 * M, N and K are multiples of 16.
 * Tile using 1D grid.
 **/
__global__ void wmmaExample1DGrid(const int M, const int N, const int K,
                                  const float alpha, const float beta,
                                  const half *mtrA, const half *mtrB, float *mtrC);

/**
 * This is an example of matrix multiplication and addition using the WMMA API.
 * C = alpha * A * B + beta * C
 * All three matrices A,B and C are stored in row major order.
 * M, N and K are multiples of 16.
 * Tile using 1D grid.
 **/
__global__ void wmmaExample2DGrid(const int M, const int N, const int K,
                                  const float alpha, const float beta,
                                  const half *mtrA, const half *mtrB, float *mtrC);
