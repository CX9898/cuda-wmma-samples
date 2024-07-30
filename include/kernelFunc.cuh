#pragma once

#define WARP_SIZE 32

__global__ void convertFp32ToFp16(const int n, const float *in, half *out);

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

/**
 * This is an example of matrix multiplication and addition using the WMMA API.
 * C = alpha * A * B + beta * C
 * The A and C matrices are row-major order stores, and the B matrix is column-major order stores.
 * M, N and K are multiples of 16.
 * Tile using 2D grid.
 **/
__global__ void wmmaExample2DGrid2(const int M, const int N, const int K,
                                  const float alpha, const float beta,
                                  const half *mtrA, const half *mtrB, float *mtrC);

/**
 * This is an example of matrix multiplication and addition using the WMMA API.
 * C = alpha * A * B + beta * C
 * All three matrices A,B and C are stored in col major order.
 * M, N and K are multiples of 16.
 * Tile using 2D grid.
 **/
__global__ void wmmaExample2DGrid3(const int M, const int N, const int K,
                                   const float alpha, const float beta,
                                   const half *mtrA, const half *mtrB, float *mtrC) ;
