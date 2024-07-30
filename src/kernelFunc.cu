#include <mma.h>

#include "matrixSetting.hpp"
#include "kernelFunc.cuh"

using namespace nvcuda;

__global__ void convertFp32ToFp16(const int n, const float *in, half *out) {
    int idx = (int) (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void mmaExampleCommon(const int M, const int N, const int K,
                                 const float alpha, const float beta,
                                 const half *mtrA, const half *mtrB, float *mtrC) {
    const int idx = (int) (blockDim.x * blockIdx.x + threadIdx.x);
    if (idx >= M * N) {
        return;
    }

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    const int cRow = idx / ldc;
    const int cCol = idx % ldc;

    const int aRowOffset = cRow * lda;
    const int bColOffset = cCol;

    float counter = 0.0;

    for (int kIter = 0; kIter < K; ++kIter) {
        counter += float(mtrA[aRowOffset + kIter]) * float(mtrB[bColOffset + kIter * ldb]);
    }

    const int cIdx = cRow * ldc + cCol;
    mtrC[cIdx] = alpha * counter + beta * mtrC[cIdx];
}

__global__ void wmmaExample1DGrid(const int M, const int N, const int K,
                                  const float alpha, const float beta,
                                  const half *mtrA, const half *mtrB, float *mtrC) {
    int warpId = (int) (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    // Due to the use of 1D grid,
    // warp first iterates over the columns of the resulting matrix and then over the rows
    int cRow = (warpId * WMMA_M) % M;
    int cCol = 0;
    if (warpId > 0) {
        cCol = warpId * WMMA_M / M * WMMA_N;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag; // Fragment accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    // Leading dimensions. Packed with no transpositions.
    int lda = K;
    int ldb = N;
    int ldc = N;

    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        int aRow = cRow;
        int aCol = kIter;

        int bRow = kIter;
        int bCol = cCol;

        const auto aTileOffsetPrt = mtrA + aRow * lda + aCol;
        const auto bTileOffsetPrt = mtrB + bRow * ldb + bCol;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < N && bCol < K) {
            wmma::load_matrix_sync(aFrag, aTileOffsetPrt, lda);
            wmma::load_matrix_sync(bFrag, bTileOffsetPrt, ldb);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    const auto cTileOffsetPrt = mtrC + cRow * ldc + cCol;

    // Bounds checking
    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(cFrag, cTileOffsetPrt, ldc, wmma::mem_row_major);

#pragma unroll
        for (int idx = 0; idx < cFrag.num_elements; ++idx) {
            cFrag.x[idx] = alpha * accFrag.x[idx] + beta * cFrag.x[idx];
        }

        wmma::store_matrix_sync(cTileOffsetPrt, cFrag, ldc, wmma::mem_row_major);
    }

}

__global__ void wmmaExample2DGrid(const int M, const int N, const int K,
                                  const float alpha, const float beta,
                                  const half *mtrA, const half *mtrB, float *mtrC) {
    const int warpM = (int) (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (int) (blockDim.y * blockIdx.y + threadIdx.y);

    const int cRow = warpM * WMMA_M;
    const int cCol = warpN * WMMA_N;

    if (cRow >= M || cCol >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    const int lda = K;
    const int ldb = N;
    const int ldc = N;

    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRow = cRow;
        const int aCol = kIter;

        const int bRow = kIter;
        const int bCol = cCol;

        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            const auto aOffsetPtr = mtrA + aRow * lda + aCol;
            const auto bOffsetPtr = mtrB + bRow * ldb + bCol;

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    const auto cOffsetPtr = mtrC + cRow * ldc + cCol;
    wmma::load_matrix_sync(cFrag, cOffsetPtr, ldc, wmma::mem_col_major);

#pragma unroll
    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
        cFrag.x[idx] = alpha * accFrag.x[idx] + beta * cFrag.x[idx];
    }

    wmma::store_matrix_sync(cOffsetPtr, cFrag, ldc, wmma::mem_col_major);
}

__global__ void wmmaExample2DGrid2(const int M, const int N, const int K,
                                   const float alpha, const float beta,
                                   const half *mtrA, const half *mtrB, float *mtrC) {
    const int warpM = (int) (blockDim.x * blockIdx.x + threadIdx.x) / WARP_SIZE;
    const int warpN = (int) (blockDim.y * blockIdx.y + threadIdx.y);

    const int cRow = warpM * WMMA_M;
    const int cCol = warpN * WMMA_N;

    if (cRow >= M || cCol >= N) {
        return;
    }

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    const int lda = K;
    const int ldb = K;
    const int ldc = N;

    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRow = cRow;
        const int aCol = kIter;

        const int bRow = cCol;
        const int bCol = kIter;

        if (aRow < M && aRow < K && bRow < N && bCol < K) {
            const auto aOffsetPtr = mtrA + aRow * lda + aCol;
            const auto bOffsetPtr = mtrB + bRow * ldb + bCol;

            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    const auto cOffsetPtr = mtrC + cRow * ldc + cCol;
    wmma::load_matrix_sync(cFrag, cOffsetPtr, ldc, wmma::mem_row_major);

#pragma unroll
    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
        cFrag.x[idx] = alpha * accFrag.x[idx] + beta * cFrag.x[idx];
    }

    // Store the output
    wmma::store_matrix_sync(cOffsetPtr, cFrag, ldc, wmma::mem_row_major);
}

__global__ void wmmaExample2DGrid3(const int M, const int N, const int K,
                                   const float alpha, const float beta,
                                   const half *mtrA, const half *mtrB, float *mtrC) {
    // Tile using mtrA 2D grid
    const int warpM = (int) (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    const int warpN = (int) (blockIdx.y * blockDim.y + threadIdx.y);

    // Load in the current value of mtrC, scale it by beta, and add this our result scaled by alpha
    const int cRow = warpM * WMMA_M;
    const int cCol = warpN * WMMA_N;

    if (cRow >= M || cCol >= N) {
        return;
    }

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    // Leading dimensions. Packed with no transpositions.
    const int lda = M;
    const int ldb = K;
    const int ldc = M;

    // Loop over k
    for (int kIter = 0; kIter < K; kIter += WMMA_K) {
        const int aRow = cRow;
        const int aCol = kIter;

        const int bRow = kIter;
        const int bCol = cCol;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            const auto aOffsetPtr = mtrA + aRow + aCol * lda;
            const auto bOffsetPtr = mtrB + bRow + bCol * ldb;

            // Load the inputs
            wmma::load_matrix_sync(aFrag, aOffsetPtr, lda);
            wmma::load_matrix_sync(bFrag, bOffsetPtr, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    const auto cOffsetPtr = mtrC + cRow + cCol * ldc;
    wmma::load_matrix_sync(cFrag, cOffsetPtr, ldc, wmma::mem_col_major);

#pragma unroll
    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
        cFrag.x[idx] = alpha * accFrag.x[idx] + beta * cFrag.x[idx];
    }

    // Store the output
    wmma::store_matrix_sync(cOffsetPtr, cFrag, ldc, wmma::mem_col_major);
}