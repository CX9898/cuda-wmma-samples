#include <mma.h>

#include "matrixSetting.hpp"
#include "kernelFunc.cuh"

using namespace nvcuda;

__global__ void convertFp32ToFp16(const int n, const float *in, half *out) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
__global__ void wmma_example(int M, int N, int K,
                             float alpha, float beta,
                             half *a, half *b, float *c) {
    // Leading dimensions. Packed with no transpositions.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Tile using a 2D grid
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare the fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over k
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;

        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load the inputs
            wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
            wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

            // Perform the matrix multiplication
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);

        }
    }

    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

#pragma unroll
        for (int i = 0; i < c_frag.num_elements; i++) {
            c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store the output
        wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
    }
}

__global__ void mmaExampleCommon(const int M, const int N, const int K,
                                 const float alpha, const float beta,
                                 const half *mtrA, const half *mtrB, float *mtrC) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
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
    wmma::load_matrix_sync(cFrag, cOffsetPtr, ldc, wmma::mem_row_major);

#pragma unroll
    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
        cFrag.x[idx] = alpha * accFrag.x[idx] + beta * cFrag.x[idx];
    }

    wmma::store_matrix_sync(cOffsetPtr, cFrag, ldc, wmma::mem_row_major);
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

    for (int kIter = 0; kIter < K; ++kIter) {
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

    for (int idx = 0; idx < cFrag.num_elements; ++idx) {
        cFrag.x[idx] = alpha * accFrag.x[idx] + beta * cFrag.x[idx];
    }

    wmma::store_matrix_sync(cOffsetPtr, cFrag, ldc, wmma::mem_row_major);
}
