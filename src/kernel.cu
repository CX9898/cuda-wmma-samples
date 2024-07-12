#include "kernel.cuh"

using namespace nvcuda;

__global__ void convertFp32ToFp16(half *out, float *in, int n) {
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
__global__ void wmma_example(int M, int N, int K, float alpha, float beta, half *a, half *b, float *c) {
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

// Tile using a 1D grid
__global__ void wmmaExample(const int M, const int N, const int K,
                            const float alpha, const float beta,
                            const half *mtrA, const half *mtrB, float *mtrC) {
    // Leading dimensions. Packed with no transpositions.
    int lda = K;
    int ldb = N;
    int ldc = N;

    int warpId = (int) (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag; // Fragment accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    for (int kOffset = 0; kOffset < K; kOffset += WMMA_K) {
        int aRowIdx = kOffset;
        int aColIdx = warpId * WMMA_M;

        int bRowIdx = warpId * WMMA_N;
        int bColIdx = kOffset;

//        if ((blockIdx.x * blockDim.x + threadIdx.x) % 32 == 0) {
//            printf(" warpId = %d ",warpId);
//            printf(" aRowIdx = %d ",aRowIdx);
//            printf(" aColIdx = %d ",aColIdx);
//            printf(" bRowIdx = %d ",bRowIdx);
//            printf(" bColIdx = %d ",bColIdx);
//            printf("\n");
//        }

        const auto aTileOffsetPrt = mtrA + aRowIdx + aColIdx * lda;
        const auto bTileOffsetPrt = mtrB + bRowIdx + bColIdx * ldb;

        if (aRowIdx < K && aColIdx < M && bRowIdx < N && bColIdx < K) {
            wmma::load_matrix_sync(aFrag, aTileOffsetPrt, lda);
            wmma::load_matrix_sync(bFrag, bTileOffsetPrt, ldb);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }
    }

    int cRowIdx = warpId * WMMA_M;
    int cColIdx = warpId * WMMA_N;
    int cOffset = cRowIdx + cColIdx * ldc;

    const auto cTileOffsetPrt = mtrC + cRowIdx + cColIdx * ldc;
    if ((blockIdx.x * blockDim.x + threadIdx.x) % 32 == 0) {
        printf(" warpId = %d ",warpId);
        printf(" cRowIdx = %d ",cRowIdx);
        printf(" cColIdx = %d ",cColIdx);
        printf(" cRowIdx + cColIdx * ldc = %d ",cRowIdx + cColIdx * ldc);
        printf("\n");
    }

    if (cRowIdx < M && cColIdx < N) {
        wmma::load_matrix_sync(cFrag, cTileOffsetPrt, ldc, wmma::mem_row_major);

#pragma unroll
        for (int i = 0; i < cFrag.num_elements; ++i) {
            cFrag.x[i] = alpha * accFrag.x[i] + beta * cFrag.x[i];
        }

        wmma::store_matrix_sync(cTileOffsetPrt, cFrag, ldc, wmma::mem_row_major);
    }

}

/* error checking */
bool checkData(const int num, const float *dataDev1, const float *dataDev2) {
    fprintf(stdout, "\nChecking results...\n");

    float *dataHost = (float *) malloc(num * sizeof(float));
    float *dataHost2 = (float *) malloc(num * sizeof(float));

    cudaErrCheck(cudaMemcpy(dataHost, dataDev1, num * sizeof(float), cudaMemcpyDeviceToHost));
    cudaErrCheck(cudaMemcpy(dataHost2, dataDev2, num * sizeof(float), cudaMemcpyDeviceToHost));

    int errors = 0;
    for (int idx = 0; idx < num; ++idx) {
        float oneData1 = dataHost[idx];
        float oneData2 = dataHost2[idx];
        float diffDats = fabs(oneData1 - oneData2);

        float relativeErr = diffDats / oneData1;
        float eps = 1e-4;
        if (relativeErr >= eps) {
            ++errors;
            if (errors < 10) {
                fprintf(stderr, "error : data1 = %f, data2 = %f\n", oneData1, oneData2);
            }
        }
    }
    fflush(stderr);

    free(dataHost);
    free(dataHost2);

    if (errors > 0) {
        fprintf(stderr, "Inconsistent data! %d errors!\n", errors);

        fflush(stderr);
        return false;
    }

    fprintf(stdout, "Result validates successfully.\n");

    return true;
}