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

//
__global__ void wmmaExample(const int M, const int N, const int K,
                            const float alpha, const float beta,
                            const half *mtrA, const half *mtrB, float *mtrC) {
    // Leading dimensions. Packed with no transpositions.
    int lda = K;
    int ldb = M;
    int ldc = N;

    // Tile using a 2D grid
    int warpIdM = (int) (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpIdN = (int) (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> accFrag; // Fragment accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> cFrag;

    wmma::fill_fragment(accFrag, 0.0f);

    for (int i = 0; i < K; i += WMMA_K) {
        int aRowLoc = warpIdM * WMMA_M;
        int aColLoc = i;

        int bRowLoc = i;
        int bColLoc = warpIdN * WMMA_N;

        const auto aTilePrt = ;
        const auto bTilePrt = ;

        if (aRowLoc < M && aColLoc < K && bRowLoc < K && bColLoc < N) {
            wmma::load_matrix_sync(aFrag, aTilePrt, K);
            wmma::load_matrix_sync(bFrag, bTilePrt, N);

            wmma::mma_sync(accFrag, aFrag, bFrag, accFrag);
        }

    }

    int cRowLoc = warpIdM * WMMA_M;
    int cColLoc = warpIdN * WMMA_N;

    const auto cTilePrt = mtrC + cRowLoc + cColLoc * ldc;

    if (cRowLoc < M && cColLoc < N) {
        wmma::load_matrix_sync(cFrag, cTilePrt, ldc, wmma::mem_row_major);

#pragma unroll
        for (int i = 0; i < cFrag.num_elements; ++i) {
            cFrag.x[i] = alpha * accFrag.x[i] + beta * cFrag.x[i];
        }

        wmma::store_matrix_sync(cTilePrt, cFrag, ldc, wmma::mem_row_major);
    }

}

int main() {
    float *aFp32;
    float *bFp32;

    half *aFp16;
    half *bFp16;

    float *cWmmaEx;
    float *cCublas;

    const float alpha = 2.0f;
    const float beta = 2.0f;

    const int numMatrixADates = MATRIX_M * MATRIX_K;
    const int numMatrixBDates = MATRIX_K * MATRIX_N;
    const int numMatrixCDates = MATRIX_M * MATRIX_N;

    cudaErrCheck(cudaMalloc((void **) &aFp32, numMatrixADates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &bFp32, numMatrixBDates * sizeof(float)));

    cudaErrCheck(cudaMalloc((void **) &aFp16, numMatrixADates * sizeof(half)));
    cudaErrCheck(cudaMalloc((void **) &bFp16, numMatrixBDates * sizeof(half)));

    cudaErrCheck(cudaMalloc((void **) &cCublas, numMatrixCDates * sizeof(float)));
    cudaErrCheck(cudaMalloc((void **) &cWmmaEx, numMatrixCDates * sizeof(float)));

    /* using curand to initialize */
    {
        curandGenerator_t curandGen;

        curandErrCheck(curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT));
        curandErrCheck(curandSetPseudoRandomGeneratorSeed(curandGen, 1337ULL));

        curandErrCheck(curandGenerateUniform(curandGen, aFp32, numMatrixADates));
        curandErrCheck(curandGenerateUniform(curandGen, bFp32, numMatrixBDates));

        float *c;
        cudaErrCheck(cudaMalloc((void **) &c, numMatrixCDates * sizeof(float)));
        curandErrCheck(curandGenerateUniform(curandGen, c, numMatrixCDates));

        cudaErrCheck(cudaMemcpy(cCublas, c, numMatrixCDates, cudaMemcpyDeviceToDevice));
        cudaErrCheck(cudaMemcpy(cWmmaEx, c, numMatrixCDates, cudaMemcpyDeviceToDevice));

        curandErrCheck(curandDestroyGenerator(curandGen));

        const int numThreadPerBlock = 256;
        convertFp32ToFp16<<< (numMatrixADates + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            aFp16, aFp32, numMatrixADates);
        convertFp32ToFp16<<< (numMatrixBDates + numThreadPerBlock - 1) / numThreadPerBlock, numThreadPerBlock>>>(
            bFp16, bFp32, numMatrixBDates);
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
                                    aFp16, CUDA_R_16F, MATRIX_M,
                                    bFp16, CUDA_R_16F, MATRIX_K,
                                    &beta,
                                    cCublas, CUDA_R_32F, MATRIX_M,
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

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        dim3 gridDim;
        dim3 blockDim;

        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmmaExample<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aFp16, bFp16, cWmmaEx);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    /* using wmma-example computation */
    {
        printf("Running with wmma-example...\n");

        cudaEvent_t startWmmaEx;
        cudaEvent_t stopWmmaEx;

        cudaErrCheck(cudaEventCreate(&startWmmaEx));
        cudaErrCheck(cudaEventCreate(&stopWmmaEx));

        dim3 gridDim;
        dim3 blockDim;

        // blockDim.x must be a multple of warpSize
        // 128x4 means we have 16 warps and a block computes a 64x64 output tile
        blockDim.x = 128;
        blockDim.y = 4;

        gridDim.x = (MATRIX_M + (WMMA_M * blockDim.x / 32 - 1)) / (WMMA_M * blockDim.x / 32);
        gridDim.y = (MATRIX_N + WMMA_N * blockDim.y - 1) / (WMMA_N * blockDim.y);

        cudaErrCheck(cudaEventRecord(startWmmaEx));
        wmma_example<<<gridDim, blockDim>>>(MATRIX_M, MATRIX_N, MATRIX_K, alpha, beta, aFp16, bFp16, cWmmaEx);
        cudaErrCheck(cudaEventRecord(stopWmmaEx));
        cudaErrCheck(cudaEventSynchronize(stopWmmaEx));

        float wmmaTime;
        cudaErrCheck(cudaEventElapsedTime(&wmmaTime, startWmmaEx, stopWmmaEx));
        printf("wmmaExample time : %fms\n", wmmaTime);

        cudaErrCheck(cudaEventDestroy(startWmmaEx));
        cudaErrCheck(cudaEventDestroy(stopWmmaEx));
    }

    /* error checking */
    {
        printf("\nChecking results...\n");

        float *cCublasHost = (float *) malloc(numMatrixCDates * sizeof(float));
        float *cWmmaExHost = (float *) malloc(numMatrixCDates * sizeof(float));

        cudaErrCheck(cudaMemcpy(cWmmaExHost, cWmmaEx, numMatrixCDates * sizeof(float), cudaMemcpyDeviceToHost));
        cudaErrCheck(cudaMemcpy(cCublasHost, cCublas, numMatrixCDates * sizeof(float), cudaMemcpyDeviceToHost));

        int errors = 0;
        for (int idx = 0; idx < numMatrixCDates; ++idx) {
            float cublasRes = cCublasHost[idx];
            float wmmaExRes = cWmmaExHost[idx];
            float diffDats = fabs(cublasRes - wmmaExRes);

            float relativeErr = diffDats / cublasRes;
            float eps = 1e-4;
            if (relativeErr >= eps) {
                ++errors;
                if (errors < 10) {
                    printf("error : cublasRes = %f, wmmaExRes = %f\n", cublasRes, wmmaExRes);
                }
            }
        }

        if (errors > 0) {
            printf("wmmaExample does not agree with cuBLAS! %d errors!\n", errors);
        } else {
            printf("Results verified: cublas and WMMA agree.\n");
        }

        free(cCublasHost);
        free(cWmmaExHost);
    }

    cudaErrCheck(cudaFree(aFp32));
    cudaErrCheck(cudaFree(bFp32));
    cudaErrCheck(cudaFree(aFp16));
    cudaErrCheck(cudaFree(bFp16));
    cudaErrCheck(cudaFree(cCublas));
    cudaErrCheck(cudaFree(cWmmaEx));

    return 0;
}