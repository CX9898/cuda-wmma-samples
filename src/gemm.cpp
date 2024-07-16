#include "host.hpp"

template<typename T>
void matrixTileMultiplication(const int tileSizeM, const int tileSizeN, const int tileSizeK,
                              const int lda, const int ldb, const int ldc,
                              const T *aPtr, const T *bPtr, T *cPtr) {
    for (int rowIter = 0; rowIter < tileSizeM; ++rowIter) {
        for (int colIter = 0; colIter < tileSizeN; ++colIter) {
            for (int kIter = 0; kIter < tileSizeK; ++kIter) {
                *(cPtr + rowIter * ldc + colIter) += *(aPtr + rowIter * lda + kIter) * *(bPtr + colIter + kIter * ldb);
            }
        }
    }

}

// All three matrices A,B,C must be row-major ordered
template<typename T>
void gemmHost(const int M, const int N, const int K,
              const T alpha, const T beta,
              const std::vector<T> &mtrA, const std::vector<T> &mtrB, std::vector<T> &mtrC) {
    const int numMtrC = M * N;
    mtrC.resize(numMtrC);

#pragma omp parallel for
    for (int cRowTileIter = 0; cRowTileIter < M; cRowTileIter += WMMA_M) {
        for (int cColTileIter = 0; cColTileIter < N; cColTileIter += WMMA_N) {
            const int lda = K;
            const int ldb = N;
            const int ldc = N;

            const int numCTile = WMMA_M * WMMA_N;
            std::vector<T> acc(numCTile);

            for (int tileIter = 0; tileIter < K; tileIter += WMMA_K) {
                const auto aOffsetPtr = mtrA.data() + cRowTileIter * lda + tileIter;
                const auto bOffsetPtr = mtrB.data() + (cRowTileIter + tileIter) * ldc + cColTileIter;
                matrixTileMultiplication(WMMA_M, WMMA_N, WMMA_K, lda, ldb, ldc, aOffsetPtr, bOffsetPtr, acc);
            }
            auto cOffsetPtr = mtrC.data() + cRowTileIter * ldc + cColTileIter;


        }
    }
}