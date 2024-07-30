#pragma once

// The dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Must be multiples of 16 for wmma code to work
const int MATRIX_M = 500 * WMMA_M;
const int MATRIX_N = 500 * WMMA_N;
const int MATRIX_K = 500 * WMMA_K;

const int MATRIX_A_SIZE = MATRIX_M * MATRIX_K;
const int MATRIX_B_SIZE = MATRIX_K * MATRIX_N;
const int MATRIX_C_SIZE = MATRIX_M * MATRIX_N;