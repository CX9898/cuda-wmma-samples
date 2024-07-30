#pragma once

// The dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Must be multiples of WMMA supported dimensions for wmma code to work
const int MATRIX_M = 1 * WMMA_M;
const int MATRIX_N = 1 * WMMA_N;
const int MATRIX_K = 1 * WMMA_K;

const int MATRIX_A_SIZE = MATRIX_M * MATRIX_K;
const int MATRIX_B_SIZE = MATRIX_K * MATRIX_N;
const int MATRIX_C_SIZE = MATRIX_M * MATRIX_N;