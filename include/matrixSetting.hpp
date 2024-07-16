#pragma once

// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

// Must be multiples of 16 for wmma code to work
const int MATRIX_M = 1 * WMMA_M;
const int MATRIX_N = 1 * WMMA_N;
const int MATRIX_K = 1 * WMMA_K;