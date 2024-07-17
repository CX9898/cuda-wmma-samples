#pragma once

#include "matrixSetting.hpp"
#include <vector>

// All three matrices A,B,C must be row-major ordered
template<typename T>
void mmaHost(const int M, const int N, const int K,
             const T alpha, const T beta,
             const std::vector<T> &mtrA, const std::vector<T> &mtrB, std::vector<T> &mtrC);

/* error checking */
bool checkData(const int num, const float *data1, const float *data2);

bool checkData(const int num, const std::vector<float> &dataHost1, const float *dataDev2);

bool checkData(const int num, const float *dataDev, const std::vector<float> &dataHost2);

bool checkDevData(const int num, const float *dataDev1, const float *dataDev2);