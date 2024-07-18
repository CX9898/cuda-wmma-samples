#pragma once

#include "matrixSetting.hpp"
#include <vector>

// All three matrices A,B,C must be row-major ordered
void mmaHost(const int M, const int N, const int K,
             const float alpha, const float beta,
             const std::vector<float> &mtrA, const std::vector<float> &mtrB, std::vector<float> &mtrC);

/* error checking */
bool checkData(const int num, const float *data1, const float *data2);

bool checkData(const int num, const std::vector<float> &dataHost1, const float *dataDev2);

bool checkData(const int num, const float *dataDev1, const std::vector<float> &dataHost2);

bool checkDevData(const int num, const float *dataDev1, const float *dataDev2);