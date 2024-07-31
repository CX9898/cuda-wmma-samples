#include "cudaTimeCalculator.cuh"
#include "cudaErrorCheck.cuh"

cudaTimeCalculator::cudaTimeCalculator(){
    time_ = 0.0f;

    cudaErrCheck(cudaEventCreate(&star_));
    cudaErrCheck(cudaEventCreate(&stop_));
}

cudaTimeCalculator::~cudaTimeCalculator() {
    cudaErrCheck(cudaEventDestroy(star_));
    cudaErrCheck(cudaEventDestroy(stop_));
}

void cudaTimeCalculator::startClock() {
    cudaErrCheck(cudaEventRecord(star_));
}

void cudaTimeCalculator::endClock() {
    cudaErrCheck(cudaEventRecord(stop_));
    cudaErrCheck(cudaEventSynchronize(stop_));
}

float cudaTimeCalculator::getTime() {
    cudaErrCheck(cudaEventElapsedTime(&time_, star_, stop_));
    return time_;
}