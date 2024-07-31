#include "cudaTimeCalculator.cuh"
#include "cudaErrorCheck.cuh"

cudaTimeCalculator::cudaTimeCalculator(){
    time = 0.0f;

    cudaErrCheck(cudaEventCreate(&star));
    cudaErrCheck(cudaEventCreate(&stop));
}

cudaTimeCalculator::~cudaTimeCalculator() {
    cudaErrCheck(cudaEventDestroy(star));
    cudaErrCheck(cudaEventDestroy(stop));
}

void cudaTimeCalculator::startClock() {
    cudaErrCheck(cudaEventRecord(star));
}

void cudaTimeCalculator::endClock() {
    cudaErrCheck(cudaEventRecord(stop));
    cudaErrCheck(cudaEventSynchronize(stop));
}

float cudaTimeCalculator::getTime() {
    cudaErrCheck(cudaEventElapsedTime(&time, star, stop));
    return time;
}