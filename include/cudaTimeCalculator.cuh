#pragma once

class cudaTimeCalculator {
 public:
  cudaTimeCalculator();
  ~cudaTimeCalculator();

  void startClock();
  void endClock();
  float getTime();

 private:
  cudaEvent_t star_;
  cudaEvent_t stop_;

  float time_;
};

