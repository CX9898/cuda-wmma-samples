#pragma once

class cudaTimeCalculator {
 public:
  cudaTimeCalculator();
  ~cudaTimeCalculator();

  void startClock();
  void endClock();
  float getTime();

 private:
  cudaEvent_t star;
  cudaEvent_t stop;
  float time;
};

