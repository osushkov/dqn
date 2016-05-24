#pragma once

#include "../math/Math.hpp"

namespace neuralnetwork {

struct TrainingSample {
  EVector input;
  float expectedOutput;
  unsigned outputIndex;

  TrainingSample(const EVector &input, float expectedOutput, unsigned outputIndex)
      : input(input), expectedOutput(expectedOutput), outputIndex(outputIndex) {}
};
}
