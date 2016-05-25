#pragma once

#include "../math/Math.hpp"

namespace neuralnetwork {

struct TrainingSample {
  EVector startState;
  EVector endState;
  unsigned actionTaken;

  bool isEndStateTerminal;
  float rewardGained;
  float futureRewardDiscount;

  TrainingSample(const EVector &startState, const EVector &endState, unsigned actionTaken,
                 bool isEndStateTerminal, float rewardGained, float futureRewardDiscount)
      : startState(startState), endState(endState), actionTaken(actionTaken),
        isEndStateTerminal(isEndStateTerminal), rewardGained(rewardGained),
        futureRewardDiscount(futureRewardDiscount) {}
};
}
