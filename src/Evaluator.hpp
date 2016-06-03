
#pragma once

#include "learning/Agent.hpp"

class Evaluator {
  unsigned numTrials;

  unsigned primaryMicroSecondsElapsed;
  unsigned opponentMicroSecondsElapsed;

  unsigned primaryActions;
  unsigned secondaryActions;

public:
  Evaluator(unsigned numTrials);
  ~Evaluator() = default;

  // returns the ratio of wins+draws for the primary agent
  std::pair<float, float> Evaluate(learning::Agent *primary, learning::Agent *opponent);

private:
  // returns 1 for primary win, 0 for draw, -1 for opponent win
  int runTrial(learning::Agent *primary, learning::Agent *opponent);
};
