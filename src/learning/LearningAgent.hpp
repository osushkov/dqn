
#pragma once

#include "../common/Common.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include "Agent.hpp"
#include "ExperienceMoment.hpp"

using namespace connectfour;

namespace learning {

class LearningAgent : public Agent {
public:
  static EVector EncodeGameState(const GameState *state);

  LearningAgent();
  virtual ~LearningAgent();

  GameAction SelectAction(const GameState *state) override;

  void SetPRandom(float pRandom);
  void SetMaxQTemperature(float temp);

  GameAction SelectLearningAction(const GameState *state, const EVector &encodedState);
  void Learn(const vector<ExperienceMoment> &moments);

  // This is for debugging.
  float GetQValue(const GameState &state, const GameAction &action) const;

private:
  struct LearningAgentImpl;
  uptr<LearningAgentImpl> impl;
};
}
