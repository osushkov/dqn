
#pragma once

#include "../common/Common.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include "../mcts/StateActionRater.hpp"
#include "Agent.hpp"
#include "ExperienceMoment.hpp"

#include <iostream>

using namespace connectfour;
using namespace mcts;

namespace learning {

class LearningAgent : public Agent, public StateActionRater {
public:
  static EVector EncodeGameState(const GameState *state);

  LearningAgent();
  virtual ~LearningAgent();

  static uptr<LearningAgent> Read(std::istream &in);
  void Write(std::ostream &out);

  GameAction SelectAction(const GameState *state) override;

  void SetPRandom(float pRandom);
  void SetTemperature(float temperature);

  GameAction SelectLearningAction(const GameState *state, const EVector &encodedState);
  void Learn(const vector<ExperienceMoment> &moments, float learnRate);

  void Finalise(void);

  // This is for debugging.
  float GetQValue(const GameState &state, const GameAction &action) const;

  float RateGameState(const GameState &state) override;
  vector<float> RateAvailableActions(const GameState &state) override;

private:
  struct LearningAgentImpl;
  uptr<LearningAgentImpl> impl;
};
}
