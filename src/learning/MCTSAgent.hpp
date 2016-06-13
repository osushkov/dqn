#pragma once

#include "../common/Common.hpp"
#include "../mcts/StateActionRater.hpp"
#include "Agent.hpp"

namespace learning {

class MCTSAgent : public Agent {
public:
  MCTSAgent(uptr<mcts::StateActionRater> rater);
  virtual ~MCTSAgent();

  GameAction SelectAction(const GameState *state) override;

private:
  struct MCTSAgentImpl;
  uptr<MCTSAgentImpl> impl;
};
}
