#pragma once

#include "../common/Common.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include "StateActionRater.hpp"

#include <vector>

namespace mcts {

// An action and the expected utility of taking that action.
using ActionUtility = pair<GameAction, float>;

class MCTS {
public:
  MCTS(uptr<StateActionRater> rater, unsigned maxPlayoutDepth);
  virtual ~MCTS();

  // Sorted list of action utilities.
  vector<ActionUtility> ComputeUtilities(const GameState &state, unsigned iterations);

private:
  struct MCTSImpl;
  uptr<MCTSImpl> impl;
};
}
