#pragma once

#include "../common/Common.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include <vector>

using namespace connectfour;

namespace mcts {

class StateActionRater {
public:
  virtual ~StateActionRater() = default;

  virtual float RateGameState(const GameState &state) = 0;
  virtual vector<float> RateAvailableActions(const GameState &state) = 0;
};
}
