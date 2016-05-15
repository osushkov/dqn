
#pragma once

#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"

using namespace connectfour;

namespace learning {

class Agent {
public:
  virtual ~Agent() = default;
  virtual GameAction SelectAction(const GameState *state) = 0;
};
}
