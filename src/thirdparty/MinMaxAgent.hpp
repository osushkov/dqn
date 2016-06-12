#pragma once

#include "../learning/Agent.hpp"

namespace learning {

class MinMaxAgent : public Agent {
public:
  MinMaxAgent(unsigned depth);
  GameAction SelectAction(const GameState *state) override;
};
}
