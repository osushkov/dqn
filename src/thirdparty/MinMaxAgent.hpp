#pragma once

#include "../learning/Agent.hpp"

namespace learning {

class MinMaxAgent : public Agent {
public:
  GameAction SelectAction(const GameState *state) override;
};
}
