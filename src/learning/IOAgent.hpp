
#pragma once

#include "Agent.hpp"

namespace learning {

class IOAgent : public Agent {
public:
  GameAction SelectAction(const GameState *state) override;
};
}
