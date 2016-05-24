
#pragma once

#include "Agent.hpp"
#include <vector>

using namespace std;

namespace learning {

class RandomAgent : public Agent {
public:
  GameAction SelectAction(const GameState *state) override {
    auto actions = state->AvailableActions();
    assert(actions.size() > 0);

    return GameAction::ACTION(actions[rand() % actions.size()]);
  }
};
}
