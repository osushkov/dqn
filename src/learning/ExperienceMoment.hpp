#pragma once

#include "../connectfour/GameState.hpp"

using namespace connectfour;

namespace learning {

struct ExperienceMoment {
  GameState initialState;
  GameAction actionTaken;
  GameState successorState;
  float reward;

  ExperienceMoment() = default;
  ExperienceMoment(GameState initialState, GameAction actionTaken, GameState successorState,
                   float reward)
      : initialState(initialState), actionTaken(actionTaken), successorState(successorState),
        reward(reward) {}
};
}
