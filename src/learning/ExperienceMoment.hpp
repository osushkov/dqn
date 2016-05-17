#pragma once

#include "../connectfour/GameState.hpp"
#include "../math/Math.hpp"

using namespace connectfour;

namespace learning {

struct ExperienceMoment {
  EVector initialState;
  GameAction actionTaken;
  EVector successorState;
  float reward;

  ExperienceMoment() = default;
  ExperienceMoment(EVector initialState, GameAction actionTaken, EVector successorState,
                   float reward)
      : initialState(initialState), actionTaken(actionTaken), successorState(successorState),
        reward(reward) {}
};
}
