#pragma once

#include "../connectfour/GameState.hpp"
#include "../math/Math.hpp"
#include <cstdlib>

using namespace connectfour;

namespace learning {

struct ExperienceMoment {
  EVector initialState;
  GameAction actionTaken;
  EVector successorState;
  float reward;
  bool isSuccessorTerminal;

  ExperienceMoment() = default;
  ExperienceMoment(EVector initialState, GameAction actionTaken, EVector successorState,
                   float reward, bool isSuccessorTerminal)
      : initialState(initialState), actionTaken(actionTaken), successorState(successorState),
        reward(reward), isSuccessorTerminal(isSuccessorTerminal) {}
};
}
