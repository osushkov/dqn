
#pragma once

#include "GameState.hpp"

namespace connectfour {

enum class CompletionState { UNFINISHED, WIN, LOSS, DRAW };

class GameRules {
public:
  static GameRules *Instance(void); // singleton

  GameState InitialState(void) const;
  CompletionState GameCompletionState(const GameState &state) const;

private:
  GameRules() = default;
};
}
