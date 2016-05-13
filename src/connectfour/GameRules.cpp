
#include "GameRules.hpp"
#include <cassert>

using namespace connectfour;

static bool isWin(const GameState &state);
static bool isLoss(const GameState &state);
static bool haveVerticalRun(const GameState &state, unsigned length, CellState targetToken);
static bool haveHorizontalRun(const GameState &state, unsigned length, CellState targetToken);
static bool haveDiagonalRun(const GameState &state, unsigned length, CellState targetToken);

GameRules *GameRules::Instance(void) {
  static GameRules rules;
  return &rules;
}

GameState GameRules::InitialState(void) const { return GameState(); }

CompletionState GameRules::GameCompletionState(const GameState &state) const {
  if (state.NumTokensOnBoard() < COMPLETION_RUN) {
    return CompletionState::UNFINISHED;
  } else if (isWin(state)) {
    return CompletionState::WIN;
  } else if (isLoss(state)) {
    return CompletionState::LOSS;
  } else if (state.NumTokensOnBoard() == BOARD_WIDTH * BOARD_HEIGHT) {
    return CompletionState::DRAW;
  } else {
    return CompletionState::UNFINISHED;
  }
}

bool isWin(const GameState &state) {
  return haveHorizontalRun(state, COMPLETION_RUN, CellState::MY_TOKEN) ||
         haveVerticalRun(state, COMPLETION_RUN, CellState::MY_TOKEN) ||
         haveDiagonalRun(state, COMPLETION_RUN, CellState::MY_TOKEN);
}

bool isLoss(const GameState &state) {
  return haveHorizontalRun(state, COMPLETION_RUN, CellState::OPPONENT_TOKEN) ||
         haveVerticalRun(state, COMPLETION_RUN, CellState::OPPONENT_TOKEN) ||
         haveDiagonalRun(state, COMPLETION_RUN, CellState::OPPONENT_TOKEN);
}

bool haveVerticalRun(const GameState &state, unsigned length, CellState targetToken) {
  for (unsigned r = 0; r < BOARD_HEIGHT - length + 1; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH; c++) {

      bool found = true;
      for (unsigned i = 0; i < length; i++) {
        if (state.GetCell(r + i, c) != targetToken) {
          found = false;
          break;
        }
      }

      if (found) {
        return true;
      }
    }
  }

  return false;
}

bool haveHorizontalRun(const GameState &state, unsigned length, CellState targetToken) {
  for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH - length + 1; c++) {

      bool found = true;
      for (unsigned i = 0; i < length; i++) {
        if (state.GetCell(r, c + i) != targetToken) {
          found = false;
          break;
        }
      }

      if (found) {
        return true;
      }
    }
  }

  return false;
}

bool haveDiagonalRun(const GameState &state, unsigned length, CellState targetToken) {
  for (unsigned r = 0; r < BOARD_HEIGHT - length + 1; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH - length + 1; c++) {

      bool found = true;
      for (unsigned i = 0; i < length; i++) {
        if (state.GetCell(r + i, c + i) != targetToken) {
          found = false;
          break;
        }
      }

      if (found) {
        return true;
      }
    }
  }

  for (unsigned r = 0; r < BOARD_HEIGHT - length + 1; r++) {
    for (unsigned c = length - 1; c < BOARD_WIDTH; c++) {

      bool found = true;
      for (unsigned i = 0; i < length; i++) {
        if (state.GetCell(r + i, c - i) != targetToken) {
          found = false;
          break;
        }
      }

      if (found) {
        return true;
      }
    }
  }
  return false;
}
