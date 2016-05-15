
#pragma once

#include "../common/Common.hpp"
#include "CellState.hpp"
#include "Constants.hpp"
#include "GameAction.hpp"
#include <array>
#include <iosfwd>
#include <vector>

namespace connectfour {

class GameState {
  array<CellState, BOARD_WIDTH * BOARD_HEIGHT> cells;
  array<unsigned char, BOARD_WIDTH> colHeights;
  unsigned numTokensOnBoard;

public:
  GameState();
  GameState(const GameState &other);
  GameState(GameState &&other);

  GameState &operator=(const GameState &other);
  bool operator==(const GameState &other) const;

  void PlaceToken(unsigned col);
  CellState GetCell(unsigned row, unsigned col) const;
  unsigned NumTokensOnBoard(void) const;

  // Whenever we make a move and want another agent to make a move, then we should "flip" the
  // board such that what are currently "our" tokens become "oppponent" tokens, and vice versa.
  void FlipState(void);

  size_t HashCode() const;

  // Returns indices into the GameAction::ALL_ACTIONS vector.
  vector<unsigned> AvailableActions(void) const; // TODO: this should do a JIT caching
  GameState SuccessorState(const GameAction &action) const;
};
}

std::ostream &operator<<(std::ostream &stream, const connectfour::GameState &gs);
