
#include "GameState.hpp"
#include "GameAction.hpp"
#include <cassert>
#include <ostream>

using namespace connectfour;

GameState::GameState() : numTokensOnBoard(0) {
  cells.fill(CellState::EMPTY);
  colHeights.fill(0);
}

GameState::GameState(const GameState &other)
    : cells(other.cells), colHeights(other.colHeights), numTokensOnBoard(other.numTokensOnBoard) {}

GameState::GameState(GameState &&other)
    : cells(move(other.cells)), colHeights(move(other.colHeights)),
      numTokensOnBoard(other.numTokensOnBoard) {}

GameState &GameState::operator=(const GameState &other) {
  this->cells = other.cells;
  this->colHeights = other.colHeights;
  this->numTokensOnBoard = other.numTokensOnBoard;
  return *this;
}

bool GameState::operator==(const GameState &other) const {
  assert(cells.size() == other.cells.size());

  for (unsigned i = 0; i < cells.size(); i++) {
    if (cells[i] != other.cells[i]) {
      return false;
    }
  }

  return true;
}

void GameState::PlaceToken(unsigned col) {
  assert(col < BOARD_WIDTH);
  assert(colHeights[col] <= BOARD_HEIGHT);

  if (colHeights[col] == BOARD_HEIGHT) {
    assert(false);
    return;
  }

  unsigned index = col + colHeights[col] * BOARD_WIDTH;

  assert(cells[index] == CellState::EMPTY);
  cells[index] = CellState::MY_TOKEN;
  colHeights[col]++;
  numTokensOnBoard++;
}

CellState GameState::GetCell(unsigned row, unsigned col) const {
  assert(row < BOARD_HEIGHT && col < BOARD_WIDTH);
  return cells[col + row * BOARD_WIDTH];
}

unsigned GameState::NumTokensOnBoard(void) const {
  assert(numTokensOnBoard <= BOARD_WIDTH * BOARD_HEIGHT);
  return numTokensOnBoard;
}

void GameState::FlipState(void) {
  for (auto &cs : cells) {
    if (cs == CellState::MY_TOKEN) {
      cs = CellState::OPPONENT_TOKEN;
    } else if (cs == CellState::OPPONENT_TOKEN) {
      cs = CellState::MY_TOKEN;
    }
  }
}

size_t GameState::HashCode(void) const {
  size_t result = 0;
  for (auto &c : cells) {
    result *= 3;
    result += static_cast<size_t>(c);
  }
  return result;
}

// TODO: this should be cached as it doesnt change.
vector<unsigned> GameState::AvailableActions(void) const {
  const vector<GameAction> &actionSet = GameAction::ALL_ACTIONS();

  vector<unsigned> result;
  result.reserve(actionSet.size());

  for (unsigned i = 0; i < actionSet.size(); i++) {
    if (colHeights[actionSet[i].GetColumn()] < BOARD_HEIGHT) {
      result.push_back(i);
    }
  }

  return result;
}

GameState GameState::SuccessorState(const GameAction &action) const {
  GameState result(*this);
  result.PlaceToken(action.GetColumn());
  return result;
}

std::ostream &operator<<(std::ostream &stream, const connectfour::GameState &gs) {
  for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH; c++) {
      stream << gs.GetCell(r, c);
    }
    stream << endl;
  }
  return stream;
}
