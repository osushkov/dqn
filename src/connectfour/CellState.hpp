
#pragma once

#include <ostream>

namespace connectfour {
enum class CellState { MY_TOKEN = 0, OPPONENT_TOKEN = 1, EMPTY = 2 };
}

inline std::ostream &operator<<(std::ostream &stream, const connectfour::CellState &cs) {
  switch (cs) {
  case connectfour::CellState::MY_TOKEN:
    stream << " # ";
    break;
  case connectfour::CellState::OPPONENT_TOKEN:
    stream << " 0 ";
    break;
  case connectfour::CellState::EMPTY:
    stream << " . ";
    break;
  }
  return stream;
}
