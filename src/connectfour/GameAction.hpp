
#pragma once

#include <ostream>

namespace connectfour {

class GameAction {
  unsigned col;

public:
  GameAction(unsigned col) : col(col) {}

  inline unsigned GetColumn(void) const { return col; }
  inline bool operator==(const GameAction &other) const { return col == other.col; }
  inline size_t HashCode(void) const { return col * 378551; }
};
}

inline std::ostream &operator<<(std::ostream &stream, const connectfour::GameAction &ga) {
  stream << "action_col( " << ga.GetColumn() << " )";
  return stream;
}
