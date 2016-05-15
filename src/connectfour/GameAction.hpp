
#pragma once

#include <ostream>
#include <vector>

namespace connectfour {

class GameAction {
  unsigned col;

public:
  // Returns a vector of all possible actions in the game.
  static const std::vector<GameAction> &ALL_ACTIONS(void);

  static GameAction ACTION(unsigned index);
  static std::vector<GameAction> ACTIONS(std::vector<unsigned> indices);
  static unsigned ACTION_INDEX(const GameAction &ga);

  GameAction() = default; // useful to have a no args constructor
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
