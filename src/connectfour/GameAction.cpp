
#include "GameAction.hpp"
#include "Constants.hpp"
#include <cassert>
#include <mutex>

using namespace connectfour;

static std::once_flag stateFlag;
static std::vector<GameAction> actionSet;

const std::vector<GameAction> &GameAction::ALL_ACTIONS(void) {
  std::call_once(stateFlag, []() {
    for (unsigned i = 0; i < BOARD_WIDTH; i++) {
      actionSet.push_back(GameAction(i));
    }
  });

  return actionSet;
}

GameAction GameAction::ACTION(unsigned index) { return ALL_ACTIONS()[index]; }

std::vector<GameAction> GameAction::ACTIONS(std::vector<unsigned> indices) {
  std::vector<GameAction> result;
  result.reserve(indices.size());

  for (auto i : indices) {
    result.push_back(ALL_ACTIONS()[i]);
  }

  return result;
}

unsigned GameAction::ACTION_INDEX(const GameAction &ga) {
  assert(ga.GetColumn() < BOARD_WIDTH);
  return ga.GetColumn();
}
