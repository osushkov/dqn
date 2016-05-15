
#include "IOAgent.hpp"
#include "../common/Common.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include <algorithm>
#include <iostream>
#include <vector>

using namespace learning;
using namespace connectfour;

GameAction IOAgent::SelectAction(const GameState *state) {
  cout << *state << endl;
  cout << "your move:" << endl;

  vector<GameAction> available = GameAction::ACTIONS(state->AvailableActions());
  while (true) {
    unsigned col;
    cin >> col;

    GameAction pa(col);
    if (std::find(available.begin(), available.end(), pa) != available.end()) {
      return pa;
    } else {
      cout << "invalid move" << endl;
    }
  }

  return GameAction::ALL_ACTIONS()[0];
}
