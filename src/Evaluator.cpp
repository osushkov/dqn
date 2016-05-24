
#include "Evaluator.hpp"
#include "common/Common.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include <cassert>
#include <vector>

using namespace connectfour;

Evaluator::Evaluator(unsigned numTrials) : numTrials(numTrials) { assert(numTrials > 0); }

std::pair<float, float> Evaluator::Evaluate(learning::Agent *primary,
                                            learning::Agent *opponent) const {
  assert(primary != nullptr && opponent != nullptr);

  unsigned numWins = 0;
  unsigned numDraws = 0;
  for (unsigned i = 0; i < numTrials; i++) {
    int res = runTrial(primary, opponent);
    if (res == 0) {
      std::cout << "ITS A DAR!!!" << std::endl;
      numDraws++;
    } else if (res == 1) {
      numWins++;
    }
  }
  return make_pair(numWins / static_cast<float>(numTrials),
                   numDraws / static_cast<float>(numTrials));
}

int Evaluator::runTrial(learning::Agent *primary, learning::Agent *opponent) const {
  GameRules *rules = GameRules::Instance();
  vector<learning::Agent *> agents = {primary, opponent};

  unsigned curPlayerIndex = rand() % agents.size();
  GameState curState(rules->InitialState());

  while (true) {
    learning::Agent *curPlayer = agents[curPlayerIndex];
    GameAction action = curPlayer->SelectAction(&curState);
    curState = curState.SuccessorState(action);

    switch (rules->GameCompletionState(curState)) {
    case CompletionState::WIN:
      return curPlayer == primary ? 1 : -1;
    case CompletionState::LOSS:
      assert(false); // This actually shouldn't be possible.
      return curPlayer == primary ? -1 : 1;
    case CompletionState::DRAW:
      return 0;
    case CompletionState::UNFINISHED:
      curPlayerIndex = (curPlayerIndex + 1) % agents.size();
      curState.FlipState();
      break;
    }
  }

  assert(false);
  return 0;
}
