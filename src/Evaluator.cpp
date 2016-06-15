
#include "Evaluator.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "common/Util.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/RandomAgent.hpp"
#include <cassert>
#include <vector>

using namespace connectfour;

static void printStates(const std::vector<GameState> &states) {
  for (const auto &gs : states) {
    std::cout << gs << std::endl << std::endl;
  }

  std::cout << "---------------------------" << std::endl << std::endl;
}

Evaluator::Evaluator(unsigned numTrials) : numTrials(numTrials) { assert(numTrials > 0); }

std::pair<float, float> Evaluator::Evaluate(learning::Agent *primary, learning::Agent *opponent) {
  assert(primary != nullptr && opponent != nullptr);

  primaryMicroSecondsElapsed = 0;
  opponentMicroSecondsElapsed = 0;
  primaryActions = 0;
  secondaryActions = 0;

  unsigned numWins = 0;
  unsigned numDraws = 0;
  for (unsigned i = 0; i < numTrials; i++) {
    int res = runTrial(primary, opponent);
    std::cout << "trial! winner: " << res << std::endl;
    if (res == 0) {
      // std::cout << "ITS A DRAW!!!" << std::endl;
      numDraws++;
    } else if (res == 1) {
      numWins++;
    }
  }

  // std::cout << "Primary agent micro-seconds per move: "
  //           << (primaryMicroSecondsElapsed / static_cast<float>(primaryActions)) << std::endl;
  // std::cout << "Opponent agent micro-seconds per move: "
  //           << (opponentMicroSecondsElapsed / static_cast<float>(secondaryActions)) << std::endl;

  return make_pair(numWins / static_cast<float>(numTrials),
                   numDraws / static_cast<float>(numTrials));
}

int Evaluator::runTrial(learning::Agent *primary, learning::Agent *opponent) {
  GameRules *rules = GameRules::Instance();
  vector<learning::Agent *> agents = {primary, opponent};
  learning::RandomAgent randomAgent;

  std::vector<GameState> states;

  unsigned curPlayerIndex = rand() % agents.size();
  GameState curState(rules->InitialState());
  curState = curState.SuccessorState(randomAgent.SelectAction(&curState));
  curState.FlipState();
  curState = curState.SuccessorState(randomAgent.SelectAction(&curState));
  curState.FlipState();

  while (true) {
    learning::Agent *curPlayer = agents[curPlayerIndex];

    GameAction action;
    if (Util::RandInterval(0.0, 1.0) < 0.0) {
      action = randomAgent.SelectAction(&curState);
    } else {
      Timer timer;
      timer.Start();
      action = curPlayer->SelectAction(&curState);
      timer.Stop();

      if (curPlayer == primary) {
        primaryActions++;
        primaryMicroSecondsElapsed += timer.GetNumElapsedMicroseconds();
      } else {
        secondaryActions++;
        opponentMicroSecondsElapsed += timer.GetNumElapsedMicroseconds();
      }
    }

    if (curPlayer == primary) {
      states.push_back(curState);
    }

    curState = curState.SuccessorState(action);

    switch (rules->GameCompletionState(curState)) {
    case CompletionState::WIN:
      curState.FlipState();
      states.push_back(curState);

      if (curPlayer != primary) {
        // printStates(states);
      }
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
