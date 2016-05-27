
#include "Evaluator.hpp"
#include "Trainer.hpp"
#include "common/Common.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"
#include <cstdlib>
#include <ctime>
#include <iostream>

using namespace learning;
using namespace connectfour;

int main(int argc, char **argv) {
  // srand(time(NULL));

  Trainer trainer;
  auto trainedAgent = trainer.TrainAgent(10000);

  learning::RandomAgent baselineAgent;

  Evaluator eval(1000);
  auto r = eval.Evaluate(trainedAgent.get(), &baselineAgent);
  std::cout << "r : " << r.first << " / " << r.second << std::endl;

  // RandomAgent ra;
  // Evaluator eval(1000);
  // auto r = eval.Evaluate(&ra, &ra);
  // std::cout << "r : " << r.first << " / " << r.second << std::endl;

  // learning::LearningAgent agent;
  // GameState gs = GameRules::Instance()->InitialState();
  // gs.PlaceToken(3);
  // gs.PlaceToken(5);
  //
  // std::vector<ExperienceMoment> learnMoments;
  // for (unsigned i = 0; i < GameAction::ALL_ACTIONS().size(); i++) {
  //   auto encodedStartState = LearningAgent::EncodeGameState(&gs);
  //   GameAction action = GameAction::ACTION(i);
  //   GameState successor = gs.SuccessorState(action);
  //   auto encodedSuccessor = LearningAgent::EncodeGameState(&successor);
  //
  //   if (i < GameAction::ALL_ACTIONS().size() / 2) {
  //     learnMoments.emplace_back(encodedStartState, action, encodedSuccessor, 1.0, true);
  //   } else {
  //     learnMoments.emplace_back(encodedStartState, action, encodedSuccessor, -1.0, true);
  //   }
  // }
  //
  // for (const auto &action : GameAction::ALL_ACTIONS()) {
  //   std::cout << "q value: " << action << " = " << agent.GetQValue(gs, action) << std::endl;
  // }
  //
  // std::cout << "learning..." << std::endl;
  // for (unsigned i = 0; i < 10000; i++) {
  //   agent.Learn(learnMoments);
  // }
  //
  // std::cout << "finished learning" << std::endl;
  // for (const auto &action : GameAction::ALL_ACTIONS()) {
  //   std::cout << "q value: " << action << " = " << agent.GetQValue(gs, action) << std::endl;
  // }

  return 0;
}
