
#include "Evaluator.hpp"
#include "Trainer.hpp"
#include "common/Common.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"
#include "thirdparty/MinMaxAgent.hpp"
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>

using namespace learning;
using namespace connectfour;

static constexpr bool DO_TRAINING = false;
static constexpr bool DO_EVALUATION = true;

int main(int argc, char **argv) {
  srand(time(NULL));

  // Train an agent.
  if (DO_TRAINING) {
    Trainer trainer;
    auto trainedAgent = trainer.TrainAgent(1000000);
    std::ofstream saveFile("agent.dat");
    trainedAgent->Write(saveFile);
  }

  // Evaluate a previously trained agent against a min-max agent.
  if (DO_EVALUATION) {
    std::ifstream saveFile("agent.dat");
    auto trainedAgent = learning::LearningAgent::Read(saveFile);

    MinMaxAgent minmaxAgent;
    learning::RandomAgent baselineAgent;
    Evaluator eval(1000);
    auto r = eval.Evaluate(trainedAgent.get(), &minmaxAgent);
    // auto r = eval.Evaluate(&minmaxAgent, &baselineAgent);
    std::cout << "r : " << r.first << " / " << r.second << std::endl;
  }

  return 0;
}
