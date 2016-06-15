
#include "Evaluator.hpp"
#include "Trainer.hpp"
#include "common/Common.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/MCTSAgent.hpp"
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

std::pair<float, float> evaluateAgent(learning::Agent *agent, learning::Agent *opponent) {
  Evaluator eval(1000);
  return eval.Evaluate(agent, opponent);
}

int main(int argc, char **argv) {
  srand(time(NULL));

  // Train an agent.
  if (DO_TRAINING) {
    Trainer trainer;
    trainer.AddProgressCallback([](learning::Agent *agent, unsigned iters) {
      if (iters % 5000 == 0) {
        learning::RandomAgent randomAgent;
        auto rar = evaluateAgent(agent, &randomAgent);
        std::cout << "random " << iters << "\t" << rar.first << std::endl;

        MinMaxAgent minmaxAgent1(1);
        auto mar1 = evaluateAgent(agent, &minmaxAgent1);
        std::cout << "minmax1 " << iters << "\t" << mar1.first << std::endl;

        MinMaxAgent minmaxAgent2(2);
        auto mar2 = evaluateAgent(agent, &minmaxAgent2);
        std::cout << "minmax2 " << iters << "\t" << mar2.first << std::endl;

        MinMaxAgent minmaxAgent3(3);
        auto mar3 = evaluateAgent(agent, &minmaxAgent3);
        std::cout << "minmax3 " << iters << "\t" << mar3.first << std::endl;
      }
    });

    auto trainedAgent = trainer.TrainAgent(3000000);
    std::ofstream saveFile("agent.dat");
    trainedAgent->Write(saveFile);
  }

  // Evaluate a previously trained agent against a min-max agent.
  if (DO_EVALUATION) {
    std::ifstream saveFile("agent.dat");
    auto trainedAgent = learning::LearningAgent::Read(saveFile);
    auto mctsAgent = make_unique<learning::MCTSAgent>(move(trainedAgent));

    MinMaxAgent minmaxAgent(4);
    learning::RandomAgent baselineAgent;
    Evaluator eval(100);
    auto r = eval.Evaluate(mctsAgent.get(), &minmaxAgent);
    // auto r = eval.Evaluate(&minmaxAgent, &baselineAgent);
    std::cout << "r : " << r.first << " / " << r.second << std::endl;
  }

  return 0;
}
