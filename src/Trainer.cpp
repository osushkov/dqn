
#include "Trainer.hpp"
#include "common/Common.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/Constants.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"

#include <cassert>
#include <cmath>
#include <vector>

using namespace learning;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 10000;
static constexpr float INITIAL_PRANDOM = 0.8f;
static constexpr float TARGET_PRANDOM = 0.025f;

struct Trainer::TrainerImpl {
  vector<ProgressCallback> callbacks;

  void AddProgressCallback(ProgressCallback callback) { callbacks.push_back(callback); }

  uptr<Agent> TrainAgent(unsigned iters) {
    auto experienceMemory = make_unique<ExperienceMemory>(EXPERIENCE_MEMORY_SIZE);
    auto agent = make_unique<LearningAgent>();

    float pRandom = INITIAL_PRANDOM;
    float decay = powf(TARGET_PRANDOM / INITIAL_PRANDOM, 1.0f / iters);
    assert(decay > 0.0f && decay < 1.0f);

    for (unsigned i = 0; i < iters; i++) {
      agent->SetPRandom(pRandom);
      playoutRound(agent.get(), experienceMemory.get());

      // for (unsigned j = 0; j < 2; j++) {
      learn(agent.get(), experienceMemory.get());
      // }
      pRandom *= decay;
    }

    return move(agent);
  }

  void playoutRound(LearningAgent *agent, ExperienceMemory *memory) {
    RandomAgent opponent;

    GameRules *rules = GameRules::Instance();
    GameState curState(rules->InitialState());
    unsigned curPlayerIndex = rand() % 2;

    vector<EVector> stateHistory;
    vector<GameAction> actionHistory;

    while (true) {
      assert(stateHistory.size() == actionHistory.size());

      GameAction action;
      EVector encodedState = LearningAgent::EncodeGameState(&curState);
      if (curPlayerIndex == 0) {

        action = agent->SelectLearningAction(&curState, encodedState);

        if (stateHistory.size() > 0) {
          memory->AddExperience(ExperienceMoment(stateHistory[stateHistory.size() - 1],
                                                 actionHistory[actionHistory.size() - 1],
                                                 encodedState, 0.0f, false));
        }

        stateHistory.push_back(encodedState);
        actionHistory.push_back(action);
      } else {
        action = opponent.SelectAction(&curState);
      }

      curState = curState.SuccessorState(action);

      switch (rules->GameCompletionState(curState)) {
      case CompletionState::WIN:
        if (curPlayerIndex == 0) {
          memory->AddExperience(ExperienceMoment(encodedState, action, encodedState, 1.0f, true));
        } else {
          memory->AddExperience(ExperienceMoment(stateHistory[stateHistory.size() - 1],
                                                 actionHistory[actionHistory.size() - 1],
                                                 encodedState, -1.0f, true));
        }
        return;
      case CompletionState::LOSS:
        assert(false); // This actually shouldn't be possible.
        return;
      case CompletionState::DRAW:
        if (curPlayerIndex == 0) {
          memory->AddExperience(ExperienceMoment(encodedState, action, encodedState, 0.0f, true));
        } else {
          memory->AddExperience(ExperienceMoment(stateHistory[stateHistory.size() - 1],
                                                 actionHistory[actionHistory.size() - 1],
                                                 encodedState, 0.0f, true));
        }
        return;
      case CompletionState::UNFINISHED:
        curState.FlipState();
        curPlayerIndex = (curPlayerIndex + 1) % 2;
      }
    }
  }

  void learn(LearningAgent *agent, ExperienceMemory *memory) {
    agent->Learn(memory->Sample(MOMENTS_BATCH_SIZE));
  }
};

Trainer::Trainer() : impl(new TrainerImpl()) {}
Trainer::~Trainer() = default;

void Trainer::AddProgressCallback(ProgressCallback callback) {
  impl->AddProgressCallback(callback);
}

uptr<Agent> Trainer::TrainAgent(unsigned iters) { return impl->TrainAgent(iters); }
