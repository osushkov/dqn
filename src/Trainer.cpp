
#include "Trainer.hpp"
#include "common/Common.hpp"
#include "common/Timer.hpp"
#include "connectfour/GameAction.hpp"
#include "connectfour/GameRules.hpp"
#include "connectfour/GameState.hpp"
#include "learning/Constants.hpp"
#include "learning/ExperienceMemory.hpp"
#include "learning/LearningAgent.hpp"
#include "learning/RandomAgent.hpp"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cmath>
#include <future>
#include <thread>
#include <vector>

using namespace learning;

static constexpr unsigned EXPERIENCE_MEMORY_SIZE = 100000;
static constexpr float INITIAL_PRANDOM = 0.8f;
static constexpr float TARGET_PRANDOM = 0.05f;

struct PlayoutAgent {
  LearningAgent *agent;
  ExperienceMemory *memory;

  vector<EVector> stateHistory;
  vector<GameAction> actionHistory;

  PlayoutAgent(LearningAgent *agent, ExperienceMemory *memory) : agent(agent), memory(memory) {}

  bool havePreviousState(void) { return stateHistory.size() > 0; }

  void addTransitionToMemory(EVector &curState, float reward, bool isTerminal) {
    EVector &prevState = stateHistory[stateHistory.size() - 1];
    GameAction &performedAction = actionHistory[actionHistory.size() - 1];

    memory->AddExperience(
        ExperienceMoment(prevState, performedAction, curState, reward, isTerminal));
  }

  void addMoveToHistory(const EVector &state, const GameAction &action) {
    stateHistory.push_back(state);
    actionHistory.push_back(action);
  }
};

struct Trainer::TrainerImpl {
  vector<ProgressCallback> callbacks;
  atomic<unsigned> numLearnIters;

  void AddProgressCallback(ProgressCallback callback) { callbacks.push_back(callback); }

  uptr<Agent> TrainAgent(unsigned iters) {
    auto experienceMemory = make_unique<ExperienceMemory>(EXPERIENCE_MEMORY_SIZE);
    auto agent = make_unique<LearningAgent>();

    numLearnIters = 0;
    std::thread playoutThread([this, iters, &experienceMemory, &agent]() {
      float pRandom = INITIAL_PRANDOM;
      float pRandDecay = powf(TARGET_PRANDOM / INITIAL_PRANDOM, 1.0f / iters);
      assert(pRandDecay > 0.0f && pRandDecay < 1.0f);

      while (numLearnIters.load() < iters) {
        float prand = INITIAL_PRANDOM * powf(pRandDecay, numLearnIters.load());
        agent->SetPRandom(prand);
        this->playoutRound(agent.get(), experienceMemory.get());
      }
    });

    std::thread learnThread([this, iters, &experienceMemory, &agent]() {
      while (experienceMemory->NumMemories() < 2 * MOMENTS_BATCH_SIZE) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
      }

      Timer timer;
      timer.Start();

      for (unsigned i = 0; i < iters; i++) {
        agent->Learn(experienceMemory->Sample(MOMENTS_BATCH_SIZE));
        this->numLearnIters++;
      }

      timer.Stop();
      std::cout << "learn iters per second: " << (iters / timer.GetNumElapsedSeconds())
                << std::endl;
    });

    playoutThread.join();
    learnThread.join();

    return move(agent);
  }

  // void playoutRound(LearningAgent *agent, ExperienceMemory *memory) {
  //   GameRules *rules = GameRules::Instance();
  //   GameState curState(rules->InitialState());
  //
  //   std::vector<PlayoutAgent> opponents = {PlayoutAgent(agent, memory),
  //                                          PlayoutAgent(agent, memory)};
  //   unsigned curPlayerIndex = rand() % 2;
  //
  //   while (true) {
  //     PlayoutAgent &curPlayer = opponents[curPlayerIndex];
  //     PlayoutAgent &otherPlayer = opponents[(curPlayerIndex + 1) % opponents.size()];
  //
  //     EVector encodedState = LearningAgent::EncodeGameState(&curState);
  //     GameAction action = curPlayer.agent->SelectLearningAction(&curState, encodedState);
  //
  //     if (curPlayer.havePreviousState()) {
  //       curPlayer.addTransitionToMemory(encodedState, 0.0f, false);
  //     }
  //
  //     curPlayer.addMoveToHistory(encodedState, action);
  //     curState = curState.SuccessorState(action);
  //
  //     switch (rules->GameCompletionState(curState)) {
  //     case CompletionState::WIN:
  //       encodedState = LearningAgent::EncodeGameState(&curState);
  //       curPlayer.addTransitionToMemory(encodedState, 1.0f, true);
  //       otherPlayer.addTransitionToMemory(encodedState, -1.0f, true);
  //
  //       return;
  //     case CompletionState::LOSS:
  //       assert(false); // This actually shouldn't be possible.
  //       return;
  //     case CompletionState::DRAW:
  //       encodedState = LearningAgent::EncodeGameState(&curState);
  //       curPlayer.addTransitionToMemory(encodedState, 0.0f, true);
  //       otherPlayer.addTransitionToMemory(encodedState, 0.0f, true);
  //       return;
  //     case CompletionState::UNFINISHED:
  //       curState.FlipState();
  //       curPlayerIndex = (curPlayerIndex + 1) % 2;
  //     }
  //   }
  // }

  void playoutRound(LearningAgent *agent, ExperienceMemory *memory) {
    GameRules *rules = GameRules::Instance();
    GameState curState(rules->InitialState());

    RandomAgent opponent;
    PlayoutAgent pagent(agent, memory);

    unsigned curPlayerIndex = rand() % 2;

    while (true) {

      EVector encodedState;
      GameAction action;
      if (curPlayerIndex == 0) {
        encodedState = LearningAgent::EncodeGameState(&curState);
        action = pagent.agent->SelectLearningAction(&curState, encodedState);

        if (pagent.havePreviousState()) {
          pagent.addTransitionToMemory(encodedState, 0.0f, false);
        }

        pagent.addMoveToHistory(encodedState, action);
      } else {
        action = opponent.SelectAction(&curState);
      }
      curState = curState.SuccessorState(action);

      switch (rules->GameCompletionState(curState)) {
      case CompletionState::WIN:
        encodedState = LearningAgent::EncodeGameState(&curState);
        pagent.addTransitionToMemory(encodedState, curPlayerIndex == 0 ? 1.0f : -1.0f, true);

        return;
      case CompletionState::LOSS:
        assert(false); // This actually shouldn't be possible.
        return;
      case CompletionState::DRAW:
        encodedState = LearningAgent::EncodeGameState(&curState);
        pagent.addTransitionToMemory(encodedState, 0.0f, true);
        return;
      case CompletionState::UNFINISHED:
        curState.FlipState();
        curPlayerIndex = (curPlayerIndex + 1) % 2;
      }
    }
  }
};

Trainer::Trainer() : impl(new TrainerImpl()) {}
Trainer::~Trainer() = default;

void Trainer::AddProgressCallback(ProgressCallback callback) {
  impl->AddProgressCallback(callback);
}

uptr<Agent> Trainer::TrainAgent(unsigned iters) { return impl->TrainAgent(iters); }
