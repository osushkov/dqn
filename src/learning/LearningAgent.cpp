
#include "LearningAgent.hpp"
#include "../common/Common.hpp"
#include "../common/Util.hpp"
#include "../neuralnetwork/Network.hpp"
#include "../neuralnetwork/NetworkSpec.hpp"
#include "Constants.hpp"

#include <cassert>
#include <random>

using namespace learning;

struct LearningAgent::LearningAgentImpl {
  float pRandom;
  uptr<neuralnetwork::Network> learningNet;
  uptr<neuralnetwork::Network> targetNet;
  unsigned itersSinceTargetUpdated = 0;

  LearningAgentImpl() : pRandom(0.0f) {
    neuralnetwork::NetworkSpec spec;
    spec.numInputs = BOARD_WIDTH * BOARD_HEIGHT;
    spec.numOutputs = GameAction::ALL_ACTIONS().size();
    spec.hiddenLayers = {spec.numInputs, spec.numInputs / 2};
    spec.hiddenActivation = neuralnetwork::LayerActivation::TANH;
    spec.outputActivation = neuralnetwork::LayerActivation::TANH;
    spec.maxBatchSize = MOMENTS_BATCH_SIZE;

    learningNet = make_unique<neuralnetwork::Network>(spec);
    targetNet = learningNet->ReadOnlyCopy();
    itersSinceTargetUpdated = 0;
  }

  GameAction SelectAction(const GameState *state) {
    assert(state != nullptr);
    return chooseBestAction(LearningAgent::EncodeGameState(state));
  }

  void SetPRandom(float pRandom) {
    assert(pRandom >= 0.0f && pRandom <= 1.0f);
    this->pRandom = pRandom;
  }

  GameAction SelectLearningAction(const GameState *state, const EVector &encodedState) {
    assert(state != nullptr);
    if (Util::RandInterval(0.0, 1.0) < pRandom) {
      return chooseExplorativeAction(encodedState);
    } else {
      return chooseBestAction(encodedState);
    }
  }

  void Learn(const vector<ExperienceMoment> &moments) {
    if (itersSinceTargetUpdated > TARGET_FUNCTION_UPDATE_RATE) {
      learningNet->Refresh();
      targetNet = learningNet->ReadOnlyCopy();
      itersSinceTargetUpdated = 0;
    }

    vector<neuralnetwork::TrainingSample> learnSamples;
    learnSamples.reserve(moments.size());

    for (const auto &moment : moments) {
      float mq = maxQ(moment.successorState);
      float targetValue = moment.reward + REWARD_DELAY_DISCOUNT * mq;
    }

    learningNet->Update(neuralnetwork::SamplesProvider(learnSamples));
  }

  float maxQ(const EVector &encodedState) const {
    EVector qa = targetNet->Process(encodedState);

    float maxVal = qa(0);
    for (int i = 1; i < qa.rows(); i++) {
      maxVal = max(maxVal, qa(i));
    }
    return maxVal;
  }

  GameAction chooseBestAction(const EVector &encodedState) {
    EVector qvalues = learningNet->Process(encodedState);
    assert(qvalues.rows() == static_cast<int>(GameAction::ALL_ACTIONS().size()));

    GameAction bestAction = GameAction::ACTION(0);
    float bestQValue = qvalues(0);

    for (int i = 1; i < qvalues.rows(); i++) {
      if (qvalues(i) > bestQValue) {
        bestQValue = qvalues(i);
        bestAction = GameAction::ACTION(i);
      }
    }

    return bestAction;
  }

  GameAction chooseExplorativeAction(const EVector &encodedState) {
    const auto &actions = GameAction::ALL_ACTIONS();
    return actions[rand() % actions.size()];
  }
};

EVector LearningAgent::EncodeGameState(const GameState *state) {
  EVector result(BOARD_WIDTH * BOARD_HEIGHT);

  unsigned ri = 0;
  for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH; c++) {
      switch (state->GetCell(r, c)) {
      case CellState::EMPTY:
        result(ri) = 0.0f;
        break;
      case CellState::MY_TOKEN:
        result(ri) = 1.0f;
        break;
      case CellState::OPPONENT_TOKEN:
        result(ri) = -1.0f;
        break;
      default:
        assert(false);
      }

      ri++;
    }
  }

  return result;
}

LearningAgent::LearningAgent() : impl(new LearningAgentImpl()) {}
LearningAgent::~LearningAgent() = default;

GameAction LearningAgent::SelectAction(const GameState *state) { return impl->SelectAction(state); }

void LearningAgent::SetPRandom(float pRandom) { impl->SetPRandom(pRandom); }

GameAction LearningAgent::SelectLearningAction(const GameState *state,
                                               const EVector &encodedState) {
  return impl->SelectLearningAction(state, encodedState);
}

void LearningAgent::Learn(const vector<ExperienceMoment> &moments) { impl->Learn(moments); }
