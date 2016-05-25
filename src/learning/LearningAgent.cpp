
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
    spec.numInputs = BOARD_WIDTH * BOARD_HEIGHT * 2;
    spec.numOutputs = GameAction::ALL_ACTIONS().size();
    spec.hiddenLayers = {spec.numInputs, spec.numInputs / 2, spec.numInputs / 2};
    spec.hiddenActivation = neuralnetwork::LayerActivation::LEAKY_RELU;
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
      return chooseExplorativeAction(*state);
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
      learnSamples.emplace_back(moment.initialState, moment.successorState,
                                GameAction::ACTION_INDEX(moment.actionTaken),
                                moment.isSuccessorTerminal, moment.reward, REWARD_DELAY_DISCOUNT);

      // float mq = maxQ(moment.successorState);
      //
      // float targetValue;
      // if (moment.isSuccessorTerminal) {
      //   targetValue = moment.reward;
      // } else {
      //   targetValue = moment.reward + REWARD_DELAY_DISCOUNT * mq;
      // }
      // learnSamples.emplace_back(moment.initialState, targetValue,
      //                           GameAction::ACTION_INDEX(moment.actionTaken));
    }

    learningNet->Update(neuralnetwork::SamplesProvider(learnSamples));
    itersSinceTargetUpdated++;
  }

  float GetQValue(const GameState &state, const GameAction &action) const {
    auto encodedState = LearningAgent::EncodeGameState(&state);
    EVector qvalues = learningNet->Process(encodedState);
    return qvalues(GameAction::ACTION_INDEX(action));
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
    // std::cout << "choosing best action: " << bestQValue << std::endl;
    return bestAction;
  }

  GameAction chooseExplorativeAction(const GameState &state) {
    // std::cout << "choosing random" << std::endl;
    auto aa = state.AvailableActions();
    return GameAction::ACTION(aa[rand() % aa.size()]);
  }
};

EVector LearningAgent::EncodeGameState(const GameState *state) {
  EVector result(2 * BOARD_WIDTH * BOARD_HEIGHT);
  result.fill(0.0f);

  for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
    for (unsigned c = 0; c < BOARD_WIDTH; c++) {
      unsigned ri = 2 * (c + r * BOARD_WIDTH);

      switch (state->GetCell(r, c)) {
      case CellState::MY_TOKEN:
        result(ri) = 1.0f;
        break;
      case CellState::OPPONENT_TOKEN:
        result(ri + 1) = 1.0f;
        break;
      default:
        break;
      }

      ri++;
    }
  }

  return result;
}

// EVector LearningAgent::EncodeGameState(const GameState *state) {
//   EVector result(BOARD_WIDTH * BOARD_HEIGHT * 2);
//   result.fill(0.0f)
//
//   for (unsigned r = 0; r < BOARD_HEIGHT; r++) {
//     for (unsigned c = 0; c < BOARD_WIDTH; c++) {
//       unsigned ri = 0;
//
//       switch (state->GetCell(r, c)) {
//       case CellState::EMPTY:
//         result(ri) = 0.0f;
//         break;
//       case CellState::MY_TOKEN:
//         result(ri) = 1.0f;
//         break;
//       case CellState::OPPONENT_TOKEN:
//         result(ri) = -1.0f;
//         break;
//       default:
//         assert(false);
//       }
//
//       ri++;
//     }
//   }
//
//   return result;
// }

LearningAgent::LearningAgent() : impl(new LearningAgentImpl()) {}
LearningAgent::~LearningAgent() = default;

GameAction LearningAgent::SelectAction(const GameState *state) { return impl->SelectAction(state); }

void LearningAgent::SetPRandom(float pRandom) { impl->SetPRandom(pRandom); }

GameAction LearningAgent::SelectLearningAction(const GameState *state,
                                               const EVector &encodedState) {
  return impl->SelectLearningAction(state, encodedState);
}

void LearningAgent::Learn(const vector<ExperienceMoment> &moments) { impl->Learn(moments); }

float LearningAgent::GetQValue(const GameState &state, const GameAction &action) const {
  return impl->GetQValue(state, action);
}
