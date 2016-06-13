#include "Node.hpp"
#include "../common/Util.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameRules.hpp"
#include <cassert>
#include <cmath>

using namespace mcts;

Node::Node(GameState state, unsigned playerIndex)
    : state(state), playerIndex(playerIndex), isLeaf(true), totalTrials(0), sumUtility(0.0) {

  GameRules *rules = GameRules::Instance();
  CompletionState cs = rules->GameCompletionState(state);

  isTerminal = cs != CompletionState::UNFINISHED;
  if (isTerminal) {
    totalTrials = 1;

    switch (cs) {
    case CompletionState::WIN:
      sumUtility = 1.0f;
      break;
    case CompletionState::LOSS:
      sumUtility = -1.0f;
      break;
    default:
      sumUtility = 0.0f;
      break;
    }
  }
}

bool Node::IsLeaf(void) const { return isLeaf; }
bool Node::IsTerminal(void) const { return isTerminal; }

unsigned Node::PlayerIndex(void) const { return playerIndex; }

const GameState *Node::GetState(void) const { return &state; }

vector<pair<unsigned, float>> Node::GetActionUtilities(void) const {
  vector<pair<unsigned, float>> result;
  result.reserve(children.size());

  for (const auto &edge : children) {
    result.emplace_back(edge.first, edge.second.ExpectedUtility(playerIndex));
  }
  return result;
}

Node *Node::Expand(StateActionRater *rater) {
  assert(isLeaf && !isTerminal);

  pair<vector<unsigned>, vector<float>> available = nonExpandedActions(rater);
  if (available.first.empty()) { // This can happen at a terminal game state.
    return nullptr;
  }

  if (available.first.size() == 1) {
    isLeaf = false;
  }

  for (auto &w : available.second) {
    w /= 0.1f;
  }
  available.second = Util::SoftmaxWeights(available.second);

  unsigned chosenAction = available.first[rand() % available.first.size()];
  float sample = Util::RandInterval(0.0, 1.0);
  for (unsigned i = 0; i < available.second.size(); i++) {
    sample -= available.second[i];
    if (sample <= 0.0f) {
      chosenAction = available.first[i];
      break;
    }
  }

  GameState childState = state.SuccessorState(GameAction::ACTION(chosenAction));
  childState.FlipState();

  children.emplace_back(chosenAction, Node(childState, 1 - playerIndex));
  return &children.back().second;
}

Node *Node::Select(float pRandom) {
  assert(!isLeaf && !isTerminal);
  assert(!children.empty());

  constexpr float UCB1_SCALE = 1.0;

  if (Util::RandInterval(0.0, 1.0) < pRandom) {
    return &children[rand() % children.size()].second;
  } else {
    Node *result = nullptr;
    double bestUtility = 0.0;

    for (auto &edge : children) {
      float utility =
          edge.second.ExpectedUtility(this->playerIndex) +
          UCB1_SCALE * sqrtf(log(totalTrials) / static_cast<float>(edge.second.totalTrials));

      if (result == nullptr || utility > bestUtility) {
        bestUtility = utility;
        result = &edge.second;
      }
    }

    assert(result != nullptr);
    return result;
  }
}

void Node::AddUtility(float utility) {
  totalTrials++;
  sumUtility += utility;
}

float Node::ExpectedUtility(unsigned playerIndex) const {
  assert(totalTrials > 0);

  float p = sumUtility / totalTrials;
  if (this->playerIndex == playerIndex) {
    return p;
  } else {
    return -p;
  }
}

// This is just a quick and dirty hack, can be more efficient but on small branch factor
// problems doesnt really make a difference.
pair<vector<unsigned>, vector<float>> Node::nonExpandedActions(StateActionRater *rater) {
  pair<vector<unsigned>, vector<float>> result;

  vector<float> weights = rater->RateAvailableActions(state);
  vector<unsigned> stateActions = state.AvailableActions();
  assert(weights.size() == stateActions.size());

  for (unsigned i = 0; i < stateActions.size(); i++) {
    unsigned sa = stateActions[i];

    bool shouldAdd = true;
    for (auto &c : children) {
      if (sa == c.first) {
        shouldAdd = false;
        break;
      }
    }

    if (shouldAdd) {
      result.first.push_back(sa);
      result.second.push_back(weights[i]);
    }
  }

  return result;
}
