
#include "MCTS.hpp"

#include "../common/Util.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameRules.hpp"
#include "../connectfour/GameState.hpp"
#include "Node.hpp"

using namespace mcts;

static const float P_RANDOM = 0.1f;
static const float PLAYOUT_TEMPERATURE = 0.001f;

struct MCTS::MCTSImpl {
  uptr<StateActionRater> rater;
  unsigned maxPlayoutDepth;

  MCTSImpl(uptr<StateActionRater> rater, unsigned maxPlayoutDepth)
      : rater(move(rater)), maxPlayoutDepth(maxPlayoutDepth) {}

  vector<ActionUtility> ComputeUtilities(const GameState &state, unsigned iterations) {
    Node root(state, 0);

    for (unsigned i = 0; i < iterations; i++) {
      mcIteration(&root);
    }

    vector<pair<unsigned, float>> utilities = root.GetActionUtilities();
    sort(utilities.begin(), utilities.end(),
         [](const pair<unsigned, float> &a0, const pair<unsigned, float> &a1) {
           return a0.second > a1.second;
         });

    vector<ActionUtility> result;
    for (const auto &u : utilities) {
      result.emplace_back(GameAction::ACTION(u.first), u.second);
    }
    return result;
  }

  // single iteration of monte-carlo tree search.
  void mcIteration(Node *root) {
    vector<Node *> pathFromRoot;

    Node *cur = root;
    while (!cur->IsLeaf()) {
      pathFromRoot.push_back(cur);
      cur = cur->Select(P_RANDOM);
    }
    pathFromRoot.push_back(cur);

    if (cur->IsTerminal()) {
      float utility = cur->ExpectedUtility(cur->PlayerIndex());
      feedBackUtility(utility, pathFromRoot);
      return;
    }

    Node *playoutNode = cur->Expand(rater.get());
    if (playoutNode == nullptr) {
      playoutNode = cur;
    } else {
      pathFromRoot.push_back(playoutNode);
    }

    float utility = playoutNode->IsTerminal()
                        ? playoutNode->ExpectedUtility(playoutNode->PlayerIndex())
                        : playout(playoutNode);
    feedBackUtility(utility, pathFromRoot);
  }

  float playout(Node *startNode) {
    GameRules *rules = GameRules::Instance();

    unsigned curPlayerIndex = startNode->PlayerIndex();
    GameState curState = *startNode->GetState();

    for (unsigned i = 0; i < maxPlayoutDepth; i++) {
      CompletionState completionState = rules->GameCompletionState(curState);
      if (completionState != CompletionState::UNFINISHED) {
        // Account for the fact that the winner may not be the player of the original startNode.
        // The result of this function should be the utility of the playout for the player owning
        // the startNode.
        float utilFlip = startNode->PlayerIndex() == curPlayerIndex ? 1.0f : -1.0f;

        switch (completionState) {
        case CompletionState::WIN:
          return 1.0f * utilFlip;
        case CompletionState::LOSS:
          return -1.0f * utilFlip;
        case CompletionState::DRAW:
          return 0.0f;
        default:
          assert(false);
          break;
        }
      }

      curState = randomSuccessor(curState);
      curPlayerIndex = 1 - curPlayerIndex;
    }

    float utilFlip = startNode->PlayerIndex() == curPlayerIndex ? 1.0f : -1.0f;
    switch (rules->GameCompletionState(curState)) {
    case CompletionState::WIN:
      return 1.0f * utilFlip;
    case CompletionState::LOSS:
      return -1.0f * utilFlip;
    case CompletionState::DRAW:
      return 0.0f;
    default:
      return rater->RateGameState(curState) * utilFlip;
    }
  }

  double playoutRandom(Node *startNode) {
    GameRules *rules = GameRules::Instance();

    unsigned curPlayerIndex = startNode->PlayerIndex();
    GameState curState = *startNode->GetState();

    while (true) {
      CompletionState completionState = rules->GameCompletionState(curState);
      if (completionState != CompletionState::UNFINISHED) {
        // Account for the fact that the winner may not be the player of the original startNode.
        // The result of this function should be the utility of the playout for the player owning
        // the startNode.
        float utilFlip = startNode->PlayerIndex() == curPlayerIndex ? 1.0f : -1.0f;

        switch (completionState) {
        case CompletionState::WIN:
          return 1.0f * utilFlip;
        case CompletionState::LOSS:
          return -1.0f * utilFlip;
        case CompletionState::DRAW:
          return 0.0f;
        default:
          assert(false);
          break;
        }
      }

      curState = randomSuccessor(curState);
      curPlayerIndex = 1 - curPlayerIndex;
    }

    assert(false);
    return 0.0f;
  }

  GameState randomSuccessor(const GameState &state) {
    vector<unsigned> actions = state.AvailableActions();
    assert(!actions.empty());

    GameAction action = GameAction::ACTION(actions[rand() % actions.size()]);
    GameState result = state.SuccessorState(action);
    result.FlipState();
    return result;
  }

  GameState chooseSuccessor(const GameState &state) {
    GameAction action = chooseAction(state);
    GameState successor = state.SuccessorState(action);
    successor.FlipState();
    return successor;
  }

  GameAction chooseAction(const GameState &state) {
    vector<unsigned> actions = state.AvailableActions();
    assert(actions.size() > 0);

    vector<float> weights = rater->RateAvailableActions(state);
    assert(weights.size() == actions.size());

    for (auto &w : weights) {
      w /= PLAYOUT_TEMPERATURE;
    }
    weights = Util::SoftmaxWeights(weights);

    float sample = Util::RandInterval(0.0, 1.0);
    for (unsigned i = 0; i < weights.size(); i++) {
      sample -= weights[i];
      if (sample <= 0.0f) {
        return GameAction::ACTION(actions[i]);
      }
    }

    return GameAction::ACTION(actions[rand() % actions.size()]);
  }

  void feedBackUtility(float utility, const vector<Node *> &pathFromRoot) {
    for (int i = pathFromRoot.size() - 1; i >= 0; i--) {
      pathFromRoot[i]->AddUtility(utility);
      utility = -utility;
    }
  }
};

MCTS::MCTS(uptr<StateActionRater> rater, unsigned maxPlayoutDepth)
    : impl(new MCTSImpl(move(rater), maxPlayoutDepth)) {}
MCTS::~MCTS() = default;

vector<ActionUtility> MCTS::ComputeUtilities(const GameState &state, unsigned iterations) {
  return impl->ComputeUtilities(state, iterations);
}
