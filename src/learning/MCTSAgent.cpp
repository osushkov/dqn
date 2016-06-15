
#include "MCTSAgent.hpp"
#include "../mcts/MCTS.hpp"

using namespace learning;

static constexpr unsigned MAX_PLAYOUT_DEPTH = 0;
static constexpr unsigned NUM_ITERATIONS = 20000;

struct MCTSAgent::MCTSAgentImpl {
  uptr<mcts::MCTS> mcts;

  MCTSAgentImpl(uptr<mcts::StateActionRater> rater)
      : mcts(new mcts::MCTS(move(rater), MAX_PLAYOUT_DEPTH)) {}

  GameAction SelectAction(const GameState *state) {
    vector<mcts::ActionUtility> utilities = mcts->ComputeUtilities(*state, NUM_ITERATIONS);
    return utilities[0].first;
  }
};

MCTSAgent::MCTSAgent(uptr<mcts::StateActionRater> rater) : impl(new MCTSAgentImpl(move(rater))) {}
MCTSAgent::~MCTSAgent() = default;

GameAction MCTSAgent::SelectAction(const GameState *state) { return impl->SelectAction(state); }
