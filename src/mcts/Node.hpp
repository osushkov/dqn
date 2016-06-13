#pragma once

#include "../common/Common.hpp"
#include "../connectfour/GameAction.hpp"
#include "../connectfour/GameState.hpp"
#include "StateActionRater.hpp"

using namespace connectfour;

namespace mcts {

class Node {
public:
  using Edge = pair<unsigned, Node>;

  Node(GameState state, unsigned playerIndex);
  virtual ~Node() = default;

  bool IsLeaf(void) const;
  bool IsTerminal(void) const;

  unsigned PlayerIndex(void) const;
  const GameState *GetState(void) const;

  vector<pair<unsigned, float>> GetActionUtilities(void) const;

  Node *Expand(StateActionRater *rater); // Can only be done on a 'leaf'

  // TODO: should add a policy object as input that will choose which edge to select.
  // For now it's simply e-greedy.
  Node *Select(float pRandom); // Should only be done on non-leaves.

  void AddUtility(float utility);

  // From this node, what is the probability the given player will win.
  float ExpectedUtility(unsigned playerIndex) const;

private:
  GameState state;
  vector<Edge> children;

  // This should be an enum maybe. This signifies which players turn it is for this node,
  // since this is an adversarial game, it is not always "my" turn, where "me" is defined
  // as the player at the root of the tree.
  unsigned playerIndex; // index 0 is "me", index 1 is opponent.
  bool isLeaf;
  bool isTerminal;

  unsigned totalTrials;
  float sumUtility;

  pair<vector<unsigned>, vector<float>> nonExpandedActions(StateActionRater *rater);
};
}
