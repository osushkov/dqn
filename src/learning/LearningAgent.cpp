
#include "LearningAgent.hpp"
#include <cassert>
#include <random>

using namespace learning;

struct LearningAgent::LearningAgentImpl {
  LearningAgentImpl() {}

  GameAction SelectAction(const GameState *state) { return GameAction::ACTION(0); }

  void Learn(const vector<ExperienceMoment> &moments) {}
};

LearningAgent::LearningAgent() : impl(new LearningAgentImpl()) {}
LearningAgent::~LearningAgent() = default;

GameAction LearningAgent::SelectAction(const GameState *state) { return impl->SelectAction(state); }
void LearningAgent::Learn(const vector<ExperienceMoment> &moments) { impl->Learn(moments); }
