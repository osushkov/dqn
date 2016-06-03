#pragma once

#include "common/Common.hpp"
#include "learning/LearningAgent.hpp"
#include <functional>

using ProgressCallback = function<void(learning::Agent *, unsigned)>;

class Trainer {
public:
  Trainer();
  ~Trainer();

  void AddProgressCallback(ProgressCallback callback);
  uptr<learning::LearningAgent> TrainAgent(unsigned iters);

private:
  struct TrainerImpl;
  uptr<TrainerImpl> impl;
};
