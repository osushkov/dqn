#pragma once

#include "common/Common.hpp"
#include "learning/Agent.hpp"
#include <functional>

using ProgressCallback = function<void(learning::Agent *, unsigned)>;

class Trainer {
public:
  Trainer();
  ~Trainer();

  void AddProgressCallback(ProgressCallback callback);
  uptr<learning::Agent> TrainAgent(unsigned iters);

private:
  struct TrainerImpl;
  uptr<TrainerImpl> impl;
};
