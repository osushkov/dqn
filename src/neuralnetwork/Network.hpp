
#pragma once

#include "../common/Common.hpp"
#include "../math/Math.hpp"
#include "NetworkSpec.hpp"
#include "SamplesProvider.hpp"
#include <vector>

namespace neuralnetwork {

class Network {
public:
  Network(const NetworkSpec &spec);
  virtual ~Network();

  EVector Process(const EVector &input) const;

  // TODO: add a ProcessAsync that can output results while a training batch is running.
  // This will be more relevant for RL tasks where parallel processing and training makes sense.

  // TODO: this will probably need to have some kind of learning rate as a parameter.
  void Update(const SamplesProvider &samplesProvider);

  uptr<Network> RefreshAndGetTarget(void);

private:
  // Non-copyable
  Network() = default;
  Network(const Network &other) = delete;
  Network &operator=(const Network &) = delete;

  struct NetworkImpl;
  uptr<NetworkImpl> impl;
};
}
