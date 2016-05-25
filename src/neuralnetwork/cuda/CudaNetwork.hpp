
#pragma once

#include "../../math/MatrixView.hpp"
#include "../NetworkSpec.hpp"
#include <memory>
#include <vector>

namespace neuralnetwork {
namespace cuda {

struct QBatch {
  math::MatrixView initialStates;
  math::MatrixView successorStates;
  std::vector<unsigned> actionsTaken;
  std::vector<char> isEndStateTerminal;
  std::vector<float> rewardsGained;

  float futureRewardDiscount;
};

class CudaNetwork {
public:
  CudaNetwork(const NetworkSpec &spec);
  ~CudaNetwork();

  void SetWeights(const std::vector<math::MatrixView> &weights);
  void GetWeights(std::vector<math::MatrixView> &outWeights);

  void UpdateTarget(void);
  void Train(const QBatch &qbatch);

  // math::MatrixView &batchInputs, const std::vector<float> &targetOutputs,
  //            const std::vector<unsigned> &targetOutputIndices);

private:
  struct CudaNetworkImpl;
  std::unique_ptr<CudaNetworkImpl> impl;
};
}
}
