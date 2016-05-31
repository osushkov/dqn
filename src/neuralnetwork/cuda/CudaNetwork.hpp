
#pragma once

#include "../../math/MatrixView.hpp"
#include "../NetworkSpec.hpp"
#include <memory>
#include <vector>

namespace neuralnetwork {
namespace cuda {

struct QBatch {
  unsigned batchSize;

  math::MatrixView initialStates;
  math::MatrixView successorStates;

  unsigned *actionsTaken;
  char *isEndStateTerminal;
  float *rewardsGained;

  float futureRewardDiscount;
};

class CudaNetwork {
public:
  CudaNetwork(const NetworkSpec &spec);
  ~CudaNetwork();

  void SetWeights(const std::vector<math::MatrixView> &weights);
  void GetWeights(std::vector<math::MatrixView> &outWeights);

  void UpdateTarget(void);
  void Train(const QBatch &qbatch, float learnRate);

  // math::MatrixView &batchInputs, const std::vector<float> &targetOutputs,
  //            const std::vector<unsigned> &targetOutputIndices);

private:
  struct CudaNetworkImpl;
  std::unique_ptr<CudaNetworkImpl> impl;
};
}
}
