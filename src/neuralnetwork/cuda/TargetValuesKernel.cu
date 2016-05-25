#include "TargetValuesKernel.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork::cuda;

__global__ void targetValuesKernel(LayerBatchOutputs outputs, SamplesBatch samplesBatch) {
  assert(gridDim.x == outputs.batchSize);
  const unsigned batchIndex = blockIdx.x;

  if (samplesBatch.isTerminal[batchIndex]) {
    samplesBatch.targetOutput[batchIndex] = samplesBatch.rewards[batchIndex];
  } else {
    float maxVal = *(outputs.OutputElem(batchIndex, 0));
    for (unsigned i = 1; i < outputs.layerSize - 1; i++) {
      maxVal = fmaxf(maxVal, *(outputs.OutputElem(batchIndex, i)));
    }

    samplesBatch.targetOutput[batchIndex] = samplesBatch.rewards[batchIndex] +
        samplesBatch.futureRewardDiscount * maxVal;
  }
}

void TargetValuesKernel::Apply(const LayerBatchOutputs &lastLayer, const SamplesBatch &samplesBatch,
                               cudaStream_t stream) {
  int tpb = 1;
  int bpg = lastLayer.batchSize;
  targetValuesKernel<<<bpg, tpb, 0, stream>>>(lastLayer, samplesBatch);
}
