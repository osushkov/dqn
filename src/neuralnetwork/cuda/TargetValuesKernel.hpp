#pragma once

#include "Types.hpp"

namespace neuralnetwork {
namespace cuda {
namespace TargetValuesKernel {

void Apply(const LayerBatchOutputs &lastLayer, const SamplesBatch &samplesBatch,
           cudaStream_t stream);
}
}
}
