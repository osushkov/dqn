
#include "Util.hpp"
#include <iostream>
#include <cassert>

using namespace neuralnetwork::cuda;

void util::OutputError(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "GPU error: " << cudaGetErrorString(code) << " "
        << file << "(" << line << ")" << std::endl;
    exit(code);
  }
}

void *util::AllocPushBuffer(size_t bufSize) {
  void* result = nullptr;

  cudaError_t err = cudaHostAlloc(&result, bufSize, cudaHostAllocWriteCombined);
  CheckError(err);
  assert(result != nullptr);

  return result;
}

void util::FreePushBuffer(void *buf) {
  assert(buf != nullptr);
  cudaError_t err = cudaFreeHost(buf);
  CheckError(err);
}

LayerWeights util::NewLayerWeights(unsigned inputSize, unsigned layerSize) {
  assert(inputSize > 0 && layerSize > 0);

  LayerWeights result;
  result.inputSize = inputSize;
  result.layerSize = layerSize;

  size_t width = inputSize * sizeof(float);
  size_t height = layerSize;

  // cudaError_t err = cudaMalloc(&result.weights, width * height);
  // result.pitch = width;
  cudaError_t err = cudaMallocPitch(&(result.weights), &(result.pitch), width, height);
  CheckError(err);

  return result;
}

void util::DeleteLayerWeights(LayerWeights &lw) {
  cudaError_t err = cudaFree(lw.weights);
  CheckError(err);
  lw.weights = nullptr;
}

SamplesBatch util::NewSamplesBatch(unsigned maxBatchSize, unsigned inputDim,
                                   unsigned targetOutputDim) {
  assert(maxBatchSize > 0 && inputDim > 0 && targetOutputDim > 0);

  SamplesBatch result;
  result.maxBatchSize = maxBatchSize;
  result.batchSize = 0;
  result.inputDim = inputDim;
  result.targetOutputDim = targetOutputDim;

  size_t width = inputDim * sizeof(float);
  size_t height = maxBatchSize;

  cudaError_t err = cudaMallocPitch(&result.input, &result.ipitch, width, height);
  CheckError(err);

  width = targetOutputDim * sizeof(float);
  err = cudaMallocPitch(&result.targetOutput, &result.opitch, width, height);
  CheckError(err);

  return result;
}

void util::DeleteSamplesBatch(SamplesBatch &sb) {
  cudaError_t err = cudaFree(sb.input);
  CheckError(err);
  sb.input = nullptr;

  err = cudaFree(sb.targetOutput);
  CheckError(err);
  sb.targetOutput = nullptr;
}

LayerBatchOutputs util::NewLayerBatchOutputs(unsigned maxBatchSize, unsigned layerSize) {
  assert(maxBatchSize > 0 && layerSize > 0);

  LayerBatchOutputs result;
  result.maxBatchSize = maxBatchSize;
  result.batchSize = 0;
  result.layerSize = layerSize;

  size_t width = layerSize * sizeof(float);
  size_t height = maxBatchSize;

  cudaError_t err = cudaMallocPitch(&result.output, &result.opitch, width, height);
  CheckError(err);

  err = cudaMallocPitch(&result.derivative, &result.dpitch, width, height);
  CheckError(err);

  return result;
}

void util::DeleteLayerBatchOutputs(LayerBatchOutputs &lbo) {
  cudaError_t err = cudaFree(lbo.output);
  CheckError(err);
  lbo.output = nullptr;

  err = cudaFree(lbo.derivative);
  CheckError(err);
  lbo.derivative = nullptr;
}

LayerBatchDeltas util::NewLayerBatchDeltas(unsigned maxBatchSize, unsigned layerSize) {
  assert(maxBatchSize > 0 && layerSize > 0);

  LayerBatchDeltas result;
  result.maxBatchSize = maxBatchSize;
  result.batchSize = 0;
  result.layerSize = layerSize;

  size_t width = layerSize * sizeof(float);
  size_t height = maxBatchSize;

  cudaError_t err = cudaMallocPitch(&result.delta, &result.pitch, width, height);
  CheckError(err);

  return result;
}

void util::DeleteLayerBatchDeltas(LayerBatchDeltas &lbd) {
  cudaError_t err = cudaFree(lbd.delta);
  CheckError(err);
  lbd.delta = nullptr;
}
