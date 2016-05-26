
#include "Memory.hpp"
#include "Util.hpp"
#include <cuda_runtime.h>

using namespace neuralnetwork::cuda;

void *memory::AllocPushBuffer(size_t bufSize) {
  void* result = nullptr;

  cudaError_t err = cudaHostAlloc(&result, bufSize, cudaHostAllocWriteCombined);
  CheckError(err);
  assert(result != nullptr);

  return result;
}

void memory::FreePushBuffer(void *buf) {
  assert(buf != nullptr);
  cudaError_t err = cudaFreeHost(buf);
  CheckError(err);
}
