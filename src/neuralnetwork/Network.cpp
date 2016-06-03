
#include "Network.hpp"
#include "../common/Common.hpp"
#include "../common/Util.hpp"
#include "../math/Math.hpp"
#include "../math/Tensor.hpp"
#include "Activations.hpp"
#include "cuda/CudaNetwork.hpp"
#include "cuda/Memory.hpp"

#include <atomic>
#include <boost/thread/shared_mutex.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <mutex>

using namespace neuralnetwork;

struct Network::NetworkImpl {
  mutable boost::shared_mutex rwMutex;
  mutable std::mutex trainMutex;

  NetworkSpec spec;
  math::Tensor layerWeights;
  uptr<cuda::CudaNetwork> cudaNetwork;

  unsigned inputBatchIndex;
  std::vector<cuda::QBatch> inputBatches;

  NetworkImpl(const NetworkSpec &spec, bool isTrainable) : spec(spec) {
    assert(spec.numInputs > 0 && spec.numOutputs > 0);
    initialiseWeights();

    if (isTrainable) {
      initialiseCuda();
      allocateInputBatches();
      inputBatchIndex = 0;
    } else {
      cudaNetwork = nullptr;
    }
  }

  ~NetworkImpl() { freeInputBatches(); }

  EVector Process(const EVector &input) const {
    assert(input.rows() == spec.numInputs);

    // obtain a read lock
    boost::shared_lock<boost::shared_mutex> lock(rwMutex);

    EVector layerOutput = input;
    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      LayerActivation func =
          (i == layerWeights.NumLayers() - 1) ? spec.outputActivation : spec.hiddenActivation;
      layerOutput = getLayerOutput(layerOutput, layerWeights(i), func);
    }

    return layerOutput;
  }

  void Update(const SamplesProvider &samplesProvider, float learnRate) {
    assert(cudaNetwork != nullptr);
    assert(samplesProvider.NumSamples() <= spec.maxBatchSize);

    unsigned curInputBatch = (inputBatchIndex++) % 2;

    // std::cout << "curInputBatch: " << (int)curInputBatch << std::endl;
    inputBatches[curInputBatch].batchSize = samplesProvider.NumSamples();
    inputBatches[curInputBatch].initialStates.rows = samplesProvider.NumSamples();
    inputBatches[curInputBatch].successorStates.rows = samplesProvider.NumSamples();

    for (unsigned i = 0; i < samplesProvider.NumSamples(); i++) {
      const TrainingSample &sample = samplesProvider[i];

      assert(sample.startState.cols() == 1 && sample.startState.rows() == spec.numInputs);
      assert(sample.endState.cols() == 1 && sample.endState.rows() == spec.numInputs);
      assert(sample.actionTaken < spec.numOutputs);

      for (unsigned j = 0; j < sample.startState.rows(); j++) {
        unsigned cols = inputBatches[curInputBatch].initialStates.cols;
        inputBatches[curInputBatch].initialStates.data[j + i * cols] = sample.startState(j);
        inputBatches[curInputBatch].successorStates.data[j + i * cols] = sample.endState(j);
      }

      inputBatches[curInputBatch].actionsTaken[i] = sample.actionTaken;
      inputBatches[curInputBatch].isEndStateTerminal[i] =
          static_cast<char>(sample.isEndStateTerminal);
      inputBatches[curInputBatch].rewardsGained[i] = sample.rewardGained;
      inputBatches[curInputBatch].futureRewardDiscount = sample.futureRewardDiscount;
    }

    cudaNetwork->Train(inputBatches[curInputBatch], learnRate);
  }

  uptr<NetworkImpl> RefreshAndGetTarget(void) {
    assert(cudaNetwork != nullptr);

    // obtain a write lock
    boost::unique_lock<boost::shared_mutex> lock(rwMutex);

    vector<math::MatrixView> weightViews;
    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      weightViews.push_back(math::GetMatrixView(layerWeights(i)));
    }
    cudaNetwork->UpdateTarget();
    cudaNetwork->GetWeights(weightViews);

    auto result = make_unique<NetworkImpl>(spec, false);
    result->layerWeights = layerWeights;
    return move(result);
  }

  EVector getLayerOutput(const EVector &prevLayer, const EMatrix &layerWeights,
                         LayerActivation afunc) const {
    assert(prevLayer.rows() == layerWeights.cols() - 1);

    EVector z = layerWeights * getInputWithBias(prevLayer);
    for (unsigned i = 0; i < z.rows(); i++) {
      z(i) = ActivationValue(afunc, z(i));
    }

    return z;
  }

  EVector getInputWithBias(const EVector &noBiasInput) const {
    EVector result(noBiasInput.rows() + 1);
    result(noBiasInput.rows()) = 1.0f;
    result.topRightCorner(noBiasInput.rows(), 1) = noBiasInput;
    return result;
  }

  void initialiseWeights(void) {
    if (spec.hiddenLayers.empty()) {
      layerWeights.AddLayer(createLayer(spec.numInputs, spec.numOutputs));
    } else {
      layerWeights.AddLayer(createLayer(spec.numInputs, spec.hiddenLayers[0]));

      for (unsigned i = 0; i < spec.hiddenLayers.size() - 1; i++) {
        layerWeights.AddLayer(createLayer(spec.hiddenLayers[i], spec.hiddenLayers[i + 1]));
      }

      layerWeights.AddLayer(
          createLayer(spec.hiddenLayers[spec.hiddenLayers.size() - 1], spec.numOutputs));
    }
  }

  EMatrix createLayer(unsigned inputSize, unsigned layerSize) const {
    assert(inputSize > 0 && layerSize > 0);

    unsigned numRows = layerSize;
    unsigned numCols = inputSize + 1; // +1 accounts for bias input
    float initRange = 1.0f / sqrtf(numCols);

    EMatrix result(numRows, numCols);
    for (unsigned r = 0; r < result.rows(); r++) {
      for (unsigned c = 0; c < result.cols(); c++) {
        result(r, c) = Util::RandInterval(-initRange, initRange);
      }
    }

    return result;
  }

  void initialiseCuda(void) {
    cudaNetwork = make_unique<cuda::CudaNetwork>(spec);

    vector<math::MatrixView> weights;
    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      weights.push_back(math::GetMatrixView(layerWeights(i)));
    }

    cudaNetwork->SetWeights(weights);
  }

  void allocateInputBatches(void) {
    for (unsigned i = 0; i < 2; i++) {
      cuda::QBatch batch;
      batch.actionsTaken =
          (unsigned *)cuda::memory::AllocPushBuffer(spec.maxBatchSize * sizeof(unsigned));
      batch.isEndStateTerminal =
          (char *)cuda::memory::AllocPushBuffer(spec.maxBatchSize * sizeof(char));
      batch.rewardsGained =
          (float *)cuda::memory::AllocPushBuffer(spec.maxBatchSize * sizeof(float));

      size_t mbufSize = spec.maxBatchSize * spec.numInputs * sizeof(float);

      batch.initialStates.rows = spec.maxBatchSize;
      batch.initialStates.cols = spec.numInputs;
      batch.initialStates.data = (float *)cuda::memory::AllocPushBuffer(mbufSize);

      batch.successorStates.rows = spec.maxBatchSize;
      batch.successorStates.cols = spec.numInputs;
      batch.successorStates.data = (float *)cuda::memory::AllocPushBuffer(mbufSize);

      inputBatches.push_back(batch);
    }
  }

  void freeInputBatches(void) {
    for (const auto &batch : inputBatches) {
      cuda::memory::FreePushBuffer(batch.actionsTaken);
      cuda::memory::FreePushBuffer(batch.isEndStateTerminal);
      cuda::memory::FreePushBuffer(batch.rewardsGained);

      cuda::memory::FreePushBuffer(batch.initialStates.data);
      cuda::memory::FreePushBuffer(batch.successorStates.data);
    }
  }
};

Network::Network(const NetworkSpec &spec) : impl(new NetworkImpl(spec, true)) {}
Network::~Network() = default;

uptr<Network> Network::Read(std::istream &in) {
  NetworkSpec spec = NetworkSpec::Read(in);

  uptr<Network> result = make_unique<Network>(spec);
  result->impl = make_unique<NetworkImpl>(spec, false);
  result->impl->layerWeights = math::Tensor::Read(in);
  return move(result);
}

void Network::Write(std::ostream &out) {
  impl->spec.Write(out);
  impl->layerWeights.Write(out);
}

EVector Network::Process(const EVector &input) const { return impl->Process(input); }
void Network::Update(const SamplesProvider &samplesProvider, float learnRate) {
  impl->Update(samplesProvider, learnRate);
}

uptr<Network> Network::RefreshAndGetTarget(void) {
  // Its a bit annoying I cant used make_unique coz of the private constructor...
  uptr<Network> roNetwork(new Network());
  roNetwork->impl = impl->RefreshAndGetTarget();
  return roNetwork;
}
