
#include "Network.hpp"
#include "../common/Util.hpp"
#include "../math/Math.hpp"
#include "../math/Tensor.hpp"
#include "Activations.hpp"
#include "cuda/CudaNetwork.hpp"

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

  NetworkImpl(const NetworkSpec &spec, bool isTrainable) : spec(spec) {
    assert(spec.numInputs > 0 && spec.numOutputs > 0);
    initialiseWeights();

    if (isTrainable) {
      initialiseCuda();
    } else {
      cudaNetwork = nullptr;
    }
  }

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

  void Refresh(void) {
    assert(cudaNetwork != nullptr);

    // obtain a write lock
    boost::unique_lock<boost::shared_mutex> lock(rwMutex);

    vector<math::MatrixView> weightViews;
    for (unsigned i = 0; i < layerWeights.NumLayers(); i++) {
      weightViews.push_back(math::GetMatrixView(layerWeights(i)));
    }
    cudaNetwork->UpdateTarget();
    cudaNetwork->GetWeights(weightViews);
  }

  void Update(const SamplesProvider &samplesProvider) {
    assert(cudaNetwork != nullptr);
    assert(samplesProvider.NumSamples() <= spec.maxBatchSize);

    cuda::QBatch qbatch;
    qbatch.actionsTaken = std::vector<unsigned>(samplesProvider.NumSamples());
    qbatch.isEndStateTerminal = std::vector<char>(samplesProvider.NumSamples());
    qbatch.rewardsGained = std::vector<float>(samplesProvider.NumSamples());

    EMatrix initialStates(samplesProvider.NumSamples(), spec.numInputs);
    EMatrix successorStates(samplesProvider.NumSamples(), spec.numInputs);

    for (unsigned i = 0; i < samplesProvider.NumSamples(); i++) {
      const TrainingSample &sample = samplesProvider[i];

      assert(sample.startState.cols() == 1 && sample.startState.rows() == spec.numInputs);
      assert(sample.endState.cols() == 1 && sample.endState.rows() == spec.numInputs);
      assert(sample.actionTaken < spec.numOutputs);

      for (unsigned j = 0; j < sample.startState.rows(); j++) {
        initialStates(i, j) = sample.startState(j);
        successorStates(i, j) = sample.endState(j);
      }

      qbatch.actionsTaken[i] = sample.actionTaken;
      qbatch.isEndStateTerminal[i] = static_cast<char>(sample.isEndStateTerminal);
      qbatch.rewardsGained[i] = sample.rewardGained;
      qbatch.futureRewardDiscount = sample.futureRewardDiscount;
    }

    qbatch.initialStates = math::GetMatrixView(initialStates);
    qbatch.successorStates = math::GetMatrixView(successorStates);

    std::lock_guard<std::mutex> lock(trainMutex);
    cudaNetwork->Train(qbatch);
  }

  uptr<NetworkImpl> ReadOnlyCopy(void) const {
    // obtain a read lock
    boost::shared_lock<boost::shared_mutex> lock(rwMutex);

    auto result = make_unique<NetworkImpl>(spec, false);
    result->layerWeights = layerWeights;
    return result;
  }

  EVector getLayerOutput(const EVector &prevLayer, const EMatrix &layerWeights,
                         LayerActivation afunc) const {
    assert(prevLayer.rows() == layerWeights.cols() - 1);

    EVector z = layerWeights * getInputWithBias(prevLayer);
    if (afunc == LayerActivation::SOFTMAX) {
      z = softmaxActivations(z);
    } else {
      for (unsigned i = 0; i < z.rows(); i++) {
        z(i) = ActivationValue(afunc, z(i));
      }
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

  EVector softmaxActivations(const EVector &in) const {
    assert(in.rows() > 0);
    EVector result(in.rows());

    float maxVal = in(0);
    for (int r = 0; r < in.rows(); r++) {
      maxVal = fmax(maxVal, in(r));
    }

    float sum = 0.0f;
    for (int i = 0; i < in.rows(); i++) {
      result(i) = expf(in(i)-maxVal);
      sum += result(i);
    }

    for (int i = 0; i < result.rows(); i++) {
      result(i) /= sum;
    }

    return result;
  }
};

Network::Network(const NetworkSpec &spec) : impl(new NetworkImpl(spec, true)) {}
Network::~Network() = default;

EVector Network::Process(const EVector &input) const { return impl->Process(input); }
void Network::Refresh(void) { impl->Refresh(); }
void Network::Update(const SamplesProvider &samplesProvider) { impl->Update(samplesProvider); }

uptr<Network> Network::ReadOnlyCopy(void) const {
  // Its a bit annoying I cant used make_unique coz of the private constructor...
  uptr<Network> roNetwork(new Network());
  roNetwork->impl = impl->ReadOnlyCopy();
  return roNetwork;
}
