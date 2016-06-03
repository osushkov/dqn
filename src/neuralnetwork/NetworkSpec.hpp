
#pragma once

#include <iostream>
#include <vector>

namespace neuralnetwork {

enum class LayerActivation { TANH, LOGISTIC, RELU, LEAKY_RELU, LINEAR };

struct NetworkSpec {
  unsigned numInputs;
  unsigned numOutputs;
  std::vector<unsigned> hiddenLayers;

  unsigned maxBatchSize;

  LayerActivation hiddenActivation;
  LayerActivation outputActivation;

  inline void Write(std::ostream &out) {
    out << numInputs << std::endl;
    out << numOutputs << std::endl;

    out << hiddenLayers.size() << std::endl;
    for (unsigned i = 0; i < hiddenLayers.size(); i++) {
      out << hiddenLayers[i] << std::endl;
    }
    out << maxBatchSize << std::endl;
    out << static_cast<int>(hiddenActivation) << std::endl;
    out << static_cast<int>(outputActivation) << std::endl;
  }

  static NetworkSpec Read(std::istream &in) {
    NetworkSpec spec;
    in >> spec.numInputs;
    in >> spec.numOutputs;

    unsigned numLayers;
    in >> numLayers;

    for (unsigned i = 0; i < numLayers; i++) {
      unsigned layerSize;
      in >> layerSize;
      spec.hiddenLayers.push_back(layerSize);
    }

    in >> spec.maxBatchSize;

    int ha, oa;
    in >> ha;
    in >> oa;

    spec.hiddenActivation = static_cast<LayerActivation>(ha);
    spec.outputActivation = static_cast<LayerActivation>(oa);

    return spec;
  }
};
}
