#pragma once

#include "CNNConfig.h"
#include "NNLayer.h"

class MaxPoolingLayer : public NNLayer { // Max Pooling Layer
  public:
    MaxPoolingLayer(const MaxPoolingLayerConfig& config);
    virtual ~MaxPoolingLayer() = default;

    NNMatrixPtrV forward(const NNMatrixPtrV& input);
    NNMatrixPtrV backward(const NNMatrixPtrV& inputs, const NNMatrixPtrV& dOutputs);

  private:
    const std::string TAG = "MaxPoolingLayer";
    int filterSize;
    int stride;
};
