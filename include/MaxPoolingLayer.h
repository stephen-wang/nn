#pragma once

#include "CNNConfig.h"
#include "NNLayer.h"

class MaxPoolingLayer : public NNLayer { // Max Pooling Layer
  public:
    MaxPoolingLayer(const MaxPoolingLayerConfig& config);
    virtual ~MaxPoolingLayer() = default;

    NNMatrixPtrV forward(const NNMatrixPtrV& input);
    NNMatrixPtrV backward(const NNMatrixPtrV& inputs, const NNMatrixPtrV& dOutputs);
    int getFilterSize() const { return filterSize; }
    int getStride() const { return stride; }

  private:
    const std::string TAG = "MaxPoolingLayer";
    int filterSize;
    int stride;
};
