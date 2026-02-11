#pragma once

#include "CNNConfig.h"
#include "NNLayer.h"
#include "NNMatrix.h"

class ConvolutionLayer : public NNLayer { // Convolution Layer
  private:
    NNMatrixPtr zeroPad(NNMatrixPtr input);
    NNMatrixPtr convolve(NNMatrixPtr input, NNMatrixPtr filter);

  public:
    ConvolutionLayer(const ConvolutionLayerConfig& config);
    virtual ~ConvolutionLayer();

    NNMatrixPtrVector forward(const NNMatrixPtrVector& input);

  private:
    const std::string TAG = "ConvolutionLayer";
    int inChannelSize;
    int outChannelSize;
    int filterSize;
    int stride;
    int padding;
    NNMatrixPtrVector filters;
};