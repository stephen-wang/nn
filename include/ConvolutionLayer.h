#pragma once

#include "CNNConfig.h"
#include "NNLayer.h"
#include "NNMatrix.h"

#include <vector>

class ConvolutionLayer : public NNLayer { // Convolution Layer
  private:
    struct ForwardParams {
        int inputHeight = 0;
        int inputWidth = 0;
        int padHeight = 0;
        int padWidth = 0;
    };

    struct ZeroPadParams {
        int height = 0;
        int width = 0;
        int outHeight = 0;
        int outWidth = 0;
        const float* inData = nullptr;
    };

    struct ConvolveParams {
        NNMatrixPtr padInput;
        const float* inData = nullptr;
        const float* filterData = nullptr;
        int padRowSize = 0;
        int padColSize = 0;
        int outRowSize = 0;
        int outColSize = 0;
        NNMatrixPtr output;
        float* outData = nullptr;
    };

    bool prepareZeroPad(const NNMatrixPtr& input, ZeroPadParams& params) const;
    // If outParams is nullptr, only validates inputs (no outputs written).
    bool prepareForward(const NNMatrixPtrV& inputs, ForwardParams* outParams) const;
    bool prepareConvolution(const NNMatrixPtr& input, const NNMatrixPtr& filter,
                            ConvolveParams& params);
    NNMatrixPtr zeroPad(NNMatrixPtr input);
    NNMatrixPtr convolve(NNMatrixPtr input, NNMatrixPtr filter);

  public:
    ConvolutionLayer(const ConvolutionLayerConfig& config);
    virtual ~ConvolutionLayer();

    NNMatrixPtrV forward(const NNMatrixPtrV& inputs);

    // Backpropagation
    void zeroGrad();
    void applyGrad(int batchSize, float learningRate, float momentum);
    NNMatrixPtrV backward(const NNMatrixPtrV& inputs, const NNMatrixPtrV& outputs,
                          const NNMatrixPtrV& dOutputs);

  private:
    const std::string TAG = "ConvolutionLayer";
    int inChannelSize;
    int outChannelSize;
    int filterSize;
    int stride;
    int padding;
    MatrixFunc actFunction;
    NNMatrixPtrV filters;
    NNMatrixPtrV vFilters;
    NNMatrixPtrV gradFilters;

    // Per-output-channel bias (one scalar per output feature map).
    std::vector<float> bias;
    std::vector<float> vBias;
    std::vector<float> gradBias;
};