#pragma once
#include "NNFunctions.h"
#include "NNMatrix.h"

#include <memory>

enum class CNNLayerType { Convolution = 0, Pooling = 1, FullyConnected = 2 };

class CNNConfig {
  public:
    CNNConfig(CNNLayerType type, int inputSize, int outputSize)
        : type(type), inputSize(inputSize), outputSize(outputSize), kernelSize(0), stride(0) {}

    CNNConfig(CNNLayerType type, int inputSize, int outputSize, int kernelSize, int stride)
        : type(type), inputSize(inputSize), outputSize(outputSize), kernelSize(kernelSize),
          stride(stride) {}
    virtual ~CNNConfig() = default;
    CNNLayerType getType() const { return type; }
    int getInputSize() const { return inputSize; }
    int getOutputSize() const { return outputSize; }
    int getInChannelSize() const { return getInputSize(); }
    int getOutChannelSize() const { return getOutputSize(); }
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }

  private:
    CNNLayerType type;
    int inputSize;
    int outputSize;
    int kernelSize;
    int stride;
};

class ConvolutionLayerConfig : public CNNConfig {
  public:
    ConvolutionLayerConfig(int inChannelSize, int outChannelSize, int kernelSize, int stride,
                           int padding, MatrixFunc actFucntion = NNFunctions::ReLUFunc)
        : CNNConfig(CNNLayerType::Convolution, inChannelSize, outChannelSize, kernelSize, stride),
          padding(padding), actFunction(actFucntion) {}
    int getPadding() const { return padding; }
    MatrixFunc getActFunction() const { return actFunction; }

private:
    int padding;
    MatrixFunc actFunction;
};

class MaxPoolingLayerConfig : public CNNConfig {
  public:
    MaxPoolingLayerConfig(int kernelSize, int stride)
        : CNNConfig(CNNLayerType::Pooling, -1, -1, kernelSize, stride) {}
};

typedef std::shared_ptr<const CNNConfig> CNNConfigPtr;
