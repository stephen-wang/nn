#pragma once
#include <memory>

enum class CNNLayerType { Convolution = 0, Pooling = 1, FullyConnected = 2 };

class CNNConfig {
  public:
    CNNConfig(CNNLayerType type, int inputSize, int outputSize)
        : type(type), inputSize(inputSize), outputSize(outputSize) {}
    virtual ~CNNConfig() = default;
    CNNLayerType getType() const { return type; }
    int getInputSize() const { return inputSize; }
    int getOutputSize() const { return outputSize; }

  private:
    CNNLayerType type;
    int inputSize;
    int outputSize;
};

class ConvolutionLayerConfig : public CNNConfig {
  public:
    ConvolutionLayerConfig(int inChannelSize, int outChannelSize, int kernelSize, int stride,
                           int padding)
        : CNNConfig(CNNLayerType::Convolution, inChannelSize, outChannelSize),
          kernelSize(kernelSize), stride(stride), padding(padding) {}
    int getInChannelSize() const { return getInputSize(); }
    int getOutChannelSize() const { return getOutputSize(); }
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }
    int getPadding() const { return padding; }

  private:
    int kernelSize;
    int stride;
    int padding;
};

class PoolingLayerConfig : public CNNConfig {
  public:
    PoolingLayerConfig(int inputSize, int outputSize, int kernelSize, int stride)
        : CNNConfig(CNNLayerType::Pooling, inputSize, outputSize), kernelSize(kernelSize),
          stride(stride) {}
    int getKernelSize() const { return kernelSize; }
    int getStride() const { return stride; }

  private:
    int kernelSize;
    int stride;
};

typedef std::shared_ptr<const CNNConfig> CNNConfigPtr;
