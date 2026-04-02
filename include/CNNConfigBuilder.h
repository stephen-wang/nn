#pragma once

#include "CNNConfig.h"

#include <memory>
#include <vector>

class CNNConfigBuilder {
  public:
    CNNConfigBuilder& addConvolution(int inputSize, int outputSize, int kernelSize, int stride,
                                     int padding, MatrixFunc actFunction = NNFunctions::ReLUFunc) {
        configs_.push_back(std::make_shared<ConvolutionLayerConfig>(
            inputSize, outputSize, kernelSize, stride, padding, actFunction));
        return *this;
    }

    CNNConfigBuilder& addMaxPooling(int kernelSize, int stride) {
        configs_.push_back(std::make_shared<MaxPoolingLayerConfig>(kernelSize, stride));
        return *this;
    }

    CNNConfigBuilder& addFullyConnected(int inputSize, int outputSize) {
        configs_.push_back(
            std::make_shared<CNNConfig>(CNNLayerType::FullyConnected, inputSize, outputSize));
        return *this;
    }

    CNNConfigBuilder& addBatchNorm(int channels) {
        configs_.push_back(std::make_shared<BatchNormLayerConfig>(channels));
        return *this;
    }

    std::vector<std::shared_ptr<const CNNConfig>> build() const { return configs_; }

  private:
    std::vector<std::shared_ptr<const CNNConfig>> configs_;
};
