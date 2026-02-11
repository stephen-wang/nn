#include "CNN.h"

#include "ConvolutionLayer.h"
#include "FCNNLayer.h"
#include "PoolingLayer.h"
#include "nnlog/nnlog.h"

#include <memory>

CNN::CNN(const std::vector<CNNConfigPtr>& configs) {
    for (const auto& configPtr : configs) {
        if (configPtr == nullptr) {
            LOG << "Null CNNConfigPtr in configs";
            continue;
        }
        auto layerPtr = buildCNNLayer(*configPtr);
        layers.push_back(layerPtr);
    }
}

std::shared_ptr<NNLayer> CNN::buildCNNLayer(const CNNConfig& config) {
    std::shared_ptr<NNLayer> layerPtr(nullptr);
    switch (config.getType()) {
    case CNNLayerType::Convolution:
        if (const auto* convCfg = dynamic_cast<const ConvolutionLayerConfig*>(&config)) {
            layerPtr = std::make_unique<ConvolutionLayer>(*convCfg);
        } else {
            LOG << "Convolution layer requires ConvolutionLayerConfig";
        }
        break;
    case CNNLayerType::Pooling:
        if (const auto* poolingCfg = dynamic_cast<const PoolingLayerConfig*>(&config)) {
            layerPtr = std::make_unique<PoolingLayer>(*poolingCfg);
        } else {
            LOG << "Pooling layer requires PoolingLayerConfig";
        }
        break;
    case CNNLayerType::FullyConnected:
        layerPtr = std::make_unique<FCNNLayer>(config.getInputSize(), config.getOutputSize());
        break;
    default:
        LOG << "Unsupported layer type " << static_cast<int>(config.getType());
        break;
    }

    return layerPtr;
}

CNN::~CNN() {
    layers.clear();
}
