#pragma once

#include "CNNConfig.h"
#include "NNLayer.h"

#include <memory>
#include <string>
#include <vector>

class CNN { // Simple Convolutional Neural Network
  private:
    const std::string TAG = "CNN";
    std::vector<NNLayerPtr> layers;

  public:
    CNN(const std::vector<CNNConfigPtr>& configs);
    virtual ~CNN();

  private:
    std::shared_ptr<NNLayer> buildCNNLayer(const CNNConfig& config);
};