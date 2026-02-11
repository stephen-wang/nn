#pragma once

#include "CNNConfig.h"
#include "NNLayer.h"

class PoolingLayer : public NNLayer { // Pooling Layer
  public:
    PoolingLayer(const PoolingLayerConfig& config);
    virtual ~PoolingLayer();
};