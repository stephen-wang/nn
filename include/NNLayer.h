#pragma once

#include <memory>

class NNLayer {
  public:
    NNLayer() = default;
    virtual ~NNLayer() = default;
};

typedef std::shared_ptr<NNLayer> NNLayerPtr;