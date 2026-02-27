#pragma once

#include <cstdint>
#include <memory>

enum class NNLayerType : std::uint8_t {
    Unknown = 0,
    Convolution = 1,
    Pooling = 2,
    FullyConnected = 3,
    BatchNorm = 4,
};

class NNLayer {
  public:
    explicit NNLayer(NNLayerType type = NNLayerType::Unknown) : type_(type) {}
    virtual ~NNLayer() = default;

    NNLayerType getLayerType() const { return type_; }

  protected:
    void setLayerType(NNLayerType type) { type_ = type; }

  private:
    NNLayerType type_ = NNLayerType::Unknown;
};

typedef std::shared_ptr<NNLayer> NNLayerPtr;