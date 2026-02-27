#pragma once

#include "NNLayer.h"
#include "NNMatrix.h"

#include <vector>

// BatchNorm over convolutional feature maps (per-channel), with learnable gamma/beta.
//
// Expected input/output shape:
// - Batch: vector of samples
// - Sample: vector of channels
// - Channel: matrix (H x W)
class BatchNormLayer : public NNLayer {
  public:
    explicit BatchNormLayer(int channels, float eps = 1e-5f, float runningMomentum = 0.1f);

    int channels() const { return channels_; }

    // Forward over a batch.
    std::vector<NNMatrixPtrV> forwardBatch(const std::vector<NNMatrixPtrV>& batch, bool training);

    // Backward over a batch: returns dX given dY.
    // Must be called after forwardBatch(training=true) for the same batch.
    std::vector<NNMatrixPtrV> backwardBatch(const std::vector<NNMatrixPtrV>& dY);

    // SGD+momentum update for gamma/beta (no weight decay by default).
    void update(float learningRate, float momentum);

  private:
    int channels_ = 0;
    float eps_ = 1e-5f;
    float runningMomentum_ = 0.1f;

    // Parameters.
    std::vector<float> gamma_;
    std::vector<float> beta_;

    // Optimizer state.
    std::vector<float> vGamma_;
    std::vector<float> vBeta_;

    // Running stats for inference.
    std::vector<float> runningMean_;
    std::vector<float> runningVar_;

    // Cached batch stats (training forward).
    std::vector<float> batchMean_;
    std::vector<float> batchInvStd_;

    // Cached xhat per sample/channel for backward.
    std::vector<NNMatrixPtrV> xhatBySample_;

    // Gradients.
    std::vector<float> dGamma_;
    std::vector<float> dBeta_;

    // Element count per channel in the cached batch.
    std::vector<int> elemCountPerChannel_;

    bool validateBatchShape(const std::vector<NNMatrixPtrV>& batch) const;
};
