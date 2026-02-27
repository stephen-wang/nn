#pragma once

#include "CNNConfig.h"
#include "NN.h"
#include "NNDataset.h"
#include "NNLayer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class CNN : public NN { // Simple Convolutional Neural Network
  private:
    const std::string TAG = "CNN";
    std::vector<NNLayerPtr> layers;
    NNMatrixPtrV lastForwardOutputs_;
    NNMatrixPtrVV fcLayerInputs_;
    NNMatrixPtrVV fcLayerOutputs_;
    std::vector<NNMatrixPtrVV> layerOutputs_;

    // Scratch buffers reused across batches to reduce allocation pressure.
    // - `flatScratchBySample_`: flattened conv activations (input to first FC layer)
    // - `dUnflattenScratchBySample_`: reshaped gradients (from first FC back into conv stack)
    NNMatrixPtrV flatScratchBySample_;
    std::vector<NNMatrixPtrV> dUnflattenScratchBySample_;

  public:
    CNN(const std::vector<CNNConfigPtr>& configs);
    virtual ~CNN() { layers.clear(); }

    void train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum);

    void train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
               float weightDecay);

    using TrainCallback =
        std::function<void(int epoch, int totalEpochs, float loss, float accuracy)>;
    using BatchCallback = std::function<void(int epoch, int batch, const NNMatrixPtrV& input,
                                             const NNMatrix& output)>;
    using BatchStatsCallback =
        std::function<void(int epoch, int totalEpochs, int batch, int totalBatches, float batchLoss,
                           float epochLoss, float batchAccuracy)>;
    using StopCallback = std::function<bool()>;
    enum class LayerPhase : std::uint8_t { Idle = 0, Forward = 1, Backward = 2 };
    using LayerCallback =
        std::function<void(int epoch, int batch, int layerIndex, LayerPhase phase)>;

    void train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
               TrainCallback callback, LayerCallback layerCallback, BatchCallback batchCallback,
               StopCallback stopCallback, BatchStatsCallback batchStatsCallback);

    void train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
               float weightDecay, TrainCallback callback, LayerCallback layerCallback,
               BatchCallback batchCallback, StopCallback stopCallback,
               BatchStatsCallback batchStatsCallback);

  private:
    std::shared_ptr<NNLayer> buildCNNLayer(const CNNConfig& config);
    NNMatrixPtrV forward(int epoc, int batchNo, int inChannelSize, const NNMatrixPtrV& X,
                         bool training, LayerCallback layerCallback = nullptr);
    void backward(const NNMatrixPtrV& X, const NNMatrixPtrV& Y, float learningRate, float momentum,
                  float weightDecay, int epoc, int batchNo, int inChannelSize,
                  LayerCallback layerCallback = nullptr);
    float loss(NNMatrixPtrV& Y);
    float accuracy(int epoc, const NNMatrixPtrV& x_test, const NNMatrixPtrV& y_test);
};