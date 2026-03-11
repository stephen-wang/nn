#pragma once

#include "FCNNLayer.h"
#include "NN.h"
#include "NNDataset.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class DNN : public NN { // Simple Deep Neural Network
  private:
    const std::string TAG = "DNN";
    std::string checkpointFilePath_;
    bool loadCheckpointBeforeTrain_ = false;

  public:
    DNN(const std::vector<int>& config);
    using TrainCallback =
        std::function<void(int epoch, int totalEpochs, float loss, float accuracy)>;
    using BatchCallback =
        std::function<void(int epoch, int batch, const NNMatrix& input, const NNMatrix& output)>;
    using BatchStatsCallback =
        std::function<void(int epoch, int totalEpochs, int batch, int totalBatches, float batchLoss,
                           float epochLoss, float batchAccuracy)>;
    using StopCallback = std::function<bool()>;
    enum class LayerPhase : std::uint8_t { Idle = 0, Forward = 1, Backward = 2 };
    using LayerCallback =
        std::function<void(int epoch, int batch, int layerIndex, LayerPhase phase)>;
    void train(NNDataset& dataset, int epochNum, int batchSize, float learningRate, float momentum,
               TrainCallback callback = nullptr, LayerCallback layerCallback = nullptr,
               BatchCallback batchCallback = nullptr, StopCallback stopCallback = nullptr,
               BatchStatsCallback batchStatsCallback = nullptr);
    bool save(const std::string& filePath) const;
    bool load(const std::string& filePath);
    void configurePersistence(const std::string& checkpointFilePath, bool loadBeforeTrain) {
        checkpointFilePath_ = checkpointFilePath;
        loadCheckpointBeforeTrain_ = loadBeforeTrain;
    }

  private:
    NNMatrix forward(int epic, int batchNo, const std::vector<NNMatrixPtr>& X,
                     LayerCallback layerCallback);
    void backward(const std::vector<NNMatrixPtr>& X, const std::vector<NNMatrixPtr>& Y,
                  float learningRate, float momentum, int epic, int batchNo,
                  LayerCallback layerCallback);
    float loss(std::vector<NNMatrixPtr>& Y);
    NNMatrix calculateDW(const NNMatrix& input, const NNMatrix& dz);
    float accuracy(int epic, const std::vector<NNMatrixPtr>& x_test,
                   const std::vector<NNMatrixPtr>& y_test);
    NNMatrix predict(int epic, NNMatrixPtr x);
    int argmax(const NNMatrix& x);

  public:
    std::vector<FCNNLayer> layers;
    std::vector<std::vector<NNMatrix>> layerOutputs;
};