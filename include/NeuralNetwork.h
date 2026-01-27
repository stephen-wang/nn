#pragma once

#include "NNLayer.h"
#include <vector>
#include <string>
#include <functional>

class NeuralNetwork
{
private:
    const std::string TAG = "NeuralNetwork";

public:
    NeuralNetwork(const std::vector<int> &config);
    using TrainCallback = std::function<void(int epoch, int totalEpochs, float loss, float accuracy)>;
    using BatchCallback = std::function<void(int epoch, int batch, const NNMatrix &input, const NNMatrix &output)>;
    using StopCallback = std::function<bool()>;
    enum class LayerPhase
    {
        Idle = 0,
        Forward = 1,
        Backward = 2
    };
    using LayerCallback = std::function<void(int epoch, int batch, int layerIndex, LayerPhase phase)>;
    void train(std::vector<NNMatrixPtr> &X, std::vector<NNMatrixPtr> &Y, std::vector<NNMatrixPtr> &testX, std::vector<NNMatrixPtr> &testY, int epochNum, int batchSize, float learningRate, float momentum, TrainCallback callback = nullptr, LayerCallback layerCallback = nullptr, BatchCallback batchCallback = nullptr, StopCallback stopCallback = nullptr);

private:
    NNMatrix forward(int epic, int batchNo, const std::vector<NNMatrixPtr> &X, LayerCallback layerCallback);
    void backward(const std::vector<NNMatrixPtr> &X, const std::vector<NNMatrixPtr> &Y, float learningRate, float momentum, int epic, int batchNo, LayerCallback layerCallback);
    float loss(std::vector<NNMatrixPtr> &Y);
    float calculateCrossEntropyLoss(const NNMatrix &actual, const NNMatrix &expect);
    NNMatrix calculateDW(const NNMatrix &input, const NNMatrix &dz);
    float accuracy(int epic, const std::vector<NNMatrixPtr> &x_test, const std::vector<NNMatrixPtr> &y_test);
    NNMatrix predict(int epic, NNMatrixPtr x);
    int argmax(const NNMatrix &x);

public:
    std::vector<NNLayer> layers;
    std::vector<std::vector<NNMatrix>> layerOutputs;
};