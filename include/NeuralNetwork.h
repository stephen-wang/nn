#pragma once

#include "NNLayer.h"
#include <vector>
#include <string>

class NeuralNetwork {
private:
    const std::string TAG = "NeuralNetwork";

public:
    NeuralNetwork(const std::vector<int> &config);
    void train(std::vector<NNMatrixPtr> &X, std::vector<NNMatrixPtr> &Y, std::vector<NNMatrixPtr> &testX, std::vector<NNMatrixPtr> &testY, int epochNum, int batchSize, float learningRate, float momentum);

private:
    NNMatrix forward(int epic, int batchNo, const std::vector<NNMatrixPtr> &X);
    void backward(const std::vector<NNMatrixPtr> &X, const std::vector<NNMatrixPtr> &Y, float learningRate, float momentum);
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