#include "NeuralNetwork.h"
#include "NNUtils.h"
#include "NNFunctions.h"
#include <iostream>
#include <math.h>
#include <iomanip>

NeuralNetwork::NeuralNetwork(const std::vector<int> &config) {
    int configSize = config.size();
    if (configSize < 3) {
        LOG << "Invalid NeuralNetwork config " << configSize << std::endl;
        return;
    }

    for (int l = 1; l < configSize; l++) {
        auto layer = NNLayer(config[l-1], config[l]);
        //layer.dump();
        layers.push_back(layer);
    }

    layerOutputs = std::vector<std::vector<NNMatrix>>(configSize - 1, std::vector<NNMatrix>());
}

void NeuralNetwork::train(
    std::vector<NNMatrixPtr>& X,
    std::vector<NNMatrixPtr>& Y,
    std::vector<NNMatrixPtr> &testX,
    std::vector<NNMatrixPtr> &testY,
    int epochNum,
    int batchSize,
    float learningRate,
    float momentum
) {
    int e = 0;
    while (e < epochNum) {
        LOG << "Epic " << e << std::endl;
        NNUtils::shuffle(X, Y);
        int numBatches = (X.size() + 1) / batchSize;
        float epochLoss = 0.0f;
        for (int b = 0; b < numBatches; b++) {
            if (b % 200 == 0) {
                LOG << "Epic " << e << ", batch " << b << " starts" << std::endl;
                 
            }
            std::vector<NNMatrixPtr> batchX = NNUtils::getBatch(X, b, batchSize);
            std::vector<NNMatrixPtr> batchY = NNUtils::getBatch(Y, b, batchSize);
            forward(e, b, batchX);
            epochLoss += loss(batchY);
            backward(batchX, batchY, learningRate, momentum);
        }

        float avgLoss = epochLoss/ numBatches;
        float acc = accuracy(e, testX, testY);
        std::cout << "Epic " << e+1 << "/"  << epochNum << ", loss " << avgLoss << ", acc " << std::setprecision(3) <<  acc * 100 << std::endl;
        e++;
    }
}

NNMatrix NeuralNetwork::forward(int epic, int batchNo, const std::vector<NNMatrixPtr> &X) {
    for (auto& batchOutpus : layerOutputs) {
        batchOutpus.clear();
    }
 
    for (auto x: X) {
        NNMatrix input(*x);
        for (int i = 0; i < layers.size(); i++) {
        
            if (i < layers.size() - 1) {
                input = layers[i].forward(input, NNFunctions::ReLUFunc, false);
            } else {
                input = layers[i].forward(input, nullptr);
                input = NNFunctions::softmax(input);
            }
            layerOutputs[i].push_back(input);
        }
    }
    
    return layerOutputs[layers.size()-1][X.size()-1];
}

void NeuralNetwork::backward(
    const std::vector<NNMatrixPtr> &X,
    const std::vector<NNMatrixPtr> &Y,
    float learningRate,
    float momentum
) {
    int batchSize = X.size();
    std::vector<NNMatrix> dws;
    std::vector<NNMatrix> dbs;
    std::vector<NNMatrix> dzs;
    for (int i = 0; i < layers.size(); i++) {
        int layerOuputSize = layers[i].getOutputSize();
        int layerInputSize = layers[i].getInputSize();
        dws.push_back(NNMatrix(layerOuputSize, layerInputSize));
        dbs.push_back(NNMatrix(layerOuputSize, 1));
        dzs.push_back(NNMatrix(layerOuputSize, 1));
    }

    for (int i = 0; i < batchSize; i++) {
        auto &x = *(X[i]);
        auto &y = *(Y[i]);

        // output layer derivatives
        int layerSize = layers.size();
        int outputLayerId = layerSize - 1;
        dzs[outputLayerId] = layerOutputs[outputLayerId][i] - y;
        dws[outputLayerId] += calculateDW(layerOutputs[outputLayerId-1][i], dzs[outputLayerId]);
        dbs[outputLayerId] += dzs[outputLayerId];
 
        // hidden layers derivatives
        for (int l = layerSize - 2; l > 0; l--) {
            auto da = layers[l+1].calculatePrevLayerDA(dzs[l+1]);
            dzs[l] = da.elementProduct(layerOutputs[l][i].applyFunction(NNFunctions::ReLUDrevative));
            dws[l] += calculateDW(layerOutputs[l-1][i], dzs[l]);
            dbs[l] += dzs[l];
        }

        // 1st hidden layer derivative
        auto da = layers[1].calculatePrevLayerDA(dzs[1]);
        auto activeLayerOutput = layerOutputs[0][i].applyFunction(NNFunctions::ReLUDrevative);
        dzs[0] = da.elementProduct(activeLayerOutput);
        auto curDW = calculateDW(x, dzs[0]);
        dws[0] += curDW;
        dbs[0] += dzs[0];
    }

    for (auto &dw: dws) {
        dw /= (float)batchSize;
    }
    for (auto &db : dbs) {
        db /= (float)batchSize;
    }

    for (int l = 0; l < layers.size(); l++) {
        layers[l].update(dws[l], dbs[l], learningRate, momentum);
    }

}

NNMatrix NeuralNetwork::calculateDW(const NNMatrix &input, const NNMatrix &dz) {
    assert(input.getColSize() == 1);
    assert(dz.getColSize() == 1);

    NNMatrix dw(dz.getRowSize(), input.getRowSize());
    for (int i = 0; i < dw.getRowSize(); i++) {
        for (int j = 0; j < dw.getColSize(); j++) {
            dw.set(i, j, input.get(j, 0) * dz.get(i, 0));
        }
    }

    return dw;
}

float NeuralNetwork::loss(std::vector<NNMatrixPtr> &Y) {
    float totalLoss = 0.0f;
    for (int i = 0; i < Y.size(); i++) {
        auto &actual = layerOutputs[layers.size()-1][i];
        auto &expect = *(Y[i]);
        totalLoss += calculateCrossEntropyLoss(actual, expect);
    }

    return totalLoss/Y.size();

}

float NeuralNetwork::calculateCrossEntropyLoss(const NNMatrix &actual, const NNMatrix &expect) {
    assert(actual.getRowSize() == expect.getRowSize());
    assert(actual.getColSize() == 1);
    assert(expect.getColSize() == 1);

    float eps = 1e-15f;
    float loss = 0.0f;
    for (int i = 0; i < actual.getRowSize(); i++) {
        float expectElem = expect.get(i, 0);
        float actualElem = actual.get(i, 0);
        float actualElemClipped = std::max(eps, std::min(1.0f - eps, actualElem));
        loss -= expectElem * std::log(actualElemClipped);
    }

    return loss;
}

float NeuralNetwork::accuracy(int epic, const std::vector<NNMatrixPtr> &x_test, const std::vector<NNMatrixPtr> &y_test) {
    assert(x_test.size() == y_test.size());
    int correct = 0;
    for (int i = 0; i < x_test.size(); i++) {
        auto x = x_test[i];
        auto &y_true = *(y_test[i]);
        auto pred = predict(epic, x);
        if (argmax(pred) == argmax(y_true)) {
                correct += 1;
        }
    }
    return (float)correct / x_test.size();
}

NNMatrix NeuralNetwork::predict(int epic, NNMatrixPtr x) {
    std::vector<NNMatrixPtr> input = { x };
    return forward(epic, 0, input);
}

int NeuralNetwork::argmax(const NNMatrix &x) {
    assert(x.getColSize() == 1);
    return x.getIndexOfColMax(0);
}
