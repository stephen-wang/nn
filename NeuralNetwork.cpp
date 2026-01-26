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
        layer.dump();
        layers.push_back(layer);
    }

    layerOutputs = std::vector<std::vector<NNMatrix>>(configSize - 1, std::vector<NNMatrix>());
}

void NeuralNetwork::train(std::vector<NNMatrixPtr>& X, std::vector<NNMatrixPtr>& Y, std::vector<NNMatrixPtr> &testX, std::vector<NNMatrixPtr> &testY, int epochNum, int batchSize, float learningRate, float momentum) {
    int e = 0;
    while (e < epochNum) {
        LOG << "Epic " << e << std::endl;
        NNUtils::shuffle(X, Y);
        int numBatches = (X.size() + 1) / batchSize;
        float epochLoss = 0.0f;
        for (int b = 0; b < numBatches; b++) {
            if (b % 200 == 0) {
                LOG << "Epic " << e << ", batch " << b << " starts" << std::endl;
                //layers[layers.size()-1].dump();
                 
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
    //LOG << "Forward start" << std::endl;
    for (auto& batchOutpus : layerOutputs) {
        batchOutpus.clear();
    }
 
    //static int cnt = 0;
    //int batchSeq = -1;
    for (auto x: X) {
        //batchSeq++;
        NNMatrix input(*x);
        //LOG << "epic " << epic << ", batch " << batchNo << ", origin input " << std::endl;
        //x->dump();
        for (int i = 0; i < layers.size(); i++) {
            //if (cnt < 3) {
             //LOG << "epic " << epic << ", batch " << batchNo << ", batchSeq " << batchSeq << ", layer " << i << " forward input" << std::endl;
             //input.dump();
            //}
        
            if (i < layers.size() - 1) {
                // use sigmoid for hidden layers
                // if (i == 1)
                //     input = layers[i].forward(input, NNFunctions::SigmoidFunc, true);
                // else 
                    input = layers[i].forward(input, NNFunctions::SigmoidFunc, false);
            } else {
                // use softmax for output layer
                input = layers[i].forward(input, nullptr);
                input = NNFunctions::softmax(input);
            }
            //if (cnt < 3) {
                // LOG << "layer " << i << " forward output" << std::endl;
                // input.dump();
            //}
            layerOutputs[i].push_back(input);
        }
    }

    //LOG << "Forward output: " << std::endl;
    //LOG << "epic " << epic << ", batch " << batchNo << ", forward output: " << std::endl;
    //layerOutputs[layers.size()-1][X.size()-1].dump();

    return layerOutputs[layers.size()-1][X.size()-1];

    //cnt++;
    //LOG << "Forward ended" << std::endl;    
}

void NeuralNetwork::backward(const std::vector<NNMatrixPtr> &X, const std::vector<NNMatrixPtr> &Y, float learningRate, float momentum) {
    //LOG << "Backward start" << std::endl;
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
        LOG << "Dump output layer dw: " << std::endl;
        dws[outputLayerId].dump();
        dbs[outputLayerId] += dzs[outputLayerId];
 
        // hidden layers derivatives
        for (int l = layerSize - 2; l > 0; l--) {
            auto da = layers[l+1].calculatePrevLayerDA(dzs[l+1]);
            dzs[l] = da.elementProduct(layerOutputs[l][i].applyFunction(NNFunctions::SigmoidDrevative));
            dws[l] += calculateDW(layerOutputs[l-1][i], dzs[l]);
            dbs[l] += dzs[l];
        }

        // 1st hidden layer derivative
        auto da = layers[1].calculatePrevLayerDA(dzs[1]);
        //LOG << "layer " << 0 << ", dA size " << da.getRowSize() << "x" << da.getColSize() << std::endl;
       
        // layerOutputs[0][i].dump();
        // LOG << "Dump activated layer 0 output for patch " << i << ": " << std::endl;
        auto activeLayerOutput = layerOutputs[0][i].applyFunction(NNFunctions::SigmoidDrevative);
        // activeLayerOutput.dump();
        dzs[0] = da.elementProduct(activeLayerOutput);
        // LOG << "Dump layer 0 dz: " << std::endl;
        // dzs[0].dump();
        //LOG << "x size " << x.getRowSize() << "x" << x.getColSize() << std::endl;
        // LOG << "Dump current input X: " << std::endl;
        // x.dump();
        auto curDW = calculateDW(x, dzs[0]);
        // LOG << "Dump layer 0 current DW for patch " << i << ": " << std::endl;
        // curDW.dump(true);
        dws[0] += curDW;
        // LOG << "Dump layer 0 DW for patch " << i << ": " << std::endl;
        // dws[0].dump(true);
        dbs[0] += dzs[0];
    }

    for (auto &dw: dws) {
        dw /= (float)batchSize;
        // LOG << "Dump new dw: " << std::endl;
        // dw.dump();
    }
    for (auto &db : dbs) {
        db /= (float)batchSize;
    }

    for (int l = 0; l < layers.size(); l++) {
        layers[l].update(dws[l], dbs[l], learningRate, momentum);
        layers[l].dump();
    }

    //LOG << "Backward ended" << std::endl;    
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
