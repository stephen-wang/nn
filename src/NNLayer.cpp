#include "NNLayer.h"

#include <iomanip>
#include <sstream>

NNLayer::NNLayer(int inputSize, int outputSize)
    : weight(outputSize, inputSize), vWeight(outputSize, inputSize), bias(outputSize, 1),
      vBias(outputSize, 1), dz_(outputSize, 1) {
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
            weight.set(i, j, NNUtils::xavierInit(inputSize, outputSize));
        }

        bias.set(i, 0, NNUtils::xavierInit(inputSize, outputSize));
    }
}

NNMatrix NNLayer::forward(const NNMatrix& input, MatrixFunc activateFunc, bool debug) {
    auto ret = weight.dotProduct(input);
    if (debug) {
        LOG << "weight: " << std::endl;
        weight.dump();
        LOG << "input: " << std::endl;
        input.dump();
        LOG << "weight dotProduct input: " << std::endl;
        ret.dump();
    }
    ret += bias;
    if (debug) {
        LOG << "weight x input + bias: " << std::endl;
        ret.dump();
    }

    if (activateFunc != nullptr)
        ret = ret.applyFunction(activateFunc);

    if (debug) {
        LOG << "weight x input + bias, apply activation func: " << std::endl;
        ret.dump();
    }
    return ret;
}

NNMatrix NNLayer::calculatePrevLayerDA(const NNMatrix& dz) {
    NNMatrix da(weight.getColSize(), 1);
    for (int i = 0; i < da.getRowSize(); i++) {
        float daElemValue = 0.0f;
        for (int j = 0; j < weight.getRowSize(); j++) {
            daElemValue += dz.get(j, 0) * weight.get(j, i);
        }
        da.set(i, 0, daElemValue);
    }

    return da;
}

void NNLayer::update(const NNMatrix& dw, const NNMatrix& db, float alpha, float momentum) {
    for (int i = 0; i < weight.getRowSize(); i++) {
        for (int j = 0; j < weight.getColSize(); j++) {
            auto delta = momentum * vWeight.get(i, j) + alpha * dw.get(i, j);
            vWeight.set(i, j, delta);
            weight.set(i, j, weight.get(i, j) - delta);
        }
    }

    for (int i = 0; i < bias.getRowSize(); i++) {
        auto delta = momentum * vBias.get(i, 0) + alpha * db.get(i, 0);
        vBias.set(i, 0, delta);
        bias.set(i, 0, bias.get(i, 0) - delta);
    }
}

void NNLayer::dump() {
    std::stringstream ss("\n");
    ss << "Layer input size " << getInputSize() << ", output size " << getOutputSize() << std::endl;
    LOG << ss.str();
    weight.dump(true);
    bias.dump(true);
}