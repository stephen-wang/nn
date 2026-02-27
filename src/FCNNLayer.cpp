#include "FCNNLayer.h"

#include <iomanip>
#include <sstream>

FCNNLayer::FCNNLayer(int inputSize, int outputSize)
    : NNLayer(NNLayerType::FullyConnected), weight(outputSize, inputSize),
      vWeight(outputSize, inputSize), bias(outputSize, 1), vBias(outputSize, 1),
      dz_(outputSize, 1) {
    for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < inputSize; j++) {
                        weight.set(i, j, NNUtils::heInit(inputSize));
        }

        bias.set(i, 0, 0.0f);
    }
}

NNMatrix FCNNLayer::forward(const NNMatrix& input, MatrixFunc activateFunc, bool debug) {
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

NNMatrix FCNNLayer::calculatePrevLayerDA(const NNMatrix& dz) {
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

NNMatrixPtrV FCNNLayer::forward(const NNMatrixPtrV& input) {
    NNMatrixPtrV outputs;
    if (input.empty() || input[0] == nullptr || input.size() > 1) {
        LOG << "Empty input or null input[0] or multiple inputs";
        return outputs;
    }

    outputs.reserve(1);
    auto inMatrix = input[0];
    if (inMatrix == nullptr || inMatrix->getRowSize() != weight.getColSize() ||
        inMatrix->getColSize() != 1) {
        LOG << "Mismatched or null input matrix";
        return NNMatrixPtrV{};
    }

    auto outMatrix =
        std::make_shared<NNMatrix>(forward(*inMatrix, NNFunctions::SigmoidFunc, false));
    outputs.push_back(outMatrix);

    return outputs;
}

void FCNNLayer::update(const NNMatrix& dw, const NNMatrix& db, float alpha, float momentum,
                       float weightDecay) {
    for (int i = 0; i < weight.getRowSize(); i++) {
        for (int j = 0; j < weight.getColSize(); j++) {
            const float w = weight.get(i, j);
            const float g = dw.get(i, j) + weightDecay * w;
            auto delta = momentum * vWeight.get(i, j) + alpha * g;
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

void FCNNLayer::dump() {
    std::stringstream ss("\n");
    ss << "Layer input size " << getInputSize() << ", output size " << getOutputSize() << std::endl;
    LOG << ss.str();
    weight.dump(true);
    bias.dump(true);
}