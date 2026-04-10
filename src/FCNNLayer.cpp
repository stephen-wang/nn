#include "FCNNLayer.h"

#include "NNLog.h"

#include <iomanip>
#include <istream>
#include <ostream>
#include <sstream>

namespace {
bool writeMatrix(std::ostream& os, const NNMatrix& matrix) {
    const std::int32_t rows = matrix.getRowSize();
    const std::int32_t cols = matrix.getColSize();
    os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    if (!os.good() || rows <= 0 || cols <= 0) {
        return false;
    }
    const float* data = matrix.data();
    if (!data) {
        return false;
    }
    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    os.write(reinterpret_cast<const char*>(data),
             static_cast<std::streamsize>(count * sizeof(float)));
    return os.good();
}

bool readMatrix(std::istream& is, NNMatrix& matrix) {
    std::int32_t rows = 0;
    std::int32_t cols = 0;
    is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!is.good() || rows <= 0 || cols <= 0) {
        return false;
    }

    NNMatrix loaded(rows, cols, 0.0f);
    float* data = loaded.data();
    if (!data) {
        return false;
    }

    const std::size_t count = static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
    is.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count * sizeof(float)));
    if (!is.good()) {
        return false;
    }

    matrix = std::move(loaded);
    return true;
}
} // namespace

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
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < da.getRowSize(); i++) {
#else
    for (int i = 0; i < da.getRowSize(); i++) {
#endif
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
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < weight.getRowSize(); i++) {
#else
    for (int i = 0; i < weight.getRowSize(); i++) {
#endif
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

bool FCNNLayer::saveState(std::ostream& os) const {
    if (!writeMatrix(os, weight)) {
        return false;
    }
    if (!writeMatrix(os, vWeight)) {
        return false;
    }
    if (!writeMatrix(os, bias)) {
        return false;
    }
    if (!writeMatrix(os, vBias)) {
        return false;
    }
    return os.good();
}

bool FCNNLayer::loadState(std::istream& is) {
    NNMatrix newWeight(1, 1, 0.0f);
    NNMatrix newVWeight(1, 1, 0.0f);
    NNMatrix newBias(1, 1, 0.0f);
    NNMatrix newVBias(1, 1, 0.0f);

    if (!readMatrix(is, newWeight) || !readMatrix(is, newVWeight) || !readMatrix(is, newBias) ||
        !readMatrix(is, newVBias)) {
        return false;
    }

    if (!newWeight.hasSameDimension(weight) || !newVWeight.hasSameDimension(vWeight) ||
        !newBias.hasSameDimension(bias) || !newVBias.hasSameDimension(vBias)) {
        return false;
    }

    weight = std::move(newWeight);
    vWeight = std::move(newVWeight);
    bias = std::move(newBias);
    vBias = std::move(newVBias);
    return true;
}

void FCNNLayer::dump() {
    std::stringstream ss("\n");
    ss << "Layer input size " << getInputSize() << ", output size " << getOutputSize() << std::endl;
    LOG << ss.str();
    weight.dump(true);
    bias.dump(true);
}