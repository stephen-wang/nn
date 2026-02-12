#include "MaxPoolingLayer.h"

#include "NNUtils.h"

#include <algorithm>
#include <limits>

MaxPoolingLayer::MaxPoolingLayer(const MaxPoolingLayerConfig& config)
    : NNLayer(NNLayerType::Pooling), filterSize(config.getKernelSize()),
      stride(config.getStride()) {
}

NNMatrixPtrV MaxPoolingLayer::forward(const NNMatrixPtrV& inputs) {
    NNMatrixPtrV outputs;

    if (inputs.empty() || inputs[0] == nullptr) {
        LOG << "Empty inputs or null input[0]";
        return outputs;
    }

    if (filterSize <= 0 || stride <= 0) {
        LOG << "Invalid filterSize/stride " << filterSize << "/" << stride;
        return outputs;
    }

    const int inputHeight = inputs[0]->getRowSize();
    const int inputWidth = inputs[0]->getColSize();
    if (inputHeight <= 0 || inputWidth <= 0) {
        LOG << "Invalid input size " << inputHeight << "x" << inputWidth;
        return outputs;
    }

    if (inputHeight < filterSize || inputWidth < filterSize) {
        LOG << "Invalid pooling input size " << inputHeight << "x" << inputWidth << ", filter "
            << filterSize << "x" << filterSize;
        return outputs;
    }

    const int outHeight = NNUtils::ceilDiv(inputHeight - filterSize + 1, stride);
    const int outWidth = NNUtils::ceilDiv(inputWidth - filterSize + 1, stride);
    if (outHeight <= 0 || outWidth <= 0) {
        LOG << "Invalid output size " << outHeight << "x" << outWidth;
        return outputs;
    }

    outputs.reserve(inputs.size());

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& input = inputs[i];
        if (input == nullptr || input->getRowSize() != inputHeight ||
            input->getColSize() != inputWidth) {
            LOG << "Mismatched or null input channel at index " << i;
            outputs.clear();
            return outputs;
        }
        auto output = std::make_shared<NNMatrix>(outHeight, outWidth);
        for (int m = 0; m < outHeight; m++) {
            for (int n = 0; n < outWidth; n++) {
                int x = m * stride;
                int y = n * stride;
                output->set(m, n, input->getRegionMax(x, y, filterSize));
            }
        }
        outputs.push_back(output);
    }

    return outputs;
}

NNMatrixPtrV MaxPoolingLayer::backward(const NNMatrixPtrV& inputs, const NNMatrixPtrV& dOutputs) {
    NNMatrixPtrV dInputs;

    if (inputs.empty() || dOutputs.empty()) {
        LOG << "MaxPoolingLayer::backward empty inputs/dOutputs";
        return dInputs;
    }
    if (inputs.size() != dOutputs.size()) {
        LOG << "MaxPoolingLayer::backward channel mismatch: inputs=" << inputs.size()
            << ", dOutputs=" << dOutputs.size();
        return dInputs;
    }
    if (filterSize <= 0 || stride <= 0) {
        LOG << "MaxPoolingLayer::backward invalid filterSize/stride " << filterSize << "/"
            << stride;
        return dInputs;
    }

    const int inputHeight = inputs[0] ? inputs[0]->getRowSize() : 0;
    const int inputWidth = inputs[0] ? inputs[0]->getColSize() : 0;
    if (inputHeight <= 0 || inputWidth <= 0) {
        LOG << "MaxPoolingLayer::backward invalid input size";
        return dInputs;
    }

    if (inputHeight < filterSize || inputWidth < filterSize) {
        LOG << "MaxPoolingLayer::backward invalid input size " << inputHeight << "x" << inputWidth
            << ", filter " << filterSize << "x" << filterSize;
        return dInputs;
    }

    const int outHeight = dOutputs[0] ? dOutputs[0]->getRowSize() : 0;
    const int outWidth = dOutputs[0] ? dOutputs[0]->getColSize() : 0;
    if (outHeight <= 0 || outWidth <= 0) {
        LOG << "MaxPoolingLayer::backward invalid dOutput size";
        return dInputs;
    }

    const int expectedOutHeight = NNUtils::ceilDiv(inputHeight - filterSize + 1, stride);
    const int expectedOutWidth = NNUtils::ceilDiv(inputWidth - filterSize + 1, stride);
    if (outHeight != expectedOutHeight || outWidth != expectedOutWidth) {
        LOG << "MaxPoolingLayer::backward dOutput size mismatch: expected " << expectedOutHeight
            << "x" << expectedOutWidth << ", actual " << outHeight << "x" << outWidth;
        return dInputs;
    }

    dInputs.reserve(inputs.size());

    for (size_t c = 0; c < inputs.size(); ++c) {
        const auto& input = inputs[c];
        const auto& dOut = dOutputs[c];
        if (!input || !dOut) {
            LOG << "MaxPoolingLayer::backward null input/dOutput at channel " << c;
            dInputs.clear();
            return dInputs;
        }
        if (input->getRowSize() != inputHeight || input->getColSize() != inputWidth) {
            LOG << "MaxPoolingLayer::backward mismatched input size at channel " << c;
            dInputs.clear();
            return dInputs;
        }
        if (dOut->getRowSize() != outHeight || dOut->getColSize() != outWidth) {
            LOG << "MaxPoolingLayer::backward mismatched dOutput size at channel " << c;
            dInputs.clear();
            return dInputs;
        }

        auto dInput = std::make_shared<NNMatrix>(inputHeight, inputWidth, 0.0f);
        for (int m = 0; m < outHeight; ++m) {
            for (int n = 0; n < outWidth; ++n) {
                const int xBase = m * stride;
                const int yBase = n * stride;

                float maxVal = -std::numeric_limits<float>::infinity();
                int maxI = -1;
                int maxJ = -1;
                const int xEnd = std::min(inputHeight, xBase + filterSize);
                const int yEnd = std::min(inputWidth, yBase + filterSize);
                for (int i = xBase; i < xEnd; ++i) {
                    for (int j = yBase; j < yEnd; ++j) {
                        const float v = input->get(i, j);
                        if (maxI < 0 || v > maxVal) {
                            maxVal = v;
                            maxI = i;
                            maxJ = j;
                        }
                    }
                }

                if (maxI >= 0 && maxJ >= 0) {
                    dInput->set(maxI, maxJ, dInput->get(maxI, maxJ) + dOut->get(m, n));
                }
            }
        }

        dInputs.push_back(std::move(dInput));
    }

    return dInputs;
}
