#include "ConvolutionLayer.h"

#include "NNUtils.h"

#include <cstring>
#include <memory>

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayerConfig& config)
    : NNLayer(), inChannelSize(config.getInChannelSize()),
      outChannelSize(config.getOutChannelSize()), filterSize(config.getKernelSize()),
      stride(config.getStride()), padding(config.getPadding()) {

    const int filterNum = inChannelSize * outChannelSize;
    filters.reserve(filterNum);
    int inputSize = inChannelSize * filterSize * filterSize;
    int outputSize = outChannelSize * filterSize * filterSize;
    for (int i = 0; i < filterNum; i++) {
        auto filter = std::make_shared<NNMatrix>(filterSize, filterSize);
        for (int i = 0; i < filterSize; i++) {
            for (int j = 0; j < filterSize; j++) {
                filter->set(i, j, NNUtils::xavierInit(inputSize, outputSize));
            }
        }
        filters.push_back(filter);
    }
}

ConvolutionLayer::~ConvolutionLayer() {
    filters.clear();
}

NNMatrixPtrVector ConvolutionLayer::forward(const NNMatrixPtrVector& input) {
    NNMatrixPtrVector ret;
    if (input.size() != inChannelSize || input.empty() || !input[0] ||
        input[0]->getRowSize() + 2 * padding < filterSize ||
        input[0]->getColSize() + 2 * padding < filterSize) {
        LOG << "Mismatched input data: expected input channel " << inChannelSize << ", actual "
            << input.size() << ", input size " << input[0]->getRowSize() << "x"
            << input[0]->getColSize();
        return ret;
    }

    ret.reserve(inChannelSize * outChannelSize);
    for (auto inChannel = 0; inChannel < inChannelSize; inChannel++) {
        auto inputData = input[inChannel];
        for (auto outChannel = 0; outChannel < outChannelSize; outChannel++) {
            auto& filter = filters[inChannel * outChannelSize + outChannel];
            auto convResult = convolve(inputData, filter);
            ret.push_back(convResult);
        }
    }

    return ret;
}

NNMatrixPtr ConvolutionLayer::convolve(NNMatrixPtr input, NNMatrixPtr filter) {
    if (!input || !filter) {
        return nullptr;
    }

    if (stride <= 0 || filterSize <= 0) {
        LOG << "Invalid stride/filterSize " << stride << "/" << filterSize;
        return nullptr;
    }

    NNMatrixPtr padInput = (padding > 0) ? zeroPad(input) : input;
    const int padRowSize = padInput->getRowSize();
    const int padColSize = padInput->getColSize();
    if (padRowSize < filterSize || padColSize < filterSize) {
        LOG << "Invalid convolution input size " << padRowSize << "x" << padColSize << ", filter "
            << filterSize << "x" << filterSize;
        return nullptr;
    }

    const int outRowSize = NNUtils::ceilDiv(padRowSize - filterSize + 1, stride);
    const int outColSize = NNUtils::ceilDiv(padColSize - filterSize + 1, stride);
    if (outRowSize <= 0 || outColSize <= 0) {
        return nullptr;
    }

    NNMatrixPtr ret = std::make_shared<NNMatrix>(outRowSize, outColSize);
    const float* inData = padInput->data();
    const float* filterData = filter->data();
    float* outData = ret->data();

    for (int outI = 0, i = 0; outI < outRowSize; outI++, i += stride) {
        for (int outJ = 0, j = 0; outJ < outColSize; outJ++, j += stride) {
            float sum = 0.0f;
            for (int m = 0; m < filterSize; m++) {
                const int inBase = (i + m) * padColSize + j;
                const int filterBase = m * filterSize;
                for (int n = 0; n < filterSize; n++) {
                    sum += inData[inBase + n] * filterData[filterBase + n];
                }
            }
            outData[outI * outColSize + outJ] = sum;
        }
    }

    return ret;
}

NNMatrixPtr ConvolutionLayer::zeroPad(NNMatrixPtr input) {
    const int row = input->getRowSize();
    const int col = input->getColSize();
    const int outRow = row + 2 * padding;
    const int outCol = col + 2 * padding;

    NNMatrixPtr result = std::make_shared<NNMatrix>(outRow, outCol);
    const float* inData = input->data();
    float* outData = result->data();
    for (int i = 0; i < row; i++) {
        float* outRowPtr = outData + (i + padding) * outCol + padding;
        const float* inRowPtr = inData + i * col;
        std::memcpy(outRowPtr, inRowPtr, static_cast<size_t>(col) * sizeof(float));
    }

    return result;
}
