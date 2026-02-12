#include "ConvolutionLayer.h"

#include "NNUtils.h"

#include <algorithm>
#include <cstring>
#include <memory>

ConvolutionLayer::ConvolutionLayer(const ConvolutionLayerConfig& config)
    : NNLayer(NNLayerType::Convolution), inChannelSize(config.getInChannelSize()),
      outChannelSize(config.getOutChannelSize()), filterSize(config.getKernelSize()),
      stride(config.getStride()), padding(config.getPadding()),
      actFunction(config.getActFunction()) {

    if (inChannelSize <= 0 || outChannelSize <= 0 || filterSize <= 0) {
        LOG << "Invalid convolution config: inChannel/outChannel/kernel " << inChannelSize << "/"
            << outChannelSize << "/" << filterSize;
        return;
    }

    if (stride <= 0) {
        LOG << "Invalid convolution config: stride " << stride;
        return;
    }

    if (padding < 0) {
        LOG << "Invalid convolution config: padding " << padding;
        return;
    }

    const int filterNum = inChannelSize * outChannelSize;
    filters.reserve(filterNum);
    vFilters.reserve(filterNum);
    int inputSize = inChannelSize * filterSize * filterSize;
    int outputSize = outChannelSize * filterSize * filterSize;
    for (int filterIdx = 0; filterIdx < filterNum; filterIdx++) {
        auto filter = std::make_shared<NNMatrix>(filterSize, filterSize);
        for (int r = 0; r < filterSize; r++) {
            for (int c = 0; c < filterSize; c++) {
                filter->set(r, c, NNUtils::xavierInit(inputSize, outputSize));
            }
        }
        filters.push_back(filter);

        auto vFilter = std::make_shared<NNMatrix>(filterSize, filterSize, 0.0f);
        vFilters.push_back(std::move(vFilter));
    }

    // Bias per output channel.
    bias.assign(static_cast<size_t>(outChannelSize), 0.0f);
    vBias.assign(static_cast<size_t>(outChannelSize), 0.0f);
}

ConvolutionLayer::~ConvolutionLayer() {
    filters.clear();
    vFilters.clear();
    gradFilters.clear();
    bias.clear();
    vBias.clear();
    gradBias.clear();
}

void ConvolutionLayer::zeroGrad() {
    const size_t filterNum = filters.size();
    gradFilters.assign(filterNum, nullptr);
    for (size_t i = 0; i < filterNum; ++i) {
        gradFilters[i] = std::make_shared<NNMatrix>(filterSize, filterSize, 0.0f);
    }

    gradBias.assign(static_cast<size_t>(outChannelSize), 0.0f);
}

void ConvolutionLayer::applyGrad(int batchSize, float learningRate, float momentum) {
    if (batchSize <= 0) {
        LOG << "ConvolutionLayer::applyGrad invalid batchSize " << batchSize;
        return;
    }
    if (filters.size() != vFilters.size() || filters.size() != gradFilters.size()) {
        LOG << "ConvolutionLayer::applyGrad mismatched filter buffers";
        return;
    }
    if (bias.size() != static_cast<size_t>(outChannelSize) ||
        vBias.size() != static_cast<size_t>(outChannelSize) ||
        gradBias.size() != static_cast<size_t>(outChannelSize)) {
        LOG << "ConvolutionLayer::applyGrad mismatched bias buffers";
        return;
    }

    const float scale = learningRate / static_cast<float>(batchSize);
    for (size_t i = 0; i < filters.size(); ++i) {
        auto& wMat = filters[i];
        auto& vMat = vFilters[i];
        auto& gMat = gradFilters[i];
        if (!wMat || !vMat || !gMat) {
            LOG << "ConvolutionLayer::applyGrad null filter/velocity/grad";
            return;
        }

        float* w = wMat->data();
        float* v = vMat->data();
        const float* g = gMat->data();
        if (!w || !v || !g) {
            LOG << "ConvolutionLayer::applyGrad null data buffers";
            return;
        }

        const int len = filterSize * filterSize;
        for (int k = 0; k < len; ++k) {
            v[k] = momentum * v[k] + scale * g[k];
            w[k] -= v[k];
        }
    }

    for (int oc = 0; oc < outChannelSize; ++oc) {
        const size_t idx = static_cast<size_t>(oc);
        vBias[idx] = momentum * vBias[idx] + scale * gradBias[idx];
        bias[idx] -= vBias[idx];
    }
}

NNMatrixPtrV ConvolutionLayer::backward(const NNMatrixPtrV& inputs, const NNMatrixPtrV& outputs,
                                        const NNMatrixPtrV& dOutputs) {
    NNMatrixPtrV dInputs;
    ForwardParams params;
    if (!prepareForward(inputs, &params)) {
        return dInputs;
    }
    if (static_cast<int>(outputs.size()) != outChannelSize ||
        static_cast<int>(dOutputs.size()) != outChannelSize) {
        LOG << "ConvolutionLayer::backward channel mismatch: outputs=" << outputs.size()
            << ", dOutputs=" << dOutputs.size() << ", expected outChannels=" << outChannelSize;
        return dInputs;
    }
    if (gradFilters.size() != filters.size()) {
        LOG << "ConvolutionLayer::backward gradFilters not initialized; call zeroGrad()";
        return dInputs;
    }
    if (gradBias.size() != static_cast<size_t>(outChannelSize)) {
        LOG << "ConvolutionLayer::backward gradBias not initialized; call zeroGrad()";
        return dInputs;
    }

    const int inputH = params.inputHeight;
    const int inputW = params.inputWidth;
    const int padH = params.padHeight;
    const int padW = params.padWidth;

    // Prepare padded inputs and d(padded inputs)
    NNMatrixPtrV padInputs;
    padInputs.reserve(static_cast<size_t>(inChannelSize));
    NNMatrixPtrV dPadInputs;
    dPadInputs.reserve(static_cast<size_t>(inChannelSize));
    for (int ic = 0; ic < inChannelSize; ++ic) {
        auto padIn = inputs[static_cast<size_t>(ic)];
        if (padding > 0) {
            padIn = zeroPad(padIn);
        }
        if (!padIn) {
            LOG << "ConvolutionLayer::backward failed to pad input";
            return NNMatrixPtrV{};
        }
        padInputs.push_back(padIn);
        dPadInputs.push_back(std::make_shared<NNMatrix>(padH, padW, 0.0f));
    }

    // Validate output gradients shapes.
    const int outH = dOutputs[0] ? dOutputs[0]->getRowSize() : 0;
    const int outW = dOutputs[0] ? dOutputs[0]->getColSize() : 0;
    if (outH <= 0 || outW <= 0) {
        LOG << "ConvolutionLayer::backward invalid dOutput spatial size";
        return dInputs;
    }
    for (int oc = 0; oc < outChannelSize; ++oc) {
        if (!outputs[static_cast<size_t>(oc)] || !dOutputs[static_cast<size_t>(oc)]) {
            LOG << "ConvolutionLayer::backward null output/dOutput at outChannel " << oc;
            return NNMatrixPtrV{};
        }
        if (outputs[static_cast<size_t>(oc)]->getRowSize() != outH ||
            outputs[static_cast<size_t>(oc)]->getColSize() != outW ||
            dOutputs[static_cast<size_t>(oc)]->getRowSize() != outH ||
            dOutputs[static_cast<size_t>(oc)]->getColSize() != outW) {
            LOG << "ConvolutionLayer::backward mismatched output/dOutput sizes";
            return NNMatrixPtrV{};
        }
    }

    // Accumulate gradients.
    for (int oc = 0; oc < outChannelSize; ++oc) {
        const auto& outMap = outputs[oc];
        const auto& dOutMap = dOutputs[oc];

        for (int outI = 0; outI < outH; ++outI) {
            const int inIBase = outI * stride;
            for (int outJ = 0; outJ < outW; ++outJ) {
                const int inJBase = outJ * stride;

                float grad = dOutMap->get(outI, outJ);
                // Backprop through ReLU (approx using activated output, like python uses
                // conv_out>0)
                if (outMap->get(outI, outJ) <= 0.0f) {
                    grad = 0.0f;
                }
                if (grad == 0.0f) {
                    continue;
                }

                gradBias[oc] += grad;

                for (int ic = 0; ic < inChannelSize; ++ic) {
                    const size_t filterIdx = ic * outChannelSize + oc;
                    auto& filter = filters[filterIdx];
                    auto& dFilter = gradFilters[filterIdx];
                    if (!filter || !dFilter) {
                        LOG << "ConvolutionLayer::backward null filter/dFilter";
                        return NNMatrixPtrV{};
                    }

                    const auto& padIn = padInputs[ic];
                    auto& dPadIn = dPadInputs[ic];

                    for (int m = 0; m < filterSize; ++m) {
                        for (int n = 0; n < filterSize; ++n) {
                            const int inI = inIBase + m;
                            const int inJ = inJBase + n;
                            if (inI < 0 || inJ < 0 || inI >= padH || inJ >= padW) {
                                continue;
                            }

                            dFilter->set(m, n, dFilter->get(m, n) + padIn->get(inI, inJ) * grad);
                            dPadIn->set(inI, inJ, dPadIn->get(inI, inJ) + filter->get(m, n) * grad);
                        }
                    }
                }
            }
        }
    }

    // Crop padding from dPadInputs to get dInputs.
    dInputs.reserve(static_cast<size_t>(inChannelSize));
    for (int ic = 0; ic < inChannelSize; ++ic) {
        auto dIn = std::make_shared<NNMatrix>(inputH, inputW, 0.0f);
        const auto& dPad = dPadInputs[static_cast<size_t>(ic)];
        for (int i = 0; i < inputH; ++i) {
            for (int j = 0; j < inputW; ++j) {
                dIn->set(i, j, dPad->get(i + padding, j + padding));
            }
        }
        dInputs.push_back(std::move(dIn));
    }

    return dInputs;
}

bool ConvolutionLayer::prepareForward(const NNMatrixPtrV& inputs, ForwardParams* outParams) const {
    if (inputs.size() != static_cast<size_t>(inChannelSize)) {
        LOG << "Mismatched input channels: expected " << inChannelSize << ", actual "
            << inputs.size();
        return false;
    }

    for (int c = 0; c < inChannelSize; c++) {
        if (!inputs[c]) {
            LOG << "Null input channel at index " << c;
            return false;
        }
    }

    ForwardParams local;
    ForwardParams& params = outParams ? *outParams : local;

    params.inputHeight = inputs[0]->getRowSize();
    params.inputWidth = inputs[0]->getColSize();
    if (params.inputHeight <= 0 || params.inputWidth <= 0) {
        LOG << "Invalid input size " << params.inputHeight << "x" << params.inputWidth;
        return false;
    }

    // Ensure all input channels have the same spatial dimensions.
    for (int c = 1; c < inChannelSize; c++) {
        if (inputs[c]->getRowSize() != params.inputHeight ||
            inputs[c]->getColSize() != params.inputWidth) {
            LOG << "Mismatched input channel size at index " << c << ": expected "
                << params.inputHeight << "x" << params.inputWidth << ", actual "
                << inputs[c]->getRowSize() << "x" << inputs[c]->getColSize();
            return false;
        }
    }

    params.padHeight = params.inputHeight + 2 * padding;
    params.padWidth = params.inputWidth + 2 * padding;
    return true;
}

NNMatrixPtrV ConvolutionLayer::forward(const NNMatrixPtrV& inputs) {
    NNMatrixPtrV ret;
    if (!prepareForward(inputs, nullptr)) {
        return ret;
    }

    ret.reserve(outChannelSize);
    for (int outChannel = 0; outChannel < outChannelSize; outChannel++) {
        NNMatrixPtr outMap;

        for (int inChannel = 0; inChannel < inChannelSize; inChannel++) {
            auto& filter = filters[inChannel * outChannelSize + outChannel];
            auto convResult = convolve(inputs[inChannel], filter);
            if (!convResult) {
                LOG << "Convolution failed at inChannel=" << inChannel
                    << ", outChannel=" << outChannel;
                ret.clear();
                return ret;
            }

            if (!outMap) {
                outMap = std::make_shared<NNMatrix>(*convResult);
            } else {
                if (outMap->getRowSize() != convResult->getRowSize() ||
                    outMap->getColSize() != convResult->getColSize()) {
                    LOG << "Convolution result size mismatch for outChannel=" << outChannel
                        << ": expected " << outMap->getRowSize() << "x" << outMap->getColSize()
                        << ", actual " << convResult->getRowSize() << "x"
                        << convResult->getColSize();
                    ret.clear();
                    return ret;
                }
                *outMap += *convResult;
            }
        }

        if (outMap) {
            const float b = bias.size() == static_cast<size_t>(outChannelSize)
                                ? bias[static_cast<size_t>(outChannel)]
                                : 0.0f;
            if (b != 0.0f) {
                float* outData = outMap->data();
                if (!outData) {
                    LOG << "ConvolutionLayer::forward null output data";
                    ret.clear();
                    return ret;
                }
                const int len = outMap->getRowSize() * outMap->getColSize();
                for (int k = 0; k < len; ++k) {
                    outData[k] += b;
                }
            }
        }

        if (outMap && actFunction != nullptr) {
            outMap = std::make_shared<NNMatrix>(outMap->applyFunction(actFunction));
        }
        ret.push_back(outMap);
    }

    return ret;
}

bool ConvolutionLayer::prepareConvolution(const NNMatrixPtr& input, const NNMatrixPtr& filter,
                                          ConvolveParams& params) {
    if (!input || !filter) {
        LOG << "convolve: null input/filter";
        return false;
    }

    if (stride <= 0 || filterSize <= 0 || padding < 0) {
        LOG << "Invalid stride/filterSize/padding " << stride << "/" << filterSize << "/"
            << padding;
        return false;
    }

    if (filter->getRowSize() != filterSize || filter->getColSize() != filterSize) {
        LOG << "Filter size mismatch: expected " << filterSize << "x" << filterSize << ", actual "
            << filter->getRowSize() << "x" << filter->getColSize();
        return false;
    }

    params.filterData = filter->data();
    if (params.filterData == nullptr) {
        LOG << "convolve: null filter data";
        return false;
    }

    params.padInput = input;
    if (padding > 0) {
        params.padInput = zeroPad(input);
        if (!params.padInput) {
            LOG << "zeroPad failed";
            return false;
        }
    }

    params.inData = params.padInput->data();
    if (params.inData == nullptr) {
        LOG << "convolve: null input data";
        return false;
    }

    params.padRowSize = params.padInput->getRowSize();
    params.padColSize = params.padInput->getColSize();
    if (params.padRowSize < filterSize || params.padColSize < filterSize) {
        LOG << "Invalid convolution input size " << params.padRowSize << "x" << params.padColSize
            << ", filter " << filterSize << "x" << filterSize;
        return false;
    }

    params.outRowSize = NNUtils::ceilDiv(params.padRowSize - filterSize + 1, stride);
    params.outColSize = NNUtils::ceilDiv(params.padColSize - filterSize + 1, stride);
    if (params.outRowSize <= 0 || params.outColSize <= 0) {
        LOG << "Invalid output size " << params.outRowSize << "x" << params.outColSize;
        return false;
    }

    params.output = std::make_shared<NNMatrix>(params.outRowSize, params.outColSize);
    params.outData = params.output->data();
    if (params.outData == nullptr) {
        LOG << "convolve: null output data";
        return false;
    }

    return true;
}

NNMatrixPtr ConvolutionLayer::convolve(NNMatrixPtr input, NNMatrixPtr filter) {
    ConvolveParams params;
    if (!prepareConvolution(input, filter, params)) {
        return nullptr;
    }

    const int outRowSize = params.outRowSize;
    const int outColSize = params.outColSize;
    const int padColSize = params.padColSize;
    const float* inData = params.inData;
    const float* filterData = params.filterData;
    float* outData = params.outData;

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

    return params.output;
}

bool ConvolutionLayer::prepareZeroPad(const NNMatrixPtr& input, ZeroPadParams& params) const {
    if (!input) {
        LOG << "zeroPad: null input";
        return false;
    }

    if (padding < 0) {
        LOG << "zeroPad: invalid padding " << padding;
        return false;
    }

    params.height = input->getRowSize();
    params.width = input->getColSize();
    if (params.height <= 0 || params.width <= 0) {
        LOG << "zeroPad: invalid input size " << params.height << "x" << params.width;
        return false;
    }

    params.inData = input->data();
    if (params.inData == nullptr) {
        LOG << "zeroPad: null input data";
        return false;
    }

    params.outHeight = params.height + 2 * padding;
    params.outWidth = params.width + 2 * padding;
    if (params.outHeight <= 0 || params.outWidth <= 0) {
        LOG << "zeroPad: invalid output size " << params.outHeight << "x" << params.outWidth;
        return false;
    }

    return true;
}

NNMatrixPtr ConvolutionLayer::zeroPad(NNMatrixPtr input) {
    if (padding == 0) {
        return input;
    }

    ZeroPadParams params;
    if (!prepareZeroPad(input, params)) {
        return nullptr;
    }

    NNMatrixPtr result = std::make_shared<NNMatrix>(params.outHeight, params.outWidth);
    float* outData = result->data();
    if (outData == nullptr) {
        LOG << "zeroPad: null output data";
        return nullptr;
    }

    const size_t rowBytes = static_cast<size_t>(params.width) * sizeof(float);
    const size_t outStartOffset =
        static_cast<size_t>(padding) * static_cast<size_t>(params.outWidth) +
        static_cast<size_t>(padding);
    float* outRowPtr = outData + outStartOffset;
    const float* inRowPtr = params.inData;
    for (int i = 0; i < params.height; i++) {
        std::memcpy(outRowPtr, inRowPtr, rowBytes);
        outRowPtr += params.outWidth;
        inRowPtr += params.width;
    }

    return result;
}
