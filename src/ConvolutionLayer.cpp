#include "ConvolutionLayer.h"

#include "NNUtils.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <istream>
#include <memory>
#include <ostream>
#include <vector>

#if defined(NN_ENABLE_OMP)
#include <omp.h>
#endif

namespace {
bool writeFloatVector(std::ostream& os, const std::vector<float>& values) {
    const std::int32_t count = static_cast<std::int32_t>(values.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!os.good() || count < 0) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    os.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
    return os.good();
}

bool readFloatVector(std::istream& is, std::vector<float>& values) {
    std::int32_t count = 0;
    is.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!is.good() || count < 0) {
        return false;
    }
    values.assign(static_cast<std::size_t>(count), 0.0f);
    if (count == 0) {
        return true;
    }
    is.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    return is.good();
}
} // namespace

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
    const int fanIn = inChannelSize * filterSize * filterSize;
    for (int filterIdx = 0; filterIdx < filterNum; filterIdx++) {
        auto filter = std::make_shared<NNMatrix>(filterSize, filterSize);
        for (int r = 0; r < filterSize; r++) {
            for (int c = 0; c < filterSize; c++) {
                filter->set(r, c, NNUtils::heInit(fanIn));
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
    if (gradFilters.size() != filterNum) {
        gradFilters.assign(filterNum, nullptr);
        for (size_t i = 0; i < filterNum; ++i) {
            gradFilters[i] = std::make_shared<NNMatrix>(filterSize, filterSize, 0.0f);
        }
    } else {
        // Reuse and just zero memory.
        const int len = filterSize * filterSize;
        for (size_t i = 0; i < filterNum; ++i) {
            auto& g = gradFilters[i];
            if (!g) {
                g = std::make_shared<NNMatrix>(filterSize, filterSize, 0.0f);
                continue;
            }
            float* gData = g->data();
            if (gData) {
                std::fill_n(gData, len, 0.0f);
            }
        }
    }

    if (gradBias.size() != static_cast<size_t>(outChannelSize)) {
        gradBias.assign(static_cast<size_t>(outChannelSize), 0.0f);
    } else {
        std::fill(gradBias.begin(), gradBias.end(), 0.0f);
    }
}

void ConvolutionLayer::applyGrad(int batchSize, float learningRate, float momentum,
                                 float weightDecay) {
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

    const float scaleGrad = learningRate / static_cast<float>(batchSize);
    const float scaleWd = learningRate * weightDecay;
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
            v[k] = momentum * v[k] + scaleGrad * g[k] + scaleWd * w[k];
            w[k] -= v[k];
        }
    }

    for (int oc = 0; oc < outChannelSize; ++oc) {
        const size_t idx = static_cast<size_t>(oc);
        vBias[idx] = momentum * vBias[idx] + scaleGrad * gradBias[idx];
        bias[idx] -= vBias[idx];
    }
}

bool ConvolutionLayer::saveState(std::ostream& os) const {
    const std::int32_t inChannels = inChannelSize;
    const std::int32_t outChannels = outChannelSize;
    const std::int32_t kernel = filterSize;
    const std::int32_t convStride = stride;
    const std::int32_t convPadding = padding;
    const std::int32_t filterCount = static_cast<std::int32_t>(filters.size());

    os.write(reinterpret_cast<const char*>(&inChannels), sizeof(inChannels));
    os.write(reinterpret_cast<const char*>(&outChannels), sizeof(outChannels));
    os.write(reinterpret_cast<const char*>(&kernel), sizeof(kernel));
    os.write(reinterpret_cast<const char*>(&convStride), sizeof(convStride));
    os.write(reinterpret_cast<const char*>(&convPadding), sizeof(convPadding));
    os.write(reinterpret_cast<const char*>(&filterCount), sizeof(filterCount));
    if (!os.good() || filterCount < 0) {
        return false;
    }

    for (std::int32_t i = 0; i < filterCount; ++i) {
        const auto& w = filters[static_cast<std::size_t>(i)];
        const auto& v = vFilters[static_cast<std::size_t>(i)];
        if (!w || !v || !w->data() || !v->data()) {
            return false;
        }

        const std::int32_t rows = w->getRowSize();
        const std::int32_t cols = w->getColSize();
        os.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
        os.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
        if (!os.good() || rows <= 0 || cols <= 0 || v->getRowSize() != rows ||
            v->getColSize() != cols) {
            return false;
        }

        const std::size_t elemCount =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        os.write(reinterpret_cast<const char*>(w->data()),
                 static_cast<std::streamsize>(elemCount * sizeof(float)));
        os.write(reinterpret_cast<const char*>(v->data()),
                 static_cast<std::streamsize>(elemCount * sizeof(float)));
        if (!os.good()) {
            return false;
        }
    }

    return writeFloatVector(os, bias) && writeFloatVector(os, vBias) && os.good();
}

bool ConvolutionLayer::loadState(std::istream& is) {
    std::int32_t inChannels = 0;
    std::int32_t outChannels = 0;
    std::int32_t kernel = 0;
    std::int32_t convStride = 0;
    std::int32_t convPadding = 0;
    std::int32_t filterCount = 0;

    is.read(reinterpret_cast<char*>(&inChannels), sizeof(inChannels));
    is.read(reinterpret_cast<char*>(&outChannels), sizeof(outChannels));
    is.read(reinterpret_cast<char*>(&kernel), sizeof(kernel));
    is.read(reinterpret_cast<char*>(&convStride), sizeof(convStride));
    is.read(reinterpret_cast<char*>(&convPadding), sizeof(convPadding));
    is.read(reinterpret_cast<char*>(&filterCount), sizeof(filterCount));
    if (!is.good()) {
        return false;
    }

    if (inChannels != inChannelSize || outChannels != outChannelSize || kernel != filterSize ||
        convStride != stride || convPadding != padding ||
        filterCount != static_cast<std::int32_t>(filters.size())) {
        return false;
    }

    std::vector<NNMatrixPtr> newFilters;
    std::vector<NNMatrixPtr> newVFilters;
    newFilters.reserve(static_cast<std::size_t>(filterCount));
    newVFilters.reserve(static_cast<std::size_t>(filterCount));

    for (std::int32_t i = 0; i < filterCount; ++i) {
        std::int32_t rows = 0;
        std::int32_t cols = 0;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        if (!is.good() || rows <= 0 || cols <= 0) {
            return false;
        }

        auto w = std::make_shared<NNMatrix>(rows, cols, 0.0f);
        auto v = std::make_shared<NNMatrix>(rows, cols, 0.0f);
        float* wData = w ? w->data() : nullptr;
        float* vData = v ? v->data() : nullptr;
        if (!wData || !vData) {
            return false;
        }
        const std::size_t elemCount =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
        is.read(reinterpret_cast<char*>(wData),
                static_cast<std::streamsize>(elemCount * sizeof(float)));
        is.read(reinterpret_cast<char*>(vData),
                static_cast<std::streamsize>(elemCount * sizeof(float)));
        if (!is.good()) {
            return false;
        }

        newFilters.push_back(std::move(w));
        newVFilters.push_back(std::move(v));
    }

    std::vector<float> newBias;
    std::vector<float> newVBias;
    if (!readFloatVector(is, newBias) || !readFloatVector(is, newVBias)) {
        return false;
    }
    if (newBias.size() != static_cast<std::size_t>(outChannelSize) ||
        newVBias.size() != static_cast<std::size_t>(outChannelSize)) {
        return false;
    }

    filters = std::move(newFilters);
    vFilters = std::move(newVFilters);
    bias = std::move(newBias);
    vBias = std::move(newVBias);
    zeroGrad();
    return true;
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

    // Allocate dInputs directly (implicit padding avoids allocating padded buffers).
    dInputs.reserve(static_cast<size_t>(inChannelSize));
    for (int ic = 0; ic < inChannelSize; ++ic) {
        dInputs.push_back(std::make_shared<NNMatrix>(inputH, inputW, 0.0f));
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

    // Precompute convolution windows per output row/col once (shared across all channels).
    inIBaseByOutI_.resize(static_cast<size_t>(outH));
    mStartByOutI_.resize(static_cast<size_t>(outH));
    mEndByOutI_.resize(static_cast<size_t>(outH));
    for (int outI = 0; outI < outH; ++outI) {
        const int inIBase = outI * stride - padding;
        inIBaseByOutI_[static_cast<size_t>(outI)] = inIBase;
        mStartByOutI_[static_cast<size_t>(outI)] = std::max(0, -inIBase);
        mEndByOutI_[static_cast<size_t>(outI)] = std::min(filterSize, inputH - inIBase);
    }

    inJBaseByOutJ_.resize(static_cast<size_t>(outW));
    nStartByOutJ_.resize(static_cast<size_t>(outW));
    nEndByOutJ_.resize(static_cast<size_t>(outW));
    for (int outJ = 0; outJ < outW; ++outJ) {
        const int inJBase = outJ * stride - padding;
        inJBaseByOutJ_[static_cast<size_t>(outJ)] = inJBase;
        nStartByOutJ_[static_cast<size_t>(outJ)] = std::max(0, -inJBase);
        nEndByOutJ_[static_cast<size_t>(outJ)] = std::min(filterSize, inputW - inJBase);
    }

    // Accumulate gradients.
#if defined(NN_ENABLE_OMP)
    const int inLen = inputH * inputW;
    const int maxThreads = omp_get_max_threads();
    std::vector<float> threadDInputs(static_cast<size_t>(maxThreads) *
                                         static_cast<size_t>(inChannelSize) *
                                         static_cast<size_t>(inLen),
                                     0.0f);

#pragma omp parallel for
    for (int oc = 0; oc < outChannelSize; ++oc) {
        const int tid = omp_get_thread_num();
        float* threadBase =
            threadDInputs.data() + (static_cast<size_t>(tid) * static_cast<size_t>(inChannelSize) *
                                    static_cast<size_t>(inLen));

        const auto& outMap = outputs[oc];
        const auto& dOutMap = dOutputs[oc];

        const float* outData = outMap ? outMap->data() : nullptr;
        const float* dOutData = dOutMap ? dOutMap->data() : nullptr;
        if (!outData || !dOutData) {
            continue;
        }

        // Compute masked gradient once per output channel.
        const int outLen = outH * outW;
        thread_local std::vector<float> gradLocal;
        gradLocal.resize(static_cast<size_t>(outLen));

        float biasAcc = 0.0f;
        for (int idx = 0; idx < outLen; ++idx) {
            float grad = dOutData[idx];
            if (outData[idx] <= 0.0f) {
                grad = 0.0f;
            }
            gradLocal[static_cast<size_t>(idx)] = grad;
            biasAcc += grad;
        }

        // Each output channel has its own bias gradient; safe to write by index.
        gradBias[static_cast<size_t>(oc)] += biasAcc;

        for (int ic = 0; ic < inChannelSize; ++ic) {
            const size_t filterIdx = static_cast<size_t>(ic * outChannelSize + oc);
            auto& filter = filters[filterIdx];
            auto& dFilter = gradFilters[filterIdx];
            if (!filter || !dFilter) {
                continue;
            }

            const auto& in = inputs[static_cast<size_t>(ic)];
            const float* inData = in ? in->data() : nullptr;
            const float* filterData = filter->data();
            float* dFilterData = dFilter->data();
            float* dInAcc = threadBase + (static_cast<size_t>(ic) * static_cast<size_t>(inLen));
            if (!inData || !filterData || !dFilterData || !dInAcc) {
                continue;
            }

            for (int outI = 0; outI < outH; ++outI) {
                const int inIBase = inIBaseByOutI_[static_cast<size_t>(outI)];
                const int mStart = mStartByOutI_[static_cast<size_t>(outI)];
                const int mEnd = mEndByOutI_[static_cast<size_t>(outI)];
                if (mStart >= mEnd) {
                    continue;
                }

                const int outRowBase = outI * outW;
                for (int outJ = 0; outJ < outW; ++outJ) {
                    const int outIdx = outRowBase + outJ;
                    const float grad = gradLocal[static_cast<size_t>(outIdx)];
                    if (grad == 0.0f) {
                        continue;
                    }

                    const int inJBase = inJBaseByOutJ_[static_cast<size_t>(outJ)];
                    const int nStart = nStartByOutJ_[static_cast<size_t>(outJ)];
                    const int nEnd = nEndByOutJ_[static_cast<size_t>(outJ)];
                    if (nStart >= nEnd) {
                        continue;
                    }

                    for (int m = mStart; m < mEnd; ++m) {
                        const int inRowBase = (inIBase + m) * inputW + inJBase;
                        const int filterBase = m * filterSize;
                        for (int n = nStart; n < nEnd; ++n) {
                            const int inIdx = inRowBase + n;
                            const int fIdx = filterBase + n;
                            dFilterData[fIdx] += inData[inIdx] * grad;
                            dInAcc[inIdx] += filterData[fIdx] * grad;
                        }
                    }
                }
            }
        }
    }

    // Reduce thread-local dInputs accumulators into the output matrices.
    for (int ic = 0; ic < inChannelSize; ++ic) {
        auto& dIn = dInputs[static_cast<size_t>(ic)];
        float* dst = dIn ? dIn->data() : nullptr;
        if (!dst) {
            continue;
        }
        std::fill_n(dst, inLen, 0.0f);
        for (int t = 0; t < maxThreads; ++t) {
            const float* src = threadDInputs.data() +
                               (static_cast<size_t>(t) * static_cast<size_t>(inChannelSize) *
                                    static_cast<size_t>(inLen) +
                                static_cast<size_t>(ic) * static_cast<size_t>(inLen));
            for (int idx = 0; idx < inLen; ++idx) {
                dst[idx] += src[idx];
            }
        }
    }
#else
    gradBuf_.assign(static_cast<size_t>(outH) * static_cast<size_t>(outW), 0.0f);

    for (int oc = 0; oc < outChannelSize; ++oc) {
        const auto& outMap = outputs[oc];
        const auto& dOutMap = dOutputs[oc];

        const float* outData = outMap ? outMap->data() : nullptr;
        const float* dOutData = dOutMap ? dOutMap->data() : nullptr;
        if (!outData || !dOutData) {
            LOG << "ConvolutionLayer::backward null output/dOutput buffers";
            return NNMatrixPtrV{};
        }

        float biasAcc = 0.0f;
        const int outLen = outH * outW;
        for (int idx = 0; idx < outLen; ++idx) {
            float g = dOutData[idx];
            if (outData[idx] <= 0.0f) {
                g = 0.0f;
            }
            gradBuf_[static_cast<size_t>(idx)] = g;
            biasAcc += g;
        }
        gradBias[static_cast<size_t>(oc)] += biasAcc;

        for (int ic = 0; ic < inChannelSize; ++ic) {
            const size_t filterIdx = static_cast<size_t>(ic * outChannelSize + oc);
            auto& filter = filters[filterIdx];
            auto& dFilter = gradFilters[filterIdx];
            if (!filter || !dFilter) {
                LOG << "ConvolutionLayer::backward null filter/dFilter";
                return NNMatrixPtrV{};
            }

            const auto& in = inputs[static_cast<size_t>(ic)];
            auto& dIn = dInputs[static_cast<size_t>(ic)];
            const float* inData = in ? in->data() : nullptr;
            const float* filterData = filter->data();
            float* dFilterData = dFilter->data();
            float* dInData = dIn ? dIn->data() : nullptr;
            if (!inData || !filterData || !dFilterData || !dInData) {
                LOG << "ConvolutionLayer::backward null data buffers";
                return NNMatrixPtrV{};
            }

            for (int outI = 0; outI < outH; ++outI) {
                const int inIBase = inIBaseByOutI_[static_cast<size_t>(outI)];
                const int mStart = mStartByOutI_[static_cast<size_t>(outI)];
                const int mEnd = mEndByOutI_[static_cast<size_t>(outI)];
                if (mStart >= mEnd) {
                    continue;
                }

                const int outRowBase = outI * outW;
                for (int outJ = 0; outJ < outW; ++outJ) {
                    const float grad = gradBuf_[static_cast<size_t>(outRowBase + outJ)];
                    if (grad == 0.0f) {
                        continue;
                    }

                    const int inJBase = inJBaseByOutJ_[static_cast<size_t>(outJ)];
                    const int nStart = nStartByOutJ_[static_cast<size_t>(outJ)];
                    const int nEnd = nEndByOutJ_[static_cast<size_t>(outJ)];
                    if (nStart >= nEnd) {
                        continue;
                    }

                    for (int m = mStart; m < mEnd; ++m) {
                        const int inRowBase = (inIBase + m) * inputW + inJBase;
                        const int filterBase = m * filterSize;
                        for (int n = nStart; n < nEnd; ++n) {
                            const int inIdx = inRowBase + n;
                            const int fIdx = filterBase + n;
                            dFilterData[fIdx] += inData[inIdx] * grad;
                            dInData[inIdx] += filterData[fIdx] * grad;
                        }
                    }
                }
            }
        }
    }
#endif

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
    ForwardParams params;
    if (!prepareForward(inputs, &params)) {
        return ret;
    }

    const int inputH = params.inputHeight;
    const int inputW = params.inputWidth;
    // Output size matches the explicit padding formulation without allocating padded buffers.
    const int outRowSize = NNUtils::ceilDiv(inputH + 2 * padding - filterSize + 1, stride);
    const int outColSize = NNUtils::ceilDiv(inputW + 2 * padding - filterSize + 1, stride);
    if (outRowSize <= 0 || outColSize <= 0) {
        LOG << "ConvolutionLayer::forward invalid output size " << outRowSize << "x" << outColSize;
        return NNMatrixPtrV{};
    }

    // Precompute convolution windows per output row/col to avoid max/min work in hot loops.
    inIBaseByOutI_.resize(static_cast<size_t>(outRowSize));
    mStartByOutI_.resize(static_cast<size_t>(outRowSize));
    mEndByOutI_.resize(static_cast<size_t>(outRowSize));
    for (int outI = 0; outI < outRowSize; ++outI) {
        const int inIBase = outI * stride - padding;
        inIBaseByOutI_[static_cast<size_t>(outI)] = inIBase;
        mStartByOutI_[static_cast<size_t>(outI)] = std::max(0, -inIBase);
        mEndByOutI_[static_cast<size_t>(outI)] = std::min(filterSize, inputH - inIBase);
    }

    inJBaseByOutJ_.resize(static_cast<size_t>(outColSize));
    nStartByOutJ_.resize(static_cast<size_t>(outColSize));
    nEndByOutJ_.resize(static_cast<size_t>(outColSize));
    for (int outJ = 0; outJ < outColSize; ++outJ) {
        const int inJBase = outJ * stride - padding;
        inJBaseByOutJ_[static_cast<size_t>(outJ)] = inJBase;
        nStartByOutJ_[static_cast<size_t>(outJ)] = std::max(0, -inJBase);
        nEndByOutJ_[static_cast<size_t>(outJ)] = std::min(filterSize, inputW - inJBase);
    }

    ret.reserve(outChannelSize);

#if defined(NN_ENABLE_OMP)
    std::atomic<bool> hadError{false};
    ret.assign(static_cast<size_t>(outChannelSize), nullptr);
#pragma omp parallel for
    for (int outChannel = 0; outChannel < outChannelSize; outChannel++) {
#else
    for (int outChannel = 0; outChannel < outChannelSize; outChannel++) {
#endif
#if defined(NN_ENABLE_OMP)
        if (hadError.load(std::memory_order_relaxed)) {
            continue;
        }
#endif
        auto outMap = std::make_shared<NNMatrix>(outRowSize, outColSize, 0.0f);
        float* outAcc = outMap ? outMap->data() : nullptr;
        if (!outAcc) {
            LOG << "ConvolutionLayer::forward null output buffer";
#if defined(NN_ENABLE_OMP)
            hadError.store(true, std::memory_order_relaxed);
            continue;
#else
            return NNMatrixPtrV{};
#endif
        }

        for (int inChannel = 0; inChannel < inChannelSize; inChannel++) {
            auto& filter = filters[inChannel * outChannelSize + outChannel];
            if (!filter) {
                LOG << "ConvolutionLayer::forward null filter";
#if defined(NN_ENABLE_OMP)
                hadError.store(true, std::memory_order_relaxed);
                continue;
#else
                return NNMatrixPtrV{};
#endif
            }

            const auto& in = inputs[static_cast<size_t>(inChannel)];
            const float* inData = in ? in->data() : nullptr;
            const float* filterData = filter->data();
            if (!in || !inData || !filterData) {
                LOG << "ConvolutionLayer::forward null input/filter data";
#if defined(NN_ENABLE_OMP)
                hadError.store(true, std::memory_order_relaxed);
                continue;
#else
                return NNMatrixPtrV{};
#endif
            }

            for (int outI = 0; outI < outRowSize; ++outI) {
                const int inIBase = inIBaseByOutI_[static_cast<size_t>(outI)];
                const int mStart = mStartByOutI_[static_cast<size_t>(outI)];
                const int mEnd = mEndByOutI_[static_cast<size_t>(outI)];
                if (mStart >= mEnd) {
                    continue;
                }
                float* outRow = outAcc + outI * outColSize;
                for (int outJ = 0; outJ < outColSize; ++outJ) {
                    const int inJBase = inJBaseByOutJ_[static_cast<size_t>(outJ)];
                    const int nStart = nStartByOutJ_[static_cast<size_t>(outJ)];
                    const int nEnd = nEndByOutJ_[static_cast<size_t>(outJ)];
                    if (nStart >= nEnd) {
                        continue;
                    }

                    float sum = 0.0f;
                    for (int m = mStart; m < mEnd; ++m) {
                        const int inRowBase = (inIBase + m) * inputW + inJBase;
                        const int filterBase = m * filterSize;
                        for (int n = nStart; n < nEnd; ++n) {
                            sum += inData[inRowBase + n] * filterData[filterBase + n];
                        }
                    }

                    outRow[outJ] += sum;
                }
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
#if defined(NN_ENABLE_OMP)
                    hadError.store(true, std::memory_order_relaxed);
                    continue;
#else
                    ret.clear();
                    return ret;
#endif
                }
                const int len = outMap->getRowSize() * outMap->getColSize();
                for (int k = 0; k < len; ++k) {
                    outData[k] += b;
                }
            }
        }

        if (outMap && actFunction != nullptr) {
            outMap->applyFunctionInplace(actFunction);
        }

#if defined(NN_ENABLE_OMP)
        ret[static_cast<size_t>(outChannel)] = outMap;
#else
        ret.push_back(outMap);
#endif
    }

#if defined(NN_ENABLE_OMP)
    if (hadError.load(std::memory_order_relaxed)) {
        return NNMatrixPtrV{};
    }
    // If any thread failed to produce an output map, treat it as an error.
    for (int oc = 0; oc < outChannelSize; ++oc) {
        if (!ret[static_cast<size_t>(oc)]) {
            return NNMatrixPtrV{};
        }
    }
#endif

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
