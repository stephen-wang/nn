#include "CNN.h"

#include "BatchNormLayer.h"
#include "ConvolutionLayer.h"
#include "FCNNLayer.h"
#include "MaxPoolingLayer.h"
#include "DefaultCNNConfig.h"
#include "NNUtils.h"
#include "nnlog/nnlog.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <memory>
#include <random>

namespace {
std::mt19937& rng() {
    static thread_local std::mt19937 gen(std::random_device{}());
    return gen;
}

NNMatrixPtr augmentCifar32Channel(const NNMatrixPtr& in, int pad, int cropY, int cropX,
                                 bool hflip) {
    if (!in) {
        return nullptr;
    }
    const int side = in->getRowSize();
    if (side <= 0 || in->getColSize() != side) {
        return nullptr;
    }
    if (pad < 0) {
        return nullptr;
    }

    const float* inData = in->data();
    if (!inData) {
        return nullptr;
    }

    const int paddedSide = side + 2 * pad;
    auto padded = std::make_shared<NNMatrix>(paddedSide, paddedSide, 0.0f);
    float* padData = padded ? padded->data() : nullptr;
    if (!padData) {
        return nullptr;
    }

    for (int y = 0; y < side; ++y) {
        const float* src = inData + y * side;
        float* dst = padData + (y + pad) * paddedSide + pad;
        std::copy(src, src + side, dst);
    }

    cropY = std::max(0, std::min(cropY, 2 * pad));
    cropX = std::max(0, std::min(cropX, 2 * pad));

    auto out = std::make_shared<NNMatrix>(side, side, 0.0f);
    float* outData = out ? out->data() : nullptr;
    if (!outData) {
        return nullptr;
    }

    for (int y = 0; y < side; ++y) {
        const float* srcRow = padData + (y + cropY) * paddedSide + cropX;
        float* dstRow = outData + y * side;
        if (!hflip) {
            std::copy(srcRow, srcRow + side, dstRow);
        } else {
            for (int x = 0; x < side; ++x) {
                dstRow[x] = srcRow[side - 1 - x];
            }
        }
    }

    return out;
}

NNMatrixPtrV maybeAugmentCifar32Sample(const NNMatrixPtrV& sample, int pad) {
    if (!CIFAR100_CNN_USE_DATA_AUGMENTATION) {
        return sample;
    }
    if (static_cast<int>(sample.size()) != CIFAR100_CNN_IN_CHANNELS) {
        return sample;
    }
    if (!sample[0] || !sample[1] || !sample[2]) {
        return sample;
    }
    if (sample[0]->getRowSize() != 32 || sample[0]->getColSize() != 32 ||
        sample[1]->getRowSize() != 32 || sample[1]->getColSize() != 32 ||
        sample[2]->getRowSize() != 32 || sample[2]->getColSize() != 32) {
        return sample;
    }

    std::uniform_int_distribution<int> cropDist(0, std::max(0, 2 * pad));
    std::bernoulli_distribution flipDist(0.5);

    const int cropY = cropDist(rng());
    const int cropX = cropDist(rng());
    const bool hflip = flipDist(rng());

    NNMatrixPtrV out;
    out.reserve(sample.size());
    for (size_t c = 0; c < sample.size(); ++c) {
        auto aug = augmentCifar32Channel(sample[c], pad, cropY, cropX, hflip);
        if (!aug) {
            return sample;
        }
        out.push_back(std::move(aug));
    }
    return out;
}
} // namespace

CNN::CNN(const std::vector<CNNConfigPtr>& configs) {
    for (const auto& configPtr : configs) {
        if (configPtr == nullptr) {
            LOG << "Null CNNConfigPtr in configs";
            continue;
        }
        auto layerPtr = buildCNNLayer(*configPtr);
        layers.push_back(layerPtr);
    }
}

std::shared_ptr<NNLayer> CNN::buildCNNLayer(const CNNConfig& config) {
    std::shared_ptr<NNLayer> layerPtr(nullptr);
    switch (config.getType()) {
    case CNNLayerType::Convolution:
        if (const auto* convCfg = dynamic_cast<const ConvolutionLayerConfig*>(&config)) {
            layerPtr = std::make_unique<ConvolutionLayer>(*convCfg);
        } else {
            LOG << "Convolution layer requires ConvolutionLayerConfig";
        }
        break;
    case CNNLayerType::Pooling:
        if (const auto* poolingCfg = dynamic_cast<const MaxPoolingLayerConfig*>(&config)) {
            layerPtr = std::make_unique<MaxPoolingLayer>(*poolingCfg);
        } else {
            LOG << "Pooling layer requires MaxPoolingLayerConfig";
        }
        break;
    case CNNLayerType::FullyConnected:
        layerPtr = std::make_unique<FCNNLayer>(config.getInputSize(), config.getOutputSize());
        break;
    case CNNLayerType::BatchNorm:
        layerPtr = std::make_unique<BatchNormLayer>(config.getInputSize());
        break;
    default:
        LOG << "Unsupported layer type " << static_cast<int>(config.getType());
        break;
    }

    return layerPtr;
}

NNMatrixPtrV CNN::forward(int epoc, int batchNo, int inChannelSize, const NNMatrixPtrV& X,
                          bool training, LayerCallback layerCallback) {
    (void) epoc;
    (void) batchNo;

    NNMatrixPtrV outputs;
    if (inChannelSize <= 0) {
        LOG << "CNN::forward invalid inChannelSize " << inChannelSize;
        return outputs;
    }
    if (X.empty() || X.size() % inChannelSize != 0) {
        LOG << "CNN::forward input size " << X.size() << " not divisible by inChannelSize "
            << inChannelSize;
        return outputs;
    }

    const size_t sampleCount = X.size() / inChannelSize;
    outputs.assign(sampleCount, nullptr);

    // Build per-sample input channel vectors.
    std::vector<NNMatrixPtrV> curBySample(sampleCount);
    for (size_t s = 0; s < sampleCount; ++s) {
        NNMatrixPtrV cur;
        cur.reserve(inChannelSize);
        const size_t base = s * static_cast<size_t>(inChannelSize);
        for (int c = 0; c < inChannelSize; ++c) {
            cur.push_back(X[base + static_cast<size_t>(c)]);
        }
        curBySample[s] = std::move(cur);
    }

    // Cache layer outputs for backward(): [layer][sample][channel].
    layerOutputs_.assign(layers.size(), NNMatrixPtrVV{});
    for (size_t li = 0; li < layers.size(); ++li) {
        layerOutputs_[li].assign(sampleCount, NNMatrixPtrV{});
    }

    // Cache FC inputs/outputs for backward(): [fcLayerIdx][sample].
    int fcLayerCount = 0;
    for (const auto& layer : layers) {
        if (layer && layer->getLayerType() == NNLayerType::FullyConnected) {
            fcLayerCount += 1;
        }
    }
    fcLayerInputs_.assign(fcLayerCount, NNMatrixPtrV(sampleCount, nullptr));
    fcLayerOutputs_.assign(fcLayerCount, NNMatrixPtrV(sampleCount, nullptr));

    int fcIdx = 0;
    for (size_t li = 0; li < layers.size(); ++li) {
        auto& layer = layers[li];
        if (!layer) {
            LOG << "CNN::forward null layer";
            break;
        }

        if (layerCallback) {
            layerCallback(epoc, batchNo, static_cast<int>(li), LayerPhase::Forward);
        }

        switch (layer->getLayerType()) {
        case NNLayerType::Convolution: {
            auto* conv = static_cast<ConvolutionLayer*>(layer.get());
            for (size_t s = 0; s < sampleCount; ++s) {
                if (curBySample[s].empty()) {
                    continue;
                }
                curBySample[s] = conv->forward(curBySample[s]);
                layerOutputs_[li][s] = curBySample[s];
            }
            break;
        }
        case NNLayerType::Pooling: {
            auto* pool = static_cast<MaxPoolingLayer*>(layer.get());
            for (size_t s = 0; s < sampleCount; ++s) {
                if (curBySample[s].empty()) {
                    continue;
                }
                curBySample[s] = pool->forward(curBySample[s]);
                layerOutputs_[li][s] = curBySample[s];
            }
            break;
        }
        case NNLayerType::BatchNorm: {
            auto* bn = static_cast<BatchNormLayer*>(layer.get());
            curBySample = bn->forwardBatch(curBySample, training);
            for (size_t s = 0; s < sampleCount; ++s) {
                layerOutputs_[li][s] = curBySample[s];
            }
            break;
        }
        case NNLayerType::FullyConnected: {
            auto* fc = static_cast<FCNNLayer*>(layer.get());
            const bool isLastLayer = (li + 1 == layers.size());
            if (fcIdx < 0 || fcIdx >= fcLayerCount) {
                LOG << "CNN::forward FC cache index out of range: fcIdx=" << fcIdx
                    << ", fcLayerCount=" << fcLayerCount;
                break;
            }
            for (size_t s = 0; s < sampleCount; ++s) {
                if (curBySample[s].empty()) {
                    continue;
                }
                NNMatrixPtr flat;
                if (curBySample[s].size() == 1 && curBySample[s][0] &&
                    curBySample[s][0]->getColSize() == 1 &&
                    curBySample[s][0]->getRowSize() == fc->getInputSize()) {
                    flat = curBySample[s][0];
                } else {
                    flat = NNUtils::flattenAndConcat(curBySample[s]);
                }
                if (!flat) {
                    curBySample[s].clear();
                    continue;
                }

                fcLayerInputs_[fcIdx][s] = flat;
                NNMatrix out = !isLastLayer
                                   ? fc->forward(*flat, NNFunctions::ReLUFunc, false)
                                   : NNFunctions::softmax(fc->forward(*flat, nullptr, false));

                auto outPtr = std::make_shared<NNMatrix>(std::move(out));
                fcLayerOutputs_[fcIdx][s] = outPtr;
                curBySample[s].assign(1, outPtr);
                layerOutputs_[li][s] = curBySample[s];
            }
            fcIdx += 1;
            break;
        }
        default:
            LOG << "CNN::forward unsupported layer type";
            break;
        }
    }

    for (size_t s = 0; s < sampleCount; ++s) {
        if (curBySample[s].size() == 1 && curBySample[s][0]) {
            outputs[s] = curBySample[s][0];
        }
    }

    lastForwardOutputs_ = outputs;
    return outputs;
}

static NNMatrix calculateDW(const NNMatrix& input, const NNMatrix& dz) {
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

void CNN::backward(const NNMatrixPtrV& X, const NNMatrixPtrV& Y, float learningRate, float momentum,
                   float weightDecay, int epoc, int batchNo, int inChannelSize,
                   LayerCallback layerCallback) {
    (void) epoc;
    (void) batchNo;

    if (X.empty() || Y.empty()) {
        return;
    }
    if (inChannelSize <= 0) {
        LOG << "CNN::backward invalid inChannelSize " << inChannelSize;
        return;
    }
    if (X.size() % inChannelSize != 0) {
        LOG << "CNN::backward X.size() not divisible by inChannelSize: X=" << X.size()
            << ", channels=" << inChannelSize;
        return;
    }
    const size_t sampleCount = X.size() / inChannelSize;
    if (sampleCount != Y.size()) {
        LOG << "CNN::backward size mismatch: samples=" << sampleCount << ", labels=" << Y.size();
        return;
    }

    // Collect FC layers in order.
    std::vector<FCNNLayer*> fcLayers;
    fcLayers.reserve(layers.size());
    std::vector<ConvolutionLayer*> convLayers;
    convLayers.reserve(layers.size());
    int firstFcPos = -1;
    std::vector<int> fcLayerPositions;
    fcLayerPositions.reserve(layers.size());
    for (size_t li = 0; li < layers.size(); ++li) {
        if (!layers[li]) {
            continue;
        }
        if (firstFcPos < 0 && layers[li]->getLayerType() == NNLayerType::FullyConnected) {
            firstFcPos = static_cast<int>(li);
        }
        if (layers[li]->getLayerType() == NNLayerType::FullyConnected) {
            fcLayers.push_back(static_cast<FCNNLayer*>(layers[li].get()));
            fcLayerPositions.push_back(static_cast<int>(li));
        } else if (layers[li]->getLayerType() == NNLayerType::Convolution) {
            convLayers.push_back(static_cast<ConvolutionLayer*>(layers[li].get()));
        }
    }
    if (fcLayers.empty()) {
        LOG << "CNN::backward: no fully-connected layers to update";
        return;
    }

    // Ensure we have caches from forward(). If missing or clearly incomplete, re-run forward to
    // populate them.
    const int fcLayerCount = static_cast<int>(fcLayers.size());
    auto cachedFcSampleCount = [&]() -> size_t {
        if (static_cast<int>(fcLayerInputs_.size()) != fcLayerCount ||
            static_cast<int>(fcLayerOutputs_.size()) != fcLayerCount) {
            return 0u;
        }
        size_t count = sampleCount;
        for (int l = 0; l < fcLayerCount; ++l) {
            count = std::min(count, fcLayerInputs_[l].size());
            count = std::min(count, fcLayerOutputs_[l].size());
        }
        return count;
    };

    size_t fcCachedCount = cachedFcSampleCount();
    if (fcCachedCount == 0 || fcCachedCount < sampleCount) {
        (void) forward(epoc, batchNo, inChannelSize, X, true, layerCallback);
        fcCachedCount = cachedFcSampleCount();
    }
    if (fcLayerInputs_.size() != fcLayerCount || fcLayerOutputs_.size() != fcLayerCount) {
        LOG << "CNN::backward: missing FC caches after forward()";
        return;
    }

    // Prepare conv grads for this batch.
    for (auto* conv : convLayers) {
        if (conv) {
            conv->zeroGrad();
        }
    }

    // Accumulate FC gradients over valid samples.
    std::vector<NNMatrix> dws;
    std::vector<NNMatrix> dbs;
    std::vector<NNMatrix> dzs;
    dws.reserve(fcLayers.size());
    dbs.reserve(fcLayers.size());
    dzs.reserve(fcLayers.size());
    for (int i = 0; i < fcLayerCount; ++i) {
        dws.emplace_back(fcLayers[i]->getOutputSize(), fcLayers[i]->getInputSize());
        dbs.emplace_back(fcLayers[i]->getOutputSize(), 1);
        dzs.emplace_back(fcLayers[i]->getOutputSize(), 1);
    }

    const int outId = fcLayerCount - 1;
    const size_t loopCount = std::min(sampleCount, fcCachedCount);
    if (loopCount == 0) {
        LOG << "CNN::backward: empty FC cache";
        return;
    }

    std::vector<NNMatrixPtr> dFlatBySample(loopCount, nullptr);
    int validFcSampleCount = 0;

    for (size_t i = 0; i < loopCount; ++i) {
        if (!Y[i]) {
            continue;
        }

        bool fcOk = true;
        for (int l = 0; l < fcLayerCount; ++l) {
            if (fcLayerInputs_[l].size() <= i || fcLayerOutputs_[l].size() <= i ||
                !fcLayerInputs_[l][i] || !fcLayerOutputs_[l][i]) {
                fcOk = false;
                break;
            }
        }
        if (!fcOk) {
            continue;
        }

        // Output layer derivative (softmax + cross-entropy): dz = y_hat - y.
        if (layerCallback) {
            layerCallback(epoc, batchNo, fcLayerPositions[outId], LayerPhase::Backward);
        }
        dzs[outId] = *fcLayerOutputs_[outId][i] - *Y[i];
        dws[outId] += calculateDW(*fcLayerInputs_[outId][i], dzs[outId]);
        dbs[outId] += dzs[outId];

        // Hidden FC layers derivative (ReLU).
        for (int l = fcLayerCount - 2; l >= 0; --l) {
            if (layerCallback) {
                layerCallback(epoc, batchNo, fcLayerPositions[l], LayerPhase::Backward);
            }
            auto da = fcLayers[l + 1]->calculatePrevLayerDA(dzs[l + 1]);
            dzs[l] =
                da.elementProduct(fcLayerOutputs_[l][i]->applyFunction(NNFunctions::ReLUDrevative));
            dws[l] += calculateDW(*fcLayerInputs_[l][i], dzs[l]);
            dbs[l] += dzs[l];
        }

        // Gradient w.r.t. the input of the first FC layer.
        if (firstFcPos > 0) {
            NNMatrix dFlat = fcLayers[0]->calculatePrevLayerDA(dzs[0]);
            if (dFlat.getColSize() == 1) {
                dFlatBySample[i] = std::make_shared<NNMatrix>(std::move(dFlat));
            }
        }

        validFcSampleCount += 1;
    }

    if (validFcSampleCount <= 0) {
        return;
    }

    // Average FC gradients and update.
    for (auto& dw : dws) {
        dw /= static_cast<float>(validFcSampleCount);
    }
    for (auto& db : dbs) {
        db /= static_cast<float>(validFcSampleCount);
    }
    for (int l = 0; l < fcLayerCount; ++l) {
        fcLayers[l]->update(dws[l], dbs[l], learningRate, momentum, weightDecay);
    }

    // Backprop into conv/pool/bn stack.
    if (firstFcPos <= 0) {
        return;
    }
    const int preFcPos = firstFcPos - 1;
    if (preFcPos < 0 || static_cast<size_t>(preFcPos) >= layers.size()) {
        return;
    }
    if (layerOutputs_.size() != layers.size() ||
        layerOutputs_[static_cast<size_t>(preFcPos)].size() < loopCount) {
        return;
    }

    // Build per-sample inputs from flat X (for li==0 backprop).
    std::vector<NNMatrixPtrV> inputBySample(loopCount);
    for (size_t i = 0; i < loopCount; ++i) {
        NNMatrixPtrV sampleInputs;
        sampleInputs.reserve(inChannelSize);
        const size_t base = i * static_cast<size_t>(inChannelSize);
        for (int c = 0; c < inChannelSize; ++c) {
            sampleInputs.push_back(X[base + static_cast<size_t>(c)]);
        }
        inputBySample[i] = std::move(sampleInputs);
    }

    // Un-flatten dFlat into per-channel gradients matching the pre-FC activation.
    std::vector<NNMatrixPtrV> dCurBySample(loopCount);
    int validConvSampleCount = 0;
    for (size_t i = 0; i < loopCount; ++i) {
        if (!dFlatBySample[i]) {
            continue;
        }
        const auto& preFcOut = layerOutputs_[static_cast<size_t>(preFcPos)][i];
        if (preFcOut.empty()) {
            continue;
        }

        int expectedFlat = 0;
        for (const auto& m : preFcOut) {
            if (!m) {
                expectedFlat = -1;
                break;
            }
            expectedFlat += m->getRowSize() * m->getColSize();
        }
        if (expectedFlat <= 0 || expectedFlat != dFlatBySample[i]->getRowSize()) {
            continue;
        }

        NNMatrixPtrV dCur;
        dCur.reserve(preFcOut.size());
        const float* dFlatData = dFlatBySample[i]->data();
        int offset = 0;
        for (const auto& m : preFcOut) {
            const int r = m->getRowSize();
            const int c = m->getColSize();
            const int len = r * c;
            auto g = std::make_shared<NNMatrix>(r, c, 0.0f);
            float* gData = g ? g->data() : nullptr;
            if (!dFlatData || !gData) {
                dCur.clear();
                break;
            }
            std::copy(dFlatData + offset, dFlatData + offset + len, gData);
            offset += len;
            dCur.push_back(std::move(g));
        }
        if (!dCur.empty()) {
            dCurBySample[i] = std::move(dCur);
            validConvSampleCount += 1;
        }
    }

    if (validConvSampleCount <= 0) {
        return;
    }

    for (int li = preFcPos; li >= 0; --li) {
        if (layerCallback) {
            layerCallback(epoc, batchNo, li, LayerPhase::Backward);
        }
        auto& layer = layers[static_cast<size_t>(li)];
        if (!layer) {
            return;
        }

        if (layer->getLayerType() == NNLayerType::BatchNorm) {
            auto* bn = static_cast<BatchNormLayer*>(layer.get());
            dCurBySample = bn->backwardBatch(dCurBySample);
            bn->update(learningRate, momentum);
            continue;
        }

        for (size_t i = 0; i < loopCount; ++i) {
            if (dCurBySample[i].empty()) {
                continue;
            }

            const auto& layerIn =
                (li == 0) ? inputBySample[i] : layerOutputs_[static_cast<size_t>(li - 1)][i];

            if (layer->getLayerType() == NNLayerType::Pooling) {
                auto* pool = static_cast<MaxPoolingLayer*>(layer.get());
                dCurBySample[i] = pool->backward(layerIn, dCurBySample[i]);
            } else if (layer->getLayerType() == NNLayerType::Convolution) {
                auto* conv = static_cast<ConvolutionLayer*>(layer.get());
                const auto& layerOut = layerOutputs_[static_cast<size_t>(li)][i];
                dCurBySample[i] = conv->backward(layerIn, layerOut, dCurBySample[i]);
            } else {
                dCurBySample[i].clear();
            }
        }
    }

    for (auto* conv : convLayers) {
        if (conv) {
            conv->applyGrad(validConvSampleCount, learningRate, momentum, weightDecay);
        }
    }

    // NOTE: Gradients are propagated through max-pool and convolution layers, but there is no
    // bias term in ConvolutionLayer and the activation derivative is approximated from the
    // activated output (ReLU).
}

void CNN::train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate,
                float momentum) {
    train(dataSet, epochNum, batchSize, learningRate, momentum, 0.0f, nullptr, nullptr, nullptr,
          nullptr, nullptr);
}

void CNN::train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
                float weightDecay) {
    train(dataSet, epochNum, batchSize, learningRate, momentum, weightDecay, nullptr, nullptr,
          nullptr, nullptr, nullptr);
}

void CNN::train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
                TrainCallback callback, LayerCallback layerCallback, BatchCallback batchCallback,
                StopCallback stopCallback, BatchStatsCallback batchStatsCallback) {
    train(dataSet, epochNum, batchSize, learningRate, momentum, 0.0f, callback, layerCallback,
          batchCallback, stopCallback, batchStatsCallback);
}

void CNN::train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
                float weightDecay, TrainCallback callback, LayerCallback layerCallback,
                BatchCallback batchCallback, StopCallback stopCallback,
                BatchStatsCallback batchStatsCallback) {
    constexpr int kLogEveryNBatches = 50;

    // Simple step LR schedule: decay by 10x at ~1/3 and ~2/3 of training.
    // For short runs (e.g. 9 epochs), this prevents mid-run overshoot/decay in accuracy.
    const int lrStep1 = std::max(1, epochNum / 3);
    const int lrStep2 = std::max(lrStep1 + 1, (epochNum * 2) / 3);

    int e = 0;
    while (e < epochNum) {
        if (stopCallback && stopCallback()) {
            return;
        }

        float curLearningRate = learningRate;
        if (e >= lrStep2) {
            curLearningRate *= 0.01f;
        } else if (e >= lrStep1) {
            curLearningRate *= 0.1f;
        }

        LOG << "Epoc " << e << "/" << epochNum << ", trainData " << dataSet.trainInput_.size()
            << ", trainLabel " << dataSet.trainLabel_.size() << ", lr " << curLearningRate
            << std::endl;
        auto& trainData = dataSet.trainInput_;
        auto& trainLabel = dataSet.trainLabel_;

        const auto shuffleInfo = NNUtils::shuffleSamples(trainData, trainLabel);
        const int inChannelSize = shuffleInfo.inChannelSize;
        const int sampleCount = shuffleInfo.sampleCount;

        int numBatches = NNUtils::ceilDiv(sampleCount, batchSize);
        float epochLoss = 0.0f;

        std::vector<NNMatrixPtr> batchX;
        std::vector<NNMatrixPtr> batchY;
        batchX.reserve(static_cast<size_t>(batchSize) *
                       static_cast<size_t>(std::max(1, inChannelSize)));
        batchY.reserve(static_cast<size_t>(batchSize));

        for (int b = 0; b < numBatches; b++) {
            if (stopCallback && stopCallback()) {
                return;
            }
            // if (b % 200 == 0) {
            //     LOG << "Epoc " << e << ", batch " << b << " starts" << std::endl;
            // }

            const int startSample = b * batchSize;
            const int endSample = std::min(startSample + batchSize, sampleCount);
            if (b % kLogEveryNBatches == 0) {
                LOG << "Epoc " << e << ", batch " << b << " starts, startSample " << startSample
                    << ", endSample " << endSample << std::endl;
            }
            if (startSample >= endSample || endSample > sampleCount) {
                LOG << "CNN::train: no data in current batch" << std::endl;
                break;
            }

            const int batchSampleCount = endSample - startSample;
            const int expectedBatchXCount = batchSampleCount * inChannelSize;

            batchX.clear();
            batchY.clear();
            batchX.reserve(static_cast<size_t>(expectedBatchXCount));
            batchY.reserve(static_cast<size_t>(batchSampleCount));

            for (int i = startSample; i < endSample; ++i) {
                const int base = i * inChannelSize;
                const int last = base + inChannelSize;
                if (base < 0 || last > static_cast<int>(trainData.size())) {
                    LOG << "CNN::train batch build out-of-range: sample=" << i << ", base=" << base
                        << ", inChannelSize=" << inChannelSize
                        << ", trainData.size()=" << trainData.size();
                    batchX.clear();
                    batchY.clear();
                    break;
                }

                NNMatrixPtrV sample;
                sample.reserve(inChannelSize);
                for (int c = 0; c < inChannelSize; ++c) {
                    sample.push_back(trainData[base + c]);
                }

                // CIFAR-style augmentation (only if sample looks like 32x32 RGB).
                if (inChannelSize == CIFAR100_CNN_IN_CHANNELS) {
                    sample = maybeAugmentCifar32Sample(sample, CIFAR100_CNN_AUGMENT_PAD);
                }

                if (static_cast<int>(sample.size()) != inChannelSize) {
                    batchX.clear();
                    batchY.clear();
                    break;
                }

                for (int c = 0; c < inChannelSize; ++c) {
                    batchX.push_back(sample[static_cast<size_t>(c)]);
                }
                if (!trainLabel.empty()) {
                    if (i < 0 || i >= static_cast<int>(trainLabel.size())) {
                        LOG << "CNN::train label out-of-range: sample=" << i
                            << ", trainLabel.size()=" << trainLabel.size();
                        batchX.clear();
                        batchY.clear();
                        break;
                    }
                    batchY.push_back(trainLabel[i]);
                }
            }

            if (batchX.empty()) {
                continue;
            }
            if (static_cast<int>(batchX.size()) != expectedBatchXCount) {
                LOG << "CNN::train batchX incomplete: got=" << batchX.size()
                    << ", expected=" << expectedBatchXCount << ", inChannelSize=" << inChannelSize;
                continue;
            }
            if (!trainLabel.empty() && static_cast<int>(batchY.size()) != batchSampleCount) {
                LOG << "CNN::train batchY incomplete: got=" << batchY.size()
                    << ", expected=" << batchSampleCount;
                continue;
            }

            // Avoid per-batch spam; forward/backward are the hot path.
            auto preds = forward(e, b, inChannelSize, batchX, true, layerCallback);

            const size_t expectedBatchXCountSz = size_t(batchSampleCount) * size_t(inChannelSize);
            if (batchCallback && !preds.empty() && batchX.size() >= expectedBatchXCountSz) {
                int chosen = -1;
                for (int i = 0; i < batchSampleCount; ++i) {
                    if (preds[i]) {
                        chosen = i;
                        break;
                    }
                }
                if (chosen >= 0) {
                    NNMatrixPtrV sampleIn;
                    sampleIn.reserve(inChannelSize);
                    const int base = chosen * inChannelSize;
                    for (int c = 0; c < inChannelSize; ++c) {
                        sampleIn.push_back(batchX[base + c]);
                    }
                    batchCallback(e, b, sampleIn, *preds[chosen]);
                }
            }

            float batchLoss = loss(batchY);
            if (b % kLogEveryNBatches == 0) {
                LOG << "Batch loss " << batchLoss << std::endl;
            }
            epochLoss += batchLoss;

            if (batchStatsCallback) {
                int correct = 0;
                int valid = 0;
                for (int i = 0; i < batchSampleCount; ++i) {
                    auto p = preds[i];
                    auto y = batchY[i];
                    if (!p || !y) {
                        continue;
                    }
                    valid += 1;
                    if (argmax(*p) == argmax(*y)) {
                        correct += 1;
                    }
                }
                const float batchAcc =
                    valid > 0 ? (static_cast<float>(correct) / static_cast<float>(valid)) : 0.0f;
                const float epochAvgLoss =
                    (b + 1) > 0 ? (epochLoss / static_cast<float>(b + 1)) : 0.0f;
                batchStatsCallback(e + 1, epochNum, b + 1, numBatches, batchLoss, epochAvgLoss,
                                   batchAcc);
            }

            backward(batchX, batchY, curLearningRate, momentum, weightDecay, e, b, inChannelSize,
                     layerCallback);
            if (layerCallback) {
                layerCallback(e, b, -1, LayerPhase::Idle);
            }
        }

        float avgLoss = numBatches > 0 ? (epochLoss / static_cast<float>(numBatches)) : 0.0f;
        float acc = accuracy(e, dataSet.testInput_, dataSet.testLabel_);
        LOG << "Epoc " << e + 1 << "/" << epochNum << ", loss " << avgLoss << ", acc "
            << std::setprecision(3) << acc * 100;
        if (callback) {
            callback(e + 1, epochNum, avgLoss, acc);
        }
        e++;
    }
}

float CNN::loss(NNMatrixPtrV& Y) {
    if (Y.empty()) {
        return 0.0f;
    }

    if (lastForwardOutputs_.size() != Y.size()) {
        LOG << "CNN::loss size mismatch: pred=" << lastForwardOutputs_.size()
            << ", label=" << Y.size();
        return 0.0f;
    }

    return batchCrossEntropyLoss(lastForwardOutputs_, Y);
}

float CNN::accuracy(int epoc, const NNMatrixPtrV& x_test, const NNMatrixPtrV& y_test) {
    if (y_test.empty() || x_test.empty()) {
        return 0.0f;
    }

    const int labelCount = static_cast<int>(y_test.size());
    const int dataCount = static_cast<int>(x_test.size());

    int inChannelSize = 1;
    if (labelCount > 0 && dataCount > 0 && (dataCount % labelCount) == 0) {
        inChannelSize = dataCount / labelCount;
    }
    if (inChannelSize <= 0) {
        inChannelSize = 1;
    }

    if (dataCount != labelCount * inChannelSize) {
        LOG << "CNN::accuracy size mismatch: x_test=" << dataCount << ", y_test=" << labelCount
            << ", inferred channels=" << inChannelSize;
        return 0.0f;
    }

    auto preds = forward(epoc, 0, inChannelSize, x_test, false, nullptr);
    if (preds.size() != y_test.size()) {
        LOG << "CNN::accuracy pred/label mismatch: pred=" << preds.size()
            << ", label=" << y_test.size();
        return 0.0f;
    }

    return batchAccuracy(preds, y_test);
}
