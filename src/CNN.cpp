#include "CNN.h"

#include "ConvolutionLayer.h"
#include "FCNNLayer.h"
#include "MaxPoolingLayer.h"
#include "NNUtils.h"
#include "nnlog/nnlog.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <memory>

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
    default:
        LOG << "Unsupported layer type " << static_cast<int>(config.getType());
        break;
    }

    return layerPtr;
}

NNMatrixPtrV CNN::forward(int epoc, int batchNo, int inChannelSize, const NNMatrixPtrV& X,
                          LayerCallback layerCallback) {
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
    outputs.reserve(sampleCount);

    // Cache layer outputs (conv/pool/fc) for backward().
    //
    // Shape note:
    // - Outer index is layer.
    // - Next index is sample.
    // - Innermost vector is that sample's per-channel feature maps.
    //
    // Not every sample is guaranteed to successfully produce outputs for every layer, so
    // downstream code must treat missing/empty entries as invalid for gradient updates.
    layerOutputs_.assign(layers.size(), NNMatrixPtrVV{});
    for (size_t li = 0; li < layers.size(); ++li) {
        layerOutputs_[li].reserve(sampleCount);
    }

    // Cache FC layer inputs/outputs for backward(). Each FC layer stores one flattened input and
    // one output matrix per sample (when that sample reaches the layer).
    int fcLayerCount = 0;
    for (const auto& layer : layers) {
        if (layer && layer->getLayerType() == NNLayerType::FullyConnected) {
            fcLayerCount += 1;
        }
    }

    fcLayerInputs_.assign(fcLayerCount, NNMatrixPtrV{});
    fcLayerOutputs_.assign(fcLayerCount, NNMatrixPtrV{});
    for (int i = 0; i < fcLayerCount; ++i) {
        fcLayerInputs_[i].reserve(sampleCount);
        fcLayerOutputs_[i].reserve(sampleCount);
    }

    for (size_t s = 0; s < sampleCount; ++s) {
        NNMatrixPtrV cur;
        cur.reserve(inChannelSize);

        const size_t base = s * inChannelSize;
        for (int c = 0; c < inChannelSize; ++c) {
            cur.push_back(X[base + c]);
        }

        bool sampleOk = true;
        int fcIdx = 0;
        size_t cachedLayerCount = 0;
        for (size_t li = 0; li < layers.size(); ++li) {
            auto& layer = layers[li];
            const bool isLastLayer = (li + 1 == layers.size());

            if (layerCallback) {
                layerCallback(epoc, batchNo, static_cast<int>(li), LayerPhase::Forward);
            }

            if (!layer) {
                LOG << "CNN::forward null layer";
                cur.clear();
                sampleOk = false;
                break;
            }

            switch (layer->getLayerType()) {
            case NNLayerType::Convolution: {
                auto* conv = static_cast<ConvolutionLayer*>(layer.get());
                cur = conv->forward(cur);
                break;
            }
            case NNLayerType::Pooling: {
                auto* pool = static_cast<MaxPoolingLayer*>(layer.get());
                cur = pool->forward(cur);
                break;
            }
            case NNLayerType::FullyConnected: {
                auto* fc = static_cast<FCNNLayer*>(layer.get());
                NNMatrixPtr flat;
                // Fast path: after the first FC, `cur` is already a single column vector.
                if (cur.size() == 1 && cur[0] && cur[0]->getColSize() == 1 &&
                    cur[0]->getRowSize() == fc->getInputSize()) {
                    flat = cur[0];
                } else {
                    flat = NNUtils::flattenAndConcat(cur);
                }
                if (!flat) {
                    LOG << "CNN::forward failed to flatten/concat inputs for FC layer";
                    cur.clear();
                    sampleOk = false;
                    break;
                }

                if (fcIdx < 0 || fcIdx >= fcLayerCount) {
                    LOG << "CNN::forward FC cache index out of range: fcIdx=" << fcIdx
                        << ", fcLayerCount=" << fcLayerCount;
                    cur.clear();
                    sampleOk = false;
                    break;
                }

                // Keep FC caches aligned by sample index: push exactly one entry per sample per
                // FC layer (either a real matrix or nullptr when the sample doesn't reach it).
                fcLayerInputs_[fcIdx].push_back(flat);
                NNMatrix out = !isLastLayer
                                   ? fc->forward(*flat, NNFunctions::ReLUFunc, false)
                                   : NNFunctions::softmax(fc->forward(*flat, nullptr, false));

                cur.clear();
                cur.reserve(1);
                auto outPtr = std::make_shared<NNMatrix>(std::move(out));
                cur.push_back(outPtr);
                fcLayerOutputs_[fcIdx].push_back(outPtr);
                fcIdx += 1;
                break;
            }
            default:
                LOG << "CNN::forward unsupported layer type";
                cur.clear();
                sampleOk = false;
                break;
            }

            if (cur.empty()) {
                sampleOk = false;
                break;
            }

            layerOutputs_[li].push_back(cur);
            cachedLayerCount = li + 1;
        }

        // Ensure conv/pool caches have exactly one entry for this sample.
        for (size_t li = cachedLayerCount; li < layers.size(); ++li) {
            layerOutputs_[li].push_back(NNMatrixPtrV{});
        }

        // Ensure FC caches have exactly one entry for this sample.
        for (int fi = fcIdx; fi < fcLayerCount; ++fi) {
            fcLayerInputs_[fi].push_back(nullptr);
            fcLayerOutputs_[fi].push_back(nullptr);
        }

        if (sampleOk && !cur.empty()) {
            outputs.push_back(cur[0]);
        } else {
            outputs.push_back(nullptr);
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
                   int epoc, int batchNo, int inChannelSize, LayerCallback layerCallback) {
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
        (void) forward(epoc, batchNo, inChannelSize, X, layerCallback);
        fcCachedCount = cachedFcSampleCount();
    }
    if (fcLayerInputs_.size() != fcLayerCount || fcLayerOutputs_.size() != fcLayerCount) {
        LOG << "CNN::backward: missing FC caches after forward()";
        return;
    }

    // Prepare convolution grads for this batch.
    for (auto* conv : convLayers) {
        if (conv) {
            conv->zeroGrad();
        }
    }

    // Accumulate gradients over the batch, like DNN::backward().
    std::vector<NNMatrix> dws;
    std::vector<NNMatrix> dbs;
    std::vector<NNMatrix> dzs;
    dws.reserve(fcLayers.size());
    dbs.reserve(fcLayers.size());
    dzs.reserve(fcLayers.size());
    for (int i = 0; i < fcLayerCount; ++i) {
        const int outSize = fcLayers[i]->getOutputSize();
        const int inSize = fcLayers[i]->getInputSize();
        dws.emplace_back(outSize, inSize);
        dbs.emplace_back(outSize, 1);
        dzs.emplace_back(outSize, 1);
    }

    int validFcSampleCount = 0;
    int validConvSampleCount = 0;

    const int outId = fcLayerCount - 1;

    const size_t loopCount = std::min(sampleCount, fcCachedCount);
    if (loopCount == 0) {
        LOG << "CNN::backward: empty FC cache";
        return;
    }

    for (size_t i = 0; i < loopCount; ++i) {
        if (!Y[i]) {
            LOG << "CNN::backward: null label at index " << i;
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

        validFcSampleCount += 1;

        // Backprop into conv/pool stack (if there is anything before the first FC layer).
        if (firstFcPos > 0) {
            bool convOk = true;

            if (layerOutputs_.size() != layers.size() ||
                layerOutputs_[firstFcPos - 1].size() <= i) {
                convOk = false;
            }

            if (convOk) {
                // Gradient w.r.t input of the first FC layer (flattened conv/pool activations).
                NNMatrix dFlat = fcLayers[0]->calculatePrevLayerDA(dzs[0]);
                if (dFlat.getColSize() != 1) {
                    convOk = false;
                }

                const auto& preFcOut = layerOutputs_[firstFcPos - 1][i];
                if (convOk && preFcOut.empty()) {
                    convOk = false;
                }

                int expectedFlat = 0;
                if (convOk) {
                    for (const auto& m : preFcOut) {
                        if (!m) {
                            convOk = false;
                            break;
                        }
                        expectedFlat += m->getRowSize() * m->getColSize();
                    }
                }
                if (convOk && expectedFlat != dFlat.getRowSize()) {
                    convOk = false;
                }

                if (convOk) {
                    // Un-flatten into per-channel maps.
                    NNMatrixPtrV dCur;
                    dCur.reserve(preFcOut.size());
                    const float* dFlatData = dFlat.data();
                    int offset = 0;
                    for (const auto& m : preFcOut) {
                        const int r = m->getRowSize();
                        const int c = m->getColSize();
                        const int len = r * c;
                        auto g = std::make_shared<NNMatrix>(r, c, 0.0f);
                        float* gData = g->data();
                        if (!dFlatData || !gData) {
                            convOk = false;
                            break;
                        }
                        std::copy(dFlatData + offset, dFlatData + offset + len, gData);
                        offset += len;
                        dCur.push_back(std::move(g));
                    }

                    // Sample input channels.
                    NNMatrixPtrV sampleInputs;
                    sampleInputs.reserve(inChannelSize);
                    const size_t base = i * inChannelSize;
                    for (int c = 0; c < inChannelSize; ++c) {
                        auto x = X[base + c];
                        if (!x) {
                            convOk = false;
                            break;
                        }
                        sampleInputs.push_back(std::move(x));
                    }

                    // Backprop through conv/pool layers in reverse.
                    if (convOk) {
                        for (int li = firstFcPos - 1; li >= 0; --li) {
                            if (layerCallback) {
                                layerCallback(epoc, batchNo, li, LayerPhase::Backward);
                            }
                            auto& layer = layers[li];
                            if (!layer) {
                                convOk = false;
                                break;
                            }
                            const auto& layerIn =
                                (li == 0) ? sampleInputs : layerOutputs_[li - 1][i];

                            if (layer->getLayerType() == NNLayerType::Pooling) {
                                auto* pool = static_cast<MaxPoolingLayer*>(layer.get());
                                dCur = pool->backward(layerIn, dCur);
                            } else if (layer->getLayerType() == NNLayerType::Convolution) {
                                auto* conv = static_cast<ConvolutionLayer*>(layer.get());
                                if (layerOutputs_[li].size() <= i) {
                                    convOk = false;
                                    break;
                                }
                                const auto& layerOut = layerOutputs_[li][i];
                                dCur = conv->backward(layerIn, layerOut, dCur);
                            } else {
                                convOk = false;
                                break;
                            }

                            if (dCur.empty()) {
                                convOk = false;
                                break;
                            }
                        }
                    }
                }
            }

            if (convOk) {
                validConvSampleCount += 1;
            }
        }
    }

    if (validFcSampleCount <= 0) {
        return;
    }

    // Average gradients.
    for (auto& dw : dws) {
        dw /= static_cast<float>(validFcSampleCount);
    }
    for (auto& db : dbs) {
        db /= static_cast<float>(validFcSampleCount);
    }

    // Update FC weights.
    for (int l = 0; l < fcLayerCount; ++l) {
        fcLayers[l]->update(dws[l], dbs[l], learningRate, momentum);
    }

    // Update convolution filters.
    if (validConvSampleCount > 0) {
        for (auto* conv : convLayers) {
            if (conv) {
                conv->applyGrad(validConvSampleCount, learningRate, momentum);
            }
        }
    }

    // NOTE: Gradients are propagated through max-pool and convolution layers, but there is no
    // bias term in ConvolutionLayer and the activation derivative is approximated from the
    // activated output (ReLU).
}

void CNN::train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate,
                float momentum) {
    train(dataSet, epochNum, batchSize, learningRate, momentum, nullptr, nullptr, nullptr, nullptr,
          nullptr);
}

void CNN::train(NNDataset& dataSet, int epochNum, int batchSize, float learningRate, float momentum,
                TrainCallback callback, LayerCallback layerCallback, BatchCallback batchCallback,
                StopCallback stopCallback, BatchStatsCallback batchStatsCallback) {
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
        batchX.reserve(static_cast<size_t>(batchSize) * static_cast<size_t>(std::max(1, inChannelSize)));
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

                for (int c = 0; c < inChannelSize; ++c) {
                    batchX.push_back(trainData[base + c]);
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
            auto preds = forward(e, b, inChannelSize, batchX, layerCallback);

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

            backward(batchX, batchY, curLearningRate, momentum, e, b, inChannelSize, layerCallback);
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

    auto preds = forward(epoc, 0, inChannelSize, x_test, nullptr);
    if (preds.size() != y_test.size()) {
        LOG << "CNN::accuracy pred/label mismatch: pred=" << preds.size()
            << ", label=" << y_test.size();
        return 0.0f;
    }

    return batchAccuracy(preds, y_test);
}
