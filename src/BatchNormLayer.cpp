#include "BatchNormLayer.h"

#include "nnlog/nnlog.h"

#include <algorithm>
#include <cmath>

BatchNormLayer::BatchNormLayer(int channels, float eps, float runningMomentum)
    : NNLayer(NNLayerType::BatchNorm), channels_(channels), eps_(eps),
      runningMomentum_(runningMomentum) {
    if (channels_ <= 0) {
                NNLOG_WARN("BatchNormLayer") << "invalid channels " << channels_;
        channels_ = 0;
        return;
    }

    gamma_.assign(static_cast<size_t>(channels_), 1.0f);
    beta_.assign(static_cast<size_t>(channels_), 0.0f);
    vGamma_.assign(static_cast<size_t>(channels_), 0.0f);
    vBeta_.assign(static_cast<size_t>(channels_), 0.0f);

    runningMean_.assign(static_cast<size_t>(channels_), 0.0f);
    runningVar_.assign(static_cast<size_t>(channels_), 1.0f);

    batchMean_.assign(static_cast<size_t>(channels_), 0.0f);
    batchInvStd_.assign(static_cast<size_t>(channels_), 1.0f);
    dGamma_.assign(static_cast<size_t>(channels_), 0.0f);
    dBeta_.assign(static_cast<size_t>(channels_), 0.0f);
    elemCountPerChannel_.assign(static_cast<size_t>(channels_), 0);
}

bool BatchNormLayer::validateBatchShape(const std::vector<NNMatrixPtrV>& batch) const {
    if (channels_ <= 0) {
        return false;
    }
    if (batch.empty()) {
        return false;
    }

    for (const auto& sample : batch) {
        if (sample.empty()) {
            continue; // allow empty samples; they will be ignored
        }
        if (static_cast<int>(sample.size()) != channels_) {
            return false;
        }
        for (int c = 0; c < channels_; ++c) {
            if (!sample[static_cast<size_t>(c)]) {
                return false;
            }
        }
    }

    return true;
}

std::vector<NNMatrixPtrV> BatchNormLayer::forwardBatch(const std::vector<NNMatrixPtrV>& batch,
                                                       bool training) {
    std::vector<NNMatrixPtrV> out;
    if (!validateBatchShape(batch)) {
        return out;
    }

    const size_t batchSize = batch.size();
    out.assign(batchSize, NNMatrixPtrV{});

    // Reset caches/grad buffers.
    std::fill(batchMean_.begin(), batchMean_.end(), 0.0f);
    std::fill(batchInvStd_.begin(), batchInvStd_.end(), 1.0f);
    std::fill(elemCountPerChannel_.begin(), elemCountPerChannel_.end(), 0);
    xhatBySample_.assign(batchSize, NNMatrixPtrV{});

    // Compute mean/var per channel over (batch * H * W).
    for (int c = 0; c < channels_; ++c) {
        double sum = 0.0;
        double sumSq = 0.0;
        int n = 0;

        for (size_t s = 0; s < batchSize; ++s) {
            const auto& sample = batch[s];
            if (sample.empty()) {
                continue;
            }
            const auto& m = sample[static_cast<size_t>(c)];
            if (!m) {
                continue;
            }
            const int len = m->getRowSize() * m->getColSize();
            const float* data = m->data();
            if (!data || len <= 0) {
                continue;
            }
            for (int i = 0; i < len; ++i) {
                const double v = static_cast<double>(data[i]);
                sum += v;
                sumSq += v * v;
            }
            n += len;
        }

        elemCountPerChannel_[static_cast<size_t>(c)] = n;

        if (!training) {
            // Use running stats.
            const float mean = runningMean_[static_cast<size_t>(c)];
            const float var = runningVar_[static_cast<size_t>(c)];
            batchMean_[static_cast<size_t>(c)] = mean;
            batchInvStd_[static_cast<size_t>(c)] = 1.0f / std::sqrt(std::max(0.0f, var) + eps_);
            continue;
        }

        if (n <= 0) {
            batchMean_[static_cast<size_t>(c)] = 0.0f;
            batchInvStd_[static_cast<size_t>(c)] = 1.0f;
            continue;
        }

        const float mean = static_cast<float>(sum / static_cast<double>(n));
        const double ex2 = sumSq / static_cast<double>(n);
        const double mu2 = static_cast<double>(mean) * static_cast<double>(mean);
        const float var = static_cast<float>(std::max(0.0, ex2 - mu2));

        batchMean_[static_cast<size_t>(c)] = mean;
        batchInvStd_[static_cast<size_t>(c)] = 1.0f / std::sqrt(var + eps_);

        // Update running stats.
        runningMean_[static_cast<size_t>(c)] =
            (1.0f - runningMomentum_) * runningMean_[static_cast<size_t>(c)] +
            runningMomentum_ * mean;
        runningVar_[static_cast<size_t>(c)] =
            (1.0f - runningMomentum_) * runningVar_[static_cast<size_t>(c)] +
            runningMomentum_ * var;
    }

    // Normalize and affine transform.
    for (size_t s = 0; s < batchSize; ++s) {
        const auto& sample = batch[s];
        if (sample.empty()) {
            continue;
        }

        NNMatrixPtrV sampleOut;
        sampleOut.reserve(static_cast<size_t>(channels_));
        NNMatrixPtrV sampleXhat;
        sampleXhat.reserve(static_cast<size_t>(channels_));

        for (int c = 0; c < channels_; ++c) {
            const auto& x = sample[static_cast<size_t>(c)];
            const int r = x->getRowSize();
            const int col = x->getColSize();
            const int len = r * col;

            auto y = std::make_shared<NNMatrix>(r, col, 0.0f);
            auto xhat = std::make_shared<NNMatrix>(r, col, 0.0f);
            float* yData = y ? y->data() : nullptr;
            float* xhatData = xhat ? xhat->data() : nullptr;
            const float* xData = x ? x->data() : nullptr;
            if (!yData || !xhatData || !xData || len <= 0) {
                sampleOut.clear();
                sampleXhat.clear();
                break;
            }

            const float mean = batchMean_[static_cast<size_t>(c)];
            const float invStd = batchInvStd_[static_cast<size_t>(c)];
            const float gamma = gamma_[static_cast<size_t>(c)];
            const float beta = beta_[static_cast<size_t>(c)];

            for (int i = 0; i < len; ++i) {
                const float xn = (xData[i] - mean) * invStd;
                xhatData[i] = xn;
                yData[i] = gamma * xn + beta;
            }

            sampleOut.push_back(std::move(y));
            sampleXhat.push_back(std::move(xhat));
        }

        out[s] = std::move(sampleOut);
        xhatBySample_[s] = std::move(sampleXhat);
    }

    return out;
}

std::vector<NNMatrixPtrV> BatchNormLayer::backwardBatch(const std::vector<NNMatrixPtrV>& dY) {
    std::vector<NNMatrixPtrV> dX;
    if (!validateBatchShape(dY)) {
        return dX;
    }
    if (dY.size() != xhatBySample_.size()) {
        return dX;
    }

    const size_t batchSize = dY.size();
    dX.assign(batchSize, NNMatrixPtrV{});

    std::fill(dGamma_.begin(), dGamma_.end(), 0.0f);
    std::fill(dBeta_.begin(), dBeta_.end(), 0.0f);

    // Compute dBeta and dGamma per channel.
    for (int c = 0; c < channels_; ++c) {
        double dBeta = 0.0;
        double dGamma = 0.0;

        for (size_t s = 0; s < batchSize; ++s) {
            const auto& sampleDY = dY[s];
            const auto& sampleXhat = xhatBySample_[s];
            if (sampleDY.empty() || sampleXhat.empty()) {
                continue;
            }
            const auto& dy = sampleDY[static_cast<size_t>(c)];
            const auto& xhat = sampleXhat[static_cast<size_t>(c)];
            if (!dy || !xhat) {
                continue;
            }
            const int len = dy->getRowSize() * dy->getColSize();
            const float* dyData = dy->data();
            const float* xhatData = xhat->data();
            if (!dyData || !xhatData || len <= 0) {
                continue;
            }
            for (int i = 0; i < len; ++i) {
                dBeta += static_cast<double>(dyData[i]);
                dGamma += static_cast<double>(dyData[i]) * static_cast<double>(xhatData[i]);
            }
        }

        dBeta_[static_cast<size_t>(c)] = static_cast<float>(dBeta);
        dGamma_[static_cast<size_t>(c)] = static_cast<float>(dGamma);
    }

    // Compute dX.
    for (size_t s = 0; s < batchSize; ++s) {
        const auto& sampleDY = dY[s];
        const auto& sampleXhat = xhatBySample_[s];
        if (sampleDY.empty() || sampleXhat.empty()) {
            continue;
        }

        NNMatrixPtrV sampleDX;
        sampleDX.reserve(static_cast<size_t>(channels_));

        for (int c = 0; c < channels_; ++c) {
            const auto& dy = sampleDY[static_cast<size_t>(c)];
            const auto& xhat = sampleXhat[static_cast<size_t>(c)];
            if (!dy || !xhat) {
                sampleDX.clear();
                break;
            }

            const int r = dy->getRowSize();
            const int col = dy->getColSize();
            const int len = r * col;

            auto dx = std::make_shared<NNMatrix>(r, col, 0.0f);
            float* dxData = dx ? dx->data() : nullptr;
            const float* dyData = dy->data();
            const float* xhatData = xhat->data();
            if (!dxData || !dyData || !xhatData || len <= 0) {
                sampleDX.clear();
                break;
            }

            const int N = std::max(1, elemCountPerChannel_[static_cast<size_t>(c)]);
            const float invN = 1.0f / static_cast<float>(N);
            const float invStd = batchInvStd_[static_cast<size_t>(c)];
            const float gamma = gamma_[static_cast<size_t>(c)];
            const float dBeta = dBeta_[static_cast<size_t>(c)];
            const float dGamma = dGamma_[static_cast<size_t>(c)];

            for (int i = 0; i < len; ++i) {
                const float term1 = static_cast<float>(N) * dyData[i];
                const float term2 = dBeta;
                const float term3 = xhatData[i] * dGamma;
                dxData[i] = invN * gamma * invStd * (term1 - term2 - term3);
            }

            sampleDX.push_back(std::move(dx));
        }

        dX[s] = std::move(sampleDX);
    }

    return dX;
}

void BatchNormLayer::update(float learningRate, float momentum) {
    if (channels_ <= 0) {
        return;
    }

    for (int c = 0; c < channels_; ++c) {
        const size_t idx = static_cast<size_t>(c);
        const int N = std::max(1, elemCountPerChannel_[idx]);
        const float invN = 1.0f / static_cast<float>(N);

        const float gGamma = dGamma_[idx] * invN;
        const float gBeta = dBeta_[idx] * invN;

        vGamma_[idx] = momentum * vGamma_[idx] + learningRate * gGamma;
        vBeta_[idx] = momentum * vBeta_[idx] + learningRate * gBeta;

        gamma_[idx] -= vGamma_[idx];
        beta_[idx] -= vBeta_[idx];
    }
}
