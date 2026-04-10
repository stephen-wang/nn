#include "BNLayer.h"

#include "NNStreamUtils.h"
#include "nnlog/nnlog.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

BNLayer::BNLayer(int channels, float eps, float runningMomentum)
    : NNLayer(NNLayerType::BatchNorm), channels_(channels), eps_(eps),
      runningMomentum_(runningMomentum) {
    if (channels_ <= 0) {
        NNLOG_WARN("BNLayer") << "invalid channels " << channels_;
        channels_ = 0;
        return;
    }

    gamma_.assign(static_cast<size_t>(channels_), 1.0f);
    beta_.assign(static_cast<size_t>(channels_), 0.0f);
    vGamma_.assign(static_cast<size_t>(channels_), 0.0f);
    vBeta_.assign(static_cast<size_t>(channels_), 0.0f);

    runningMean_.assign(static_cast<size_t>(channels_), 0.0f);
    runningVar_.assign(static_cast<size_t>(channels_), 1.0f);

    batchMean_.assign(static_cast<size_t>(channels_), 0.0);
    batchInvStd_.assign(static_cast<size_t>(channels_), 1.0);
    dGamma_.assign(static_cast<size_t>(channels_), 0.0f);
    dBeta_.assign(static_cast<size_t>(channels_), 0.0f);
    elemCountPerChannel_.assign(static_cast<size_t>(channels_), 0);
}

bool BNLayer::validateBatchShape(const std::vector<NNMatrixPtrV>& batch) const {
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

std::vector<NNMatrixPtrV> BNLayer::forwardBatch(const std::vector<NNMatrixPtrV>& batch,
                                                bool training) {
    std::vector<NNMatrixPtrV> out;
    if (!validateBatchShape(batch)) {
        return out;
    }

    const size_t batchSize = batch.size();
    out.assign(batchSize, NNMatrixPtrV{});

    // Reset caches/grad buffers.
    std::fill(batchMean_.begin(), batchMean_.end(), 0.0);
    std::fill(batchInvStd_.begin(), batchInvStd_.end(), 1.0);
    std::fill(elemCountPerChannel_.begin(), elemCountPerChannel_.end(), 0);
    xhatBySample_.assign(batchSize, NNMatrixPtrV{});

    // Compute mean/var per channel over (batch * H * W).
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int c = 0; c < channels_; ++c) {
#else
    for (int c = 0; c < channels_; ++c) {
#endif
        double weightedMeanSum = 0.0;
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
            const int len = m->elementCount();
            if (len <= 0) {
                continue;
            }
            weightedMeanSum += static_cast<double>(m->mean()) * static_cast<double>(len);
            n += len;
        }

        elemCountPerChannel_[static_cast<size_t>(c)] = n;

        if (!training) {
            // Use running stats.
            const double mean = static_cast<double>(runningMean_[static_cast<size_t>(c)]);
            const double var = static_cast<double>(runningVar_[static_cast<size_t>(c)]);
            batchMean_[static_cast<size_t>(c)] = mean;
            batchInvStd_[static_cast<size_t>(c)] = 1.0 / std::sqrt(std::max(0.0, var) + eps_);
            continue;
        }

        if (n <= 0) {
            batchMean_[static_cast<size_t>(c)] = 0.0;
            batchInvStd_[static_cast<size_t>(c)] = 1.0;
            continue;
        }

        const double mean = weightedMeanSum / static_cast<double>(n);
        double squaredDiffSum = 0.0;

        for (size_t s = 0; s < batchSize; ++s) {
            const auto& sample = batch[s];
            if (sample.empty()) {
                continue;
            }
            const auto& m = sample[static_cast<size_t>(c)];
            if (!m) {
                continue;
            }
            if (m->elementCount() <= 0) {
                continue;
            }
            squaredDiffSum += m->squaredDiffSum(mean);
        }

        const double var = std::max(0.0, squaredDiffSum / static_cast<double>(n));

        batchMean_[static_cast<size_t>(c)] = mean;
        batchInvStd_[static_cast<size_t>(c)] = 1.0 / std::sqrt(var + eps_);

        // Update running stats.
        runningMean_[static_cast<size_t>(c)] =
            (1.0f - runningMomentum_) * runningMean_[static_cast<size_t>(c)] +
            runningMomentum_ * static_cast<float>(mean);
        runningVar_[static_cast<size_t>(c)] =
            (1.0f - runningMomentum_) * runningVar_[static_cast<size_t>(c)] +
            runningMomentum_ * static_cast<float>(var);
    }

    // Normalize and affine transform.
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (size_t s = 0; s < batchSize; ++s) {
#else
    for (size_t s = 0; s < batchSize; ++s) {
#endif
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
            if (!x || x->elementCount() <= 0) {
                sampleOut.clear();
                sampleXhat.clear();
                break;
            }

            const double mean = batchMean_[static_cast<size_t>(c)];
            const double invStd = batchInvStd_[static_cast<size_t>(c)];
            const float gamma = gamma_[static_cast<size_t>(c)];
            const float beta = beta_[static_cast<size_t>(c)];

            auto xhat = std::make_shared<NNMatrix>(x->normalized(mean, invStd));
            auto y = std::make_shared<NNMatrix>(*xhat);
            if (!xhat || !y) {
                sampleOut.clear();
                sampleXhat.clear();
                break;
            }
            y->applyFunctionInplace([gamma, beta](float value) { return gamma * value + beta; });

            sampleOut.push_back(std::move(y));
            sampleXhat.push_back(std::move(xhat));
        }

        out[s] = std::move(sampleOut);
        xhatBySample_[s] = std::move(sampleXhat);
    }

    return out;
}

std::vector<NNMatrixPtrV> BNLayer::backwardBatch(const std::vector<NNMatrixPtrV>& dY) {
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
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int c = 0; c < channels_; ++c) {
#else
    for (int c = 0; c < channels_; ++c) {
#endif
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
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (size_t s = 0; s < batchSize; ++s) {
#else
    for (size_t s = 0; s < batchSize; ++s) {
#endif
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
            const double invStd = batchInvStd_[static_cast<size_t>(c)];
            const float gamma = gamma_[static_cast<size_t>(c)];
            const float dBeta = dBeta_[static_cast<size_t>(c)];
            const float dGamma = dGamma_[static_cast<size_t>(c)];

            for (int i = 0; i < len; ++i) {
                const float term1 = static_cast<float>(N) * dyData[i];
                const float term2 = dBeta;
                const float term3 = xhatData[i] * dGamma;
                dxData[i] = static_cast<float>(invN * gamma * invStd * (term1 - term2 - term3));
            }

            sampleDX.push_back(std::move(dx));
        }

        dX[s] = std::move(sampleDX);
    }

    return dX;
}

void BNLayer::update(float learningRate, float momentum) {
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

bool BNLayer::saveState(std::ostream& os) const {
    const std::int32_t channels = channels_;
    os.write(reinterpret_cast<const char*>(&channels), sizeof(channels));
    os.write(reinterpret_cast<const char*>(&eps_), sizeof(eps_));
    os.write(reinterpret_cast<const char*>(&runningMomentum_), sizeof(runningMomentum_));
    if (!os.good() || channels <= 0) {
        return false;
    }

    return NNStreamUtils::writeVector(os, gamma_) && NNStreamUtils::writeVector(os, beta_) &&
           NNStreamUtils::writeVector(os, vGamma_) && NNStreamUtils::writeVector(os, vBeta_) &&
           NNStreamUtils::writeVector(os, runningMean_) &&
           NNStreamUtils::writeVector(os, runningVar_) && os.good();
}

bool BNLayer::loadState(std::istream& is) {
    std::int32_t channels = 0;
    float eps = 0.0f;
    float runningMomentum = 0.0f;
    is.read(reinterpret_cast<char*>(&channels), sizeof(channels));
    is.read(reinterpret_cast<char*>(&eps), sizeof(eps));
    is.read(reinterpret_cast<char*>(&runningMomentum), sizeof(runningMomentum));
    if (!is.good() || channels <= 0) {
        return false;
    }

    if (channels != channels_) {
        return false;
    }

    std::vector<float> gamma;
    std::vector<float> beta;
    std::vector<float> vGamma;
    std::vector<float> vBeta;
    std::vector<float> runningMean;
    std::vector<float> runningVar;
    if (!NNStreamUtils::readVector(is, gamma) || !NNStreamUtils::readVector(is, beta) ||
        !NNStreamUtils::readVector(is, vGamma) || !NNStreamUtils::readVector(is, vBeta) ||
        !NNStreamUtils::readVector(is, runningMean) || !NNStreamUtils::readVector(is, runningVar)) {
        return false;
    }

    const std::size_t expected = static_cast<std::size_t>(channels_);
    if (gamma.size() != expected || beta.size() != expected || vGamma.size() != expected ||
        vBeta.size() != expected || runningMean.size() != expected ||
        runningVar.size() != expected) {
        return false;
    }

    eps_ = eps;
    runningMomentum_ = runningMomentum;
    gamma_ = std::move(gamma);
    beta_ = std::move(beta);
    vGamma_ = std::move(vGamma);
    vBeta_ = std::move(vBeta);
    runningMean_ = std::move(runningMean);
    runningVar_ = std::move(runningVar);

    std::fill(batchMean_.begin(), batchMean_.end(), 0.0);
    std::fill(batchInvStd_.begin(), batchInvStd_.end(), 1.0);
    std::fill(dGamma_.begin(), dGamma_.end(), 0.0f);
    std::fill(dBeta_.begin(), dBeta_.end(), 0.0f);
    std::fill(elemCountPerChannel_.begin(), elemCountPerChannel_.end(), 0);
    xhatBySample_.clear();
    return true;
}
