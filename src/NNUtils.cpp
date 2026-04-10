#include "NNUtils.h"

#include "NNLog.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>

const std::string NNUtils::TAG = "NNUtils";
uint32_t NNUtils::swap_endian(uint32_t val) {
    val = ((val << 8) & 0xff00ff00) | ((val >> 8) & 0xff00ff);
    val = (val >> 16) | (val << 16);
    return val;
}

float NNUtils::xavierInit(int inputSize, int outputSize) {
    float limit = std::sqrt(6.0f / float((inputSize + outputSize)));
    return NNUtils::random(-limit, limit);
}

float NNUtils::heInit(int fanIn) {
    if (fanIn <= 0) {
        return 0.0f;
    }

    // He uniform: Var = 2/fanIn, Uniform[-limit,limit] => Var = limit^2/3
    // => limit = sqrt(6/fanIn).
    const float limit = std::sqrt(6.0f / static_cast<float>(fanIn));
    return NNUtils::random(-limit, limit);
}

void NNUtils::normalizeMnistData(std::vector<NNMatrixPtr>& data) {
    for (auto& inputPtr : data) {
        auto& input = *inputPtr;
        input /= 255.0f;
    }
}

void NNUtils::normalizeMnistLabel(std::vector<NNMatrixPtr>& labels) {
    for (auto& labelPtr : labels) {
        auto& label = *labelPtr;
        label.toOneHot();
    }
}

NNMatrixPtr NNUtils::augmentCifar32Channel(const NNMatrixPtr& in, int pad, int cropY, int cropX,
                                           bool hflip, NNMatrixPtr& reuse) {
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

    // Instead of constructing a padded temporary matrix, compute the crop directly.
    // Semantics match:
    //   padded has input placed at offset (pad,pad), zeros elsewhere
    //   crop is taken from padded at (cropY,cropX)
    //   optional horizontal flip is applied to the cropped patch
    cropY = std::max(0, std::min(cropY, 2 * pad));
    cropX = std::max(0, std::min(cropX, 2 * pad));

    // Prepare reuse buffer (allocate once and reuse across calls where provided).
    if (!reuse || reuse->getRowSize() != side || reuse->getColSize() != side) {
        reuse = std::make_shared<NNMatrix>(side, side, 0.0f);
    } else {
        float* z = reuse->data();
        if (z) {
            const int len = side * side;
            std::fill_n(z, len, 0.0f);
        }
    }

    auto out = reuse;
    float* outData = out ? out->data() : nullptr;
    if (!outData) {
        return nullptr;
    }

    for (int y = 0; y < side; ++y) {
        // y in padded coordinates is (y + cropY). Convert to input coords by subtracting pad.
        const int srcY = (y + cropY) - pad;
        float* dstRow = outData + y * side;
        if (srcY < 0 || srcY >= side) {
            // Entire row is padding (already zero-initialized).
            continue;
        }

        const float* srcBaseRow = inData + srcY * side;
        for (int x = 0; x < side; ++x) {
            const int px = hflip ? (side - 1 - x) : x;
            const int srcX = (px + cropX) - pad;
            if (srcX < 0 || srcX >= side) {
                // Padding.
                continue;
            }
            dstRow[x] = srcBaseRow[srcX];
        }
    }

    return out;
}

std::vector<NNMatrixPtr> NNUtils::read_mnist_data(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open " + filePath);
    }

    int magic;
    file.read((char*) &magic, sizeof(magic));
    magic = swap_endian(magic);
    if (magic != MNIST_IMAGE_MAGIC) {
        throw std::runtime_error("Invalid mnist image file!");
    }

    int numImages = 0, row = 0, col = 0;
    file.read((char*) &numImages, sizeof(numImages));
    numImages = swap_endian(numImages);

    file.read((char*) &row, sizeof(row));
    row = swap_endian(row);

    file.read((char*) &col, sizeof(col));
    col = swap_endian(col);

    LOG << "Totally, " << numImages << " images, width " << col << ", height " << row;
    std::vector<NNMatrixPtr> result(numImages);
    for (int i = 0; i < numImages; i++) {
        auto imgSize = row * col;
        auto imgData = std::make_shared<NNMatrix>(imgSize, 1);
        std::vector<unsigned char> buffer(imgSize, 0);
        file.read(reinterpret_cast<char*>(buffer.data()), imgSize);
        for (int j = 0; j < imgSize; j++) {
            imgData->set(j, 0, static_cast<float>(buffer[j]));
        }

        result[i] = imgData;
    }

    return result;
}

std::vector<NNMatrixPtr> NNUtils::read_mnist_labels(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open " + filePath);
    }

    int magic;
    file.read((char*) &magic, sizeof(magic));
    magic = swap_endian(magic);
    if (magic != MNIST_LABEL_MAGIC) {
        throw std::runtime_error("Invalid mnist image file!");
    }

    int numLabels = 0;
    file.read((char*) &numLabels, sizeof(numLabels));
    numLabels = swap_endian(numLabels);
    LOG << "Totally, " << numLabels << " labels";

    std::vector<NNMatrixPtr> result(numLabels);
    for (int i = 0; i < numLabels; i++) {
        unsigned char ch = 0;
        file.read((char*) &ch, 1);
        assert(ch >= 0 && ch <= 9);

        auto label = std::make_shared<NNMatrix>(10, 1);
        label->set(ch, 0, 1.0f);

        result[i] = label;
    }

    file.close();
    return result;
}

float NNUtils::random(float a, float b) {
    static std::random_device rd;                     // Non-deterministic random seed
    static std::mt19937 gen(rd());                    // Mersenne Twister engine
    std::uniform_real_distribution<float> dist(a, b); // Range [a, b]
    return dist(gen);
}

std::vector<NNMatrixPtr> NNUtils::getBatch(std::vector<NNMatrixPtr>& input, int batchNo,
                                           int batchSize) {
    std::vector<NNMatrixPtr> ret;

    if (batchSize <= 0) {
        return ret;
    }

    const int total = static_cast<int>(input.size());
    const int start = batchNo * batchSize;
    if (start < 0 || start >= total) {
        return ret;
    }

    const int end = std::min(start + batchSize, total);
    ret.reserve(static_cast<size_t>(end - start));
    ret.insert(ret.end(), input.begin() + start, input.begin() + end);

    return ret;
}

NNMatrixPtr NNUtils::flattenAndConcat(const NNMatrixPtrV& mats) {
    if (mats.empty()) {
        return nullptr;
    }

    int total = 0;
    for (const auto& m : mats) {
        if (!m) {
            return nullptr;
        }
        const int r = m->getRowSize();
        const int c = m->getColSize();
        if (r <= 0 || c <= 0) {
            return nullptr;
        }
        total += r * c;
    }
    if (total <= 0) {
        return nullptr;
    }

    auto flat = std::make_shared<NNMatrix>(total, 1);
    float* out = flat->data();
    if (!out) {
        return nullptr;
    }

    int offset = 0;
    for (const auto& m : mats) {
        const float* in = m->data();
        const int len = m->getRowSize() * m->getColSize();
        if (!in || len <= 0) {
            return nullptr;
        }
        std::copy(in, in + len, out + offset);
        offset += len;
    }

    return flat;
}

void NNUtils::shuffle(std::vector<NNMatrixPtr>& input, std::vector<NNMatrixPtr>& label) {
    const size_t n = input.size();
    if (n == 0) {
        return;
    }
    if (label.size() != n) {
        LOG << "shuffle: input size " << input.size() << " != label size " << label.size();
        return;
    }

    // In-place Fisher–Yates shuffle.
    for (size_t i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        const size_t j = dist(rng());
        std::swap(input[i], input[j]);
        std::swap(label[i], label[j]);
    }
}

NNUtils::ShuffleSampleInfo NNUtils::shuffleSamples(std::vector<NNMatrixPtr>& input,
                                                   std::vector<NNMatrixPtr>& label) {
    const int dataCount = static_cast<int>(input.size());
    const int labelCount = static_cast<int>(label.size());

    ShuffleSampleInfo info;
    info.sampleCount = labelCount;

    if (labelCount > 0 && dataCount > 0 && (dataCount % labelCount) == 0) {
        info.inChannelSize = dataCount / labelCount;
    } else {
        info.inChannelSize = 1;
    }
    if (info.inChannelSize <= 0) {
        info.inChannelSize = 1;
    }

    if (info.sampleCount <= 0 && dataCount > 0) {
        // Best-effort fallback; training without labels isn't meaningful but avoid UB.
        info.sampleCount = dataCount / info.inChannelSize;
    }

    if (info.sampleCount > 0 && dataCount != info.sampleCount * info.inChannelSize) {
        LOG << "shuffleSamples size mismatch: input=" << dataCount << ", label=" << labelCount
            << ", inferred channels=" << info.inChannelSize << ". Falling back to channelSize=1.";
        info.inChannelSize = 1;
        info.sampleCount = labelCount;
    }

    if (dataCount == 0 || info.sampleCount <= 0) {
        return info;
    }

    // Single-channel samples can use the standard paired shuffle.
    if (info.inChannelSize == 1 && !label.empty()) {
        shuffle(input, label);
        return info;
    }

    // In-place Fisher–Yates shuffle by sample, keeping per-sample channels contiguous.
    // Each sample occupies a contiguous block of `inChannelSize` matrices in `input`.
    const int ch = info.inChannelSize;
    for (int s = info.sampleCount - 1; s > 0; --s) {
        std::uniform_int_distribution<int> dist(0, s);
        const int j = dist(rng());
        if (j == s) {
            continue;
        }
        if (!label.empty()) {
            std::swap(label[static_cast<size_t>(s)], label[static_cast<size_t>(j)]);
        }
        const int baseS = s * ch;
        const int baseJ = j * ch;
        for (int c = 0; c < ch; ++c) {
            std::swap(input[static_cast<size_t>(baseS + c)], input[static_cast<size_t>(baseJ + c)]);
        }
    }

    return info;
}

NNUtils::ShuffleSampleInfo
NNUtils::shuffleSamples(std::vector<NNMatrixPtr>& input, std::vector<NNMatrixPtr>& label,
                        std::vector<std::vector<unsigned char>>& previewBytes) {
    const int dataCount = static_cast<int>(input.size());
    const int labelCount = static_cast<int>(label.size());

    ShuffleSampleInfo info;
    info.sampleCount = labelCount;

    if (labelCount > 0 && dataCount > 0 && (dataCount % labelCount) == 0) {
        info.inChannelSize = dataCount / labelCount;
    } else {
        info.inChannelSize = 1;
    }
    if (info.inChannelSize <= 0) {
        info.inChannelSize = 1;
    }

    if (info.sampleCount <= 0 && dataCount > 0) {
        info.sampleCount = dataCount / info.inChannelSize;
    }

    if (info.sampleCount > 0 && dataCount != info.sampleCount * info.inChannelSize) {
        LOG << "shuffleSamples size mismatch: input=" << dataCount << ", label=" << labelCount
            << ", inferred channels=" << info.inChannelSize << ". Falling back to channelSize=1.";
        info.inChannelSize = 1;
        info.sampleCount = labelCount;
    }

    if (dataCount == 0 || info.sampleCount <= 0) {
        return info;
    }

    const bool previewAligned =
        previewBytes.empty() || static_cast<int>(previewBytes.size()) == info.sampleCount;

    if (info.inChannelSize == 1 && !label.empty()) {
        const size_t n = input.size();
        for (size_t i = n - 1; i > 0; --i) {
            std::uniform_int_distribution<size_t> dist(0, i);
            const size_t j = dist(rng());
            std::swap(input[i], input[j]);
            std::swap(label[i], label[j]);
            if (previewAligned && !previewBytes.empty()) {
                std::swap(previewBytes[i], previewBytes[j]);
            }
        }
        return info;
    }

    const int ch = info.inChannelSize;
    for (int s = info.sampleCount - 1; s > 0; --s) {
        std::uniform_int_distribution<int> dist(0, s);
        const int j = dist(rng());
        if (j == s) {
            continue;
        }
        if (!label.empty()) {
            std::swap(label[static_cast<size_t>(s)], label[static_cast<size_t>(j)]);
        }
        if (previewAligned && !previewBytes.empty()) {
            std::swap(previewBytes[static_cast<size_t>(s)], previewBytes[static_cast<size_t>(j)]);
        }
        const int baseS = s * ch;
        const int baseJ = j * ch;
        for (int c = 0; c < ch; ++c) {
            std::swap(input[static_cast<size_t>(baseS + c)], input[static_cast<size_t>(baseJ + c)]);
        }
    }

    return info;
}

void NNUtils::applyReluInPlace(NNMatrixPtrV& sample) {
    for (auto& channel : sample) {
        if (channel) {
            channel->applyFunctionInplace(NNFunctions::ReLUFunc);
        }
    }
}

// CIFAR100_CNN_MIN_LEARNING_RATE
//  CIFAR100_CNN_WARMUP_EPOCHS
float NNUtils::cosineAnnealedLearningRate(float maxLearningRate, float minLearningRate,
                                          int warmUpEpochNum, int totalEpochNum, int batchNum,
                                          int totalBatchesSeen) {
    if (warmUpEpochNum <= 0 || totalEpochNum <= 0 || batchNum <= 0) {
        return maxLearningRate;
    }

    const int warmupSteps = std::max(1, warmUpEpochNum * batchNum);
    const int totalSteps = std::max(warmupSteps + 1, totalEpochNum * batchNum);
    const int step = std::max(0, totalBatchesSeen);

    if (step < warmupSteps) {
        const float t = static_cast<float>(step + 1) / static_cast<float>(warmupSteps);
        return minLearningRate + (maxLearningRate - minLearningRate) * t;
    }

    constexpr float kPi = 3.14159265358979323846f;
    const float progress = std::min(1.0f, static_cast<float>(step - warmupSteps) /
                                              static_cast<float>(totalSteps - warmupSteps));
    const float cosine = 0.5f * (1.0f + std::cos(kPi * progress));
    return minLearningRate + (maxLearningRate - minLearningRate) * cosine;
}

void NNUtils::applyCutoutInPlace(bool enableCoutout, NNMatrixPtrV& sample, int cutoutSize) {
    if (!enableCoutout || sample.empty() || cutoutSize <= 0) {
        return;
    }
    if (!sample[0] || sample[0]->getRowSize() <= 0 || sample[0]->getColSize() <= 0) {
        return;
    }

    const int height = sample[0]->getRowSize();
    const int width = sample[0]->getColSize();
    const int cutoutSizeLimit = std::min(height, width) / 3;
    cutoutSize = std::min(cutoutSize, cutoutSizeLimit);
    std::uniform_int_distribution<int> rowDist(0, std::max(0, height - 1));
    std::uniform_int_distribution<int> colDist(0, std::max(0, width - 1));
    const int centerY = rowDist(NNUtils::rng());
    const int centerX = colDist(NNUtils::rng());
    const int half = cutoutSize / 2;
    const int y0 = std::max(0, centerY - half);
    const int y1 = std::min(height, centerY + half + (cutoutSize % 2));
    const int x0 = std::max(0, centerX - half);
    const int x1 = std::min(width, centerX + half + (cutoutSize % 2));

    for (auto& channel : sample) {
        if (!channel || channel->getRowSize() != height || channel->getColSize() != width) {
            return;
        }
        float* data = channel->data();
        if (!data) {
            return;
        }
        for (int y = y0; y < y1; ++y) {
            float* row = data + y * width;
            std::fill(row + x0, row + x1, 0.0f);
        }
    }
}

void NNUtils::gateReluGradientInPlace(std::vector<NNMatrixPtrV>& gradients,
                                      const std::vector<NNMatrixPtrV>& activations) {
    const size_t sampleCount = std::min(gradients.size(), activations.size());
    for (size_t s = 0; s < sampleCount; ++s) {
        auto& gradSample = gradients[s];
        const auto& actSample = activations[s];
        const size_t channelCount = std::min(gradSample.size(), actSample.size());
        for (size_t c = 0; c < channelCount; ++c) {
            auto& grad = gradSample[c];
            const auto& act = actSample[c];
            if (!grad || !act || !grad->hasSameDimension(*act)) {
                continue;
            }
            *grad = grad->elementProduct(act->applyFunction(NNFunctions::ReLUDrevative));
        }
    }
}

NNMatrixPtrV NNUtils::maybeAugmentCifar32Sample(bool enableDataAugment, bool enableCutout,
                                                int cutoutSize, int channelSize,
                                                NNMatrixPtrV& sample, int pad,
                                                NNMatrixPtrV* reuseBuffers) {
    if (!enableDataAugment) {
        return sample;
    }
    if (static_cast<int>(sample.size()) != channelSize) {
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

    const int cropY = cropDist(NNUtils::rng());
    const int cropX = cropDist(NNUtils::rng());
    const bool hflip = flipDist(NNUtils::rng());

    NNMatrixPtrV out;
    out.reserve(sample.size());
    for (size_t c = 0; c < sample.size(); ++c) {
        NNMatrixPtr tempReuse; // used when caller doesn't provide reuse buffers
        NNMatrixPtr& reuseRef = (reuseBuffers && reuseBuffers->size() >= sample.size())
                                    ? (*reuseBuffers)[c]
                                    : tempReuse;
        auto aug = NNUtils::augmentCifar32Channel(sample[c], pad, cropY, cropX, hflip, reuseRef);
        if (!aug) {
            return sample;
        }
        out.push_back(std::move(aug));
    }
    NNUtils::applyCutoutInPlace(enableCutout, out, cutoutSize);
    return out;
}

NNMatrixPtr NNUtils::flattenAndConcatReuse(const NNMatrixPtrV& mats, NNMatrixPtr& reuse) {
    if (mats.empty()) {
        return nullptr;
    }

    int total = 0;
    for (const auto& m : mats) {
        if (!m) {
            return nullptr;
        }
        const int r = m->getRowSize();
        const int c = m->getColSize();
        if (r <= 0 || c <= 0) {
            return nullptr;
        }
        total += r * c;
    }
    if (total <= 0) {
        return nullptr;
    }

    if (!reuse || reuse->getRowSize() != total || reuse->getColSize() != 1) {
        reuse = std::make_shared<NNMatrix>(total, 1);
    }

    float* out = reuse ? reuse->data() : nullptr;
    if (!out) {
        return nullptr;
    }

    int offset = 0;
    for (const auto& m : mats) {
        const float* in = m ? m->data() : nullptr;
        const int len = m ? (m->getRowSize() * m->getColSize()) : 0;
        if (!in || len <= 0) {
            return nullptr;
        }
        std::copy(in, in + len, out + offset);
        offset += len;
    }
    return reuse;
}
