#include "NNUtils.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <numeric>
#include <random>

namespace {
std::mt19937& rng() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    return gen;
}
} // namespace

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

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng());

    std::vector<NNMatrixPtr> shuffledInput;
    std::vector<NNMatrixPtr> shuffledLabel;
    shuffledInput.reserve(n);
    shuffledLabel.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        const size_t idx = indices[i];
        shuffledInput.push_back(input[idx]);
        shuffledLabel.push_back(label[idx]);
    }

    input.swap(shuffledInput);
    label.swap(shuffledLabel);
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

    // Shuffle by sample index, keeping per-sample channels contiguous.
    std::vector<int> indices(info.sampleCount);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng());

    std::vector<NNMatrixPtr> shuffledData;
    shuffledData.reserve(static_cast<size_t>(info.sampleCount * info.inChannelSize));

    std::vector<NNMatrixPtr> shuffledLabel;
    if (!label.empty()) {
        shuffledLabel.reserve(static_cast<size_t>(info.sampleCount));
    }

    for (int dstSample = 0; dstSample < info.sampleCount; ++dstSample) {
        const int srcSample = indices[dstSample];
        const int base = srcSample * info.inChannelSize;
        for (int c = 0; c < info.inChannelSize; ++c) {
            shuffledData.push_back(input[base + c]);
        }
        if (!label.empty()) {
            shuffledLabel.push_back(label[srcSample]);
        }
    }

    input.swap(shuffledData);
    if (!label.empty()) {
        label.swap(shuffledLabel);
    }

    return info;
}
