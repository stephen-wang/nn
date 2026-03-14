#include "NNDatasetManager.h"

#include "CNNConfigBuilder.h"
#include "DefaultCNNConfig.h"
#include "NNUtils.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {
inline float normalizeCifarByte(unsigned char v) {
    // Map [0,255] -> [-1,1]. Equivalent to: (v/255 - 0.5) / 0.5.
    return static_cast<float>(v) / 127.5f - 1.0f;
}
} // namespace

static const char* MNIST_TRAIN_DATA_FILE = "dataset/mnist/train-images-idx3-ubyte";
static const char* MNIST_TRAIN_LABEL_FILE = "dataset/mnist/train-labels-idx1-ubyte";
static const char* MNISt_TEST_DATA_FILE = "dataset/mnist/t10k-images-idx3-ubyte";
static const char* MNIST_TEST_LABEL_FILE = "dataset/mnist/t10k-labels-idx1-ubyte";

static const char* CIFAR100_TRAIN_FILE = "dataset/cifar-100/cifar-100-binary/train.bin";
static const char* CIFAR100_TEST_FILE = "dataset/cifar-100/cifar-100-binary/test.bin";
static const char* CIFAR100_COARSE_LABEL_NAMES_FILE =
    "dataset/cifar-100/cifar-100-binary/coarse_label_names.txt";
static const char* CIFAR100_FINE_LABEL_NAMES_FILE =
    "dataset/cifar-100/cifar-100-binary/fine_label_names.txt";

const std::string NNDatasetManager::TAG = "NNFunctions";

void NNDatasetManager::readCifar100BinaryFile(
    const std::string& filePath, NNMatrixPtrV& data, NNMatrixPtrV& labels,
    std::vector<std::vector<unsigned char>>& previewBytes) {
    // CIFAR-100 binary format:
    // 1 byte coarse label, 1 byte fine label, 3072 bytes image (32x32x3, channel-major)
    constexpr std::size_t kImageBytes = 32u * 32u * 3u;
    constexpr std::size_t kRecordBytes = 2u + kImageBytes;
    constexpr int kImageSide = 32;
    constexpr int kChannelSize = kImageSide * kImageSide;
    constexpr int kNumClasses = 100;

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open " + filePath);
    }

    file.seekg(0, std::ios::end);
    const std::streampos endPos = file.tellg();
    if (endPos <= 0) {
        throw std::runtime_error("Empty file: " + filePath);
    }
    const std::size_t fileSize = static_cast<std::size_t>(endPos);
    if (fileSize % kRecordBytes != 0) {
        throw std::runtime_error("Invalid CIFAR-100 file size (not multiple of record size): " +
                                 filePath);
    }
    const std::size_t recordCount = fileSize / kRecordBytes;
    file.seekg(0, std::ios::beg);

    // For CNN input we expose CIFAR-100 as 3 channel matrices (R,G,B) per sample.
    // The stored input format is channel-major: 1024 R, 1024 G, 1024 B.
    data.reserve(recordCount * 3);
    labels.reserve(recordCount);
    previewBytes.reserve(recordCount);

    std::vector<unsigned char> buffer(kImageBytes);
    for (std::size_t i = 0; i < recordCount; ++i) {
        unsigned char coarse = 0;
        unsigned char fine = 0;
        file.read(reinterpret_cast<char*>(&coarse), 1);
        file.read(reinterpret_cast<char*>(&fine), 1);
        file.read(reinterpret_cast<char*>(buffer.data()),
                  static_cast<std::streamsize>(kImageBytes));
        if (!file) {
            throw std::runtime_error("Unexpected EOF while reading: " + filePath);
        }

        previewBytes.emplace_back(buffer.begin(), buffer.end());

        (void) coarse; // coarse label currently unused

        auto r = std::make_shared<NNMatrix>(kImageSide, kImageSide);
        auto g = std::make_shared<NNMatrix>(kImageSide, kImageSide);
        auto b = std::make_shared<NNMatrix>(kImageSide, kImageSide);

        for (int row = 0; row < kImageSide; ++row) {
            for (int col = 0; col < kImageSide; ++col) {
                const int idx = row * kImageSide + col;
                // Normalize to approximately zero-mean range: [0,1] -> [-1,1].
                // This is a lightweight alternative to per-channel mean/std normalization.
                const float rf = normalizeCifarByte(buffer[static_cast<std::size_t>(idx)]);
                const float gf =
                    normalizeCifarByte(buffer[static_cast<std::size_t>(kChannelSize + idx)]);
                const float bf =
                    normalizeCifarByte(buffer[static_cast<std::size_t>(2 * kChannelSize + idx)]);
                r->set(row, col, rf);
                g->set(row, col, gf);
                b->set(row, col, bf);
            }
        }

        if (fine >= kNumClasses) {
            throw std::runtime_error("Invalid CIFAR-100 fine label in file: " + filePath);
        }
        auto label = std::make_shared<NNMatrix>(kNumClasses, 1);
        label->set(static_cast<int>(fine), 0, 1.0f);

        data.push_back(std::move(r));
        data.push_back(std::move(g));
        data.push_back(std::move(b));
        labels.push_back(std::move(label));
    }
}

NNDataset NNDatasetManager::loadMnist() {
    LOG << "Read train data from " << MNIST_TRAIN_DATA_FILE;
    auto inputs = NNUtils::read_mnist_data(MNIST_TRAIN_DATA_FILE);
    NNUtils::normalizeMnistData(inputs);

    LOG << "Read train label from " << MNIST_TRAIN_LABEL_FILE;
    auto labels = NNUtils::read_mnist_labels(MNIST_TRAIN_LABEL_FILE);
    NNUtils::normalizeMnistLabel(labels);

    LOG << "Read test data from " << MNISt_TEST_DATA_FILE;
    auto testInputs = NNUtils::read_mnist_data(MNISt_TEST_DATA_FILE);
    NNUtils::normalizeMnistData(testInputs);

    LOG << "Read test label from " << MNIST_TEST_LABEL_FILE;
    auto testLabels = NNUtils::read_mnist_labels(MNIST_TEST_LABEL_FILE);
    NNUtils::normalizeMnistLabel(testLabels);

    return {"mnist dataset", std::move(inputs), std::move(labels), std::move(testInputs),
            std::move(testLabels)};
}

NNDataset NNDatasetManager::loadCifar100() {
    NNMatrixPtrV inputs, labels;
    std::vector<std::vector<unsigned char>> trainPreviewBytes;
    LOG << "Read CIFAR-100 train data from " << CIFAR100_TRAIN_FILE;
    readCifar100BinaryFile(CIFAR100_TRAIN_FILE, inputs, labels, trainPreviewBytes);
    LOG << "CIFAR-100 train data " << inputs.size() << ", total samples " << labels.size();

    NNMatrixPtrV testInputs, testLabels;
    std::vector<std::vector<unsigned char>> testPreviewBytes;
    LOG << "Read CIFAR-100 test data from " << CIFAR100_TEST_FILE;
    readCifar100BinaryFile(CIFAR100_TEST_FILE, testInputs, testLabels, testPreviewBytes);
    LOG << "CIFAR-100 test data: " << testInputs.size() << ", total samples " << testLabels.size();

    return {"cifar-100 dataset",        std::move(inputs),     std::move(labels),
            std::move(testInputs),      std::move(testLabels), std::move(trainPreviewBytes),
            std::move(testPreviewBytes)};
}

NNDataset NNDatasetManager::prepareCifar100Dataset(int maxTrainSamples, int maxTestSamples) {
    auto dataSet = loadCifar100();

    if (maxTrainSamples > 0 && static_cast<int>(dataSet.trainLabel_.size()) > maxTrainSamples) {
        dataSet.trainLabel_.resize(static_cast<size_t>(maxTrainSamples));
        dataSet.trainInput_.resize(static_cast<size_t>(maxTrainSamples) *
                                   static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
        dataSet.trainPreviewBytes_.resize(static_cast<size_t>(maxTrainSamples));
    }

    if (maxTestSamples > 0 && static_cast<int>(dataSet.testLabel_.size()) > maxTestSamples) {
        dataSet.testLabel_.resize(static_cast<size_t>(maxTestSamples));
        dataSet.testInput_.resize(static_cast<size_t>(maxTestSamples) *
                                  static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
        dataSet.testPreviewBytes_.resize(static_cast<size_t>(maxTestSamples));
    }

    return dataSet;
}

std::vector<CNNConfigPtr> NNDatasetManager::buildCifar100CnnConfigs() {
    CNNConfigBuilder builder;

    return builder
        .addConvolution(CIFAR100_CNN_IN_CHANNELS, CIFAR100_CNN_CONV1_OUT_CHANNELS,
                        CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                        CIFAR100_CNN_CONV_PADDING_SIZE)
        .addBatchNorm(CIFAR100_CNN_CONV1_OUT_CHANNELS)
        .addConvolution(CIFAR100_CNN_CONV1_OUT_CHANNELS, CIFAR100_CNN_CONV2_OUT_CHANNELS,
                        CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                        CIFAR100_CNN_CONV_PADDING_SIZE)
        .addBatchNorm(CIFAR100_CNN_CONV2_OUT_CHANNELS)
        .addMaxPooling(CIFAR100_CNN_POOLING_FILTER_SIZE, CIFAR100_CNN_POOLING_STRIDE_SIZE)
        .addConvolution(CIFAR100_CNN_CONV2_OUT_CHANNELS, CIFAR100_CNN_CONV3_OUT_CHANNELS,
                        CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                        CIFAR100_CNN_CONV_PADDING_SIZE)
        .addBatchNorm(CIFAR100_CNN_CONV3_OUT_CHANNELS)
        .addMaxPooling(CIFAR100_CNN_POOLING_FILTER_SIZE, CIFAR100_CNN_POOLING_STRIDE_SIZE)
        .addFullyConnected(CIFAR100_CNN_FC1_IN_SIZE, CIFAR100_CNN_FC1_OUT_SIZE)
        .addFullyConnected(CIFAR100_CNN_FC1_OUT_SIZE, CIFAR100_CNN_OUTPUT_SIZE)
        .build();
}

std::vector<std::string> NNDatasetManager::loadCifar100CoarseLabelNames() {
    return readCifar100LabelNamesFromFile(CIFAR100_COARSE_LABEL_NAMES_FILE);
}

std::vector<std::string> NNDatasetManager::loadCifar100FineLabelNames() {
    return readCifar100LabelNamesFromFile(CIFAR100_FINE_LABEL_NAMES_FILE);
}

std::vector<std::string>
NNDatasetManager::readCifar100LabelNamesFromFile(const std::string& filePath) {
    LOG << "Read CIFAR-100 coarse label names from " << filePath;

    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error(std::string("Unable to open ") + filePath);
    }

    std::vector<std::string> names;
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty()) {
            continue;
        }
        names.push_back(std::move(line));
    }

    return names;
}
