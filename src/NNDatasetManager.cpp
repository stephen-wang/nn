#include "NNDatasetManager.h"

#include "NNUtils.h"

#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

static const char* MNIST_TRAIN_DATA_FILE = "dataset/mnist/train-images-idx3-ubyte";
static const char* MNIST_TRAIN_LABEL_FILE = "dataset/mnist/train-labels-idx1-ubyte";
static const char* MNISt_TEST_DATA_FILE = "dataset/mnist/t10k-images-idx3-ubyte";
static const char* MNIST_TEST_LABEL_FILE = "dataset/mnist/t10k-labels-idx1-ubyte";

static const char* CIFAR100_TRAIN_FILE = "dataset/cifar/cifar-100-binary/train.bin";
static const char* CIFAR100_TEST_FILE = "dataset/cifar/cifar-100-binary/test.bin";
static const char* CIFAR100_COARSE_LABEL_NAMES_FILE =
    "dataset/cifar/cifar-100-binary/coarse_label_names.txt";
static const char* CIFAR100_FINE_LABEL_NAMES_FILE =
    "dataset/cifar/cifar-100-binary/fine_label_names.txt";

const std::string NNDatasetManager::TAG = "NNFunctions";

void NNDatasetManager::readCifar100BinaryFile(const std::string& filePath, NNMatrixPtrVector& data,
                                              NNMatrixPtrVector& labels) {
    // CIFAR-100 binary format:
    // 1 byte coarse label, 1 byte fine label, 3072 bytes image (32x32x3, channel-major)
    constexpr std::size_t kImageBytes = 32u * 32u * 3u;
    constexpr std::size_t kRecordBytes = 2u + kImageBytes;
    constexpr int kInputSize = static_cast<int>(kImageBytes);
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

    data.reserve(recordCount);
    labels.reserve(recordCount);

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

        (void) coarse; // coarse label currently unused

        auto input = std::make_shared<NNMatrix>(kInputSize, 1);
        for (int j = 0; j < kInputSize; ++j) {
            input->set(j, 0, static_cast<float>(buffer[static_cast<std::size_t>(j)]) / 255.0f);
        }

        if (fine >= kNumClasses) {
            throw std::runtime_error("Invalid CIFAR-100 fine label in file: " + filePath);
        }
        auto label = std::make_shared<NNMatrix>(kNumClasses, 1);
        label->set(static_cast<int>(fine), 0, 1.0f);

        data.push_back(std::move(input));
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
    NNMatrixPtrVector inputs, labels;
    LOG << "Read CIFAR-100 train data from " << CIFAR100_TRAIN_FILE;
    readCifar100BinaryFile(CIFAR100_TRAIN_FILE, inputs, labels);
    LOG << "CIFAR-100 train samples: " << inputs.size();

    NNMatrixPtrVector testInputs, testLabels;
    LOG << "Read CIFAR-100 test data from " << CIFAR100_TEST_FILE;
    readCifar100BinaryFile(CIFAR100_TEST_FILE, testInputs, testLabels);
    LOG << "CIFAR-100 test samples: " << testInputs.size();

    return {"cifar-100 dataset", std::move(inputs), std::move(labels), std::move(testInputs),
            std::move(testLabels)};
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
