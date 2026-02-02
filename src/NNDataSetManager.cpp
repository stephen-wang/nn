#include "NNDataSetManager.h"

#include "NNUtils.h"

static const char* MNIST_TRAIN_DATA_FILE = "mnist/train-images-idx3-ubyte";
static const char* MNIST_TRAIN_LABEL_FILE = "mnist/train-labels-idx1-ubyte";
static const char* MNISt_TEST_DATA_FILE = "mnist/t10k-images-idx3-ubyte";
static const char* MNIST_TEST_LABEL_FILE = "mnist/t10k-labels-idx1-ubyte";

NNDataSet NNDataSetManager::loadMnistDataSet() {
    NNLOG_INFO("nn_gui") << "Read train data from " << MNIST_TRAIN_DATA_FILE;
    auto inputs = NNUtils::read_mnist_data(MNIST_TRAIN_DATA_FILE);
    NNUtils::normalizeMnistData(inputs);

    NNLOG_INFO("nn_gui") << "Read train label from " << MNIST_TRAIN_LABEL_FILE;
    auto labels = NNUtils::read_mnist_labels(MNIST_TRAIN_LABEL_FILE);
    NNUtils::normalizeMnistLabel(labels);

    NNLOG_INFO("nn_gui") << "Read test data from " << MNISt_TEST_DATA_FILE;
    auto testInputs = NNUtils::read_mnist_data(MNISt_TEST_DATA_FILE);
    NNUtils::normalizeMnistData(testInputs);

    NNLOG_INFO("nn_gui") << "Read test label from " << MNIST_TEST_LABEL_FILE;
    auto testLabels = NNUtils::read_mnist_labels(MNIST_TEST_LABEL_FILE);
    NNUtils::normalizeMnistLabel(testLabels);

    return NNDataSet("mnist dataset", inputs, labels, testInputs, testLabels);
}
