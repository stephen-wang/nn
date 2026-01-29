#include "NNUtils.h"
#include "NeuralNetwork.h"

const char *MNIST_TRAIN_DATA_FILE = "mnist/train-images-idx3-ubyte";
const char *MNIST_TRAIN_LABEL_FILE = "mnist/train-labels-idx1-ubyte";
const char *MNISt_TEST_DATA_FILE = "mnist/t10k-images-idx3-ubyte";
const char *MNIST_TEST_LABEL_FILE = "mnist/t10k-labels-idx1-ubyte";

const int INPUT_SIZE = 784; // 28x28 pixels
const int HIDDEN1_SIZE = 128;
const int HIDDEN2_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int EPOCHS = 9;
const int BATCH_SIZE = 16;
const float LEARNING_RATE = 0.005f;
const float MOMENTUM = 0.9f;

int main(int argc, char **argv)
{
  NNLOG_INFO("main") << "Read train data from " << MNIST_TRAIN_DATA_FILE;
  auto inputs = NNUtils::read_mnist_data(MNIST_TRAIN_DATA_FILE);
  NNUtils::normalizeMnistData(inputs);

  NNLOG_INFO("main") << "Read train label from " << MNIST_TRAIN_LABEL_FILE;
  auto labels = NNUtils::read_mnist_labels(MNIST_TRAIN_LABEL_FILE);
  NNUtils::normalizeMnistLabel(labels);

  NNLOG_INFO("main") << "Read test data from " << MNISt_TEST_DATA_FILE;
  auto testInputs = NNUtils::read_mnist_data(MNISt_TEST_DATA_FILE);
  NNUtils::normalizeMnistData(testInputs);

  NNLOG_INFO("main") << "Read test label from " << MNIST_TEST_LABEL_FILE;
  auto testLabels = NNUtils::read_mnist_labels(MNIST_TEST_LABEL_FILE);
  NNUtils::normalizeMnistLabel(testLabels);

  std::vector<int> cfg{INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE};
  auto nn = NeuralNetwork(cfg);
  nn.train(inputs, labels, testInputs, testLabels, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, nullptr, nullptr, nullptr, nullptr);

  return 0;
}
