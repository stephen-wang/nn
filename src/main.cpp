#include "NNDataSetManager.h"
#include "NNUtils.h"
#include "NeuralNetwork.h"

const int INPUT_SIZE = 784; // 28x28 pixels
const int HIDDEN1_SIZE = 128;
const int HIDDEN2_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int EPOCHS = 9;
const int BATCH_SIZE = 16;
const float LEARNING_RATE = 0.005f;
const float MOMENTUM = 0.9f;

int main(int argc, char** argv) {
    auto dataSet = NNDataSetManager::loadMnistDataSet();
    std::vector<int> cfg{INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE};
    auto nn = NeuralNetwork(cfg);
    nn.train(dataSet, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, nullptr, nullptr, nullptr,
             nullptr);

    return 0;
}
