#include "ArgHelper.h"
#include "CNN.h"
#include "CNNConfigBuilder.h"
#include "DNN.h"
#include "DefaultCNNConfig.h"
#include "DefaultDNNConfig.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <iostream>
#include <vector>

static void startCnnTraining() {
    CNNConfigBuilder builder;
    auto configs =
        builder
            // add convlution layer (filter 3x3, padding 1, stride 1)
            .addConvolution(DEFAULT_CNN_INPUT_SIZE, DEFAULT_CNN_CONV_OUTPUT_SIZE,
                            DEFAULT_CNN_CONV_FILTER_SIZE, DEFAULT_CNN_CONV_PADDING_SIZE,
                            DEFAULT_CNN_CONV_STRIDE_SIZE)
            // add max pooling layer (filter 2x2, stride 2)
            .addMaxPooling(DEFAULT_CNN_POOLING_INPUT_SIZE, DEFAULT_CNN_POOLING_OUTPUT_SIZE,
                           DEFAULT_CNN_POOLING_FILTER_SIZE, DEFAULT_CNN_POOLING_STRIDE_SIZE)
            // add fully connected layer (input size 13, output size10)
            .addFullyConnected(DEFAULT_CNN_FLATTEN_INPUT_SIZE, DEFAULT_CNN_OUTPUT_SIZE)
            .build();

    [[maybe_unused]] CNN cnn(configs);
}

static void startDnnTraining() {
    auto dataSet = NNDatasetManager::loadMnist();
    std::vector<int> cfg{DEFAULT_DNN_INPUT_SIZE, DEFAULT_DNN_HIDDEN1_SIZE, DEFAULT_DNN_HIDDEN2_SIZE,
                         DEFAULT_DNN_OUTPUT_SIZE};
    auto nn = DNN(cfg);
    nn.train(dataSet, DEFAULT_DNN_EPOCHS, DEFAULT_DNN_BATCH_SIZE, DEFAULT_DNN_LEARNING_RATE,
             DEFAULT_DNN_MOMENTUM, nullptr, nullptr, nullptr, nullptr);
}

int main(int argc, char** argv) {
    ArgHelper argHelper(argc, argv);
    const ModelType modelType = argHelper.modelType();

    const int guiExitCode = argHelper.maybeRunGui(modelType);
    if (guiExitCode >= 0) {
        return guiExitCode;
    }

    if (argHelper.helpRequested()) {
        argHelper.printUsage(std::cout, argv[0] ? argv[0] : "./main");
        return 0;
    }

    if (modelType == ModelType::DNN) {
        startDnnTraining();
    } else {
        startCnnTraining();
    }

    return 0;
}
