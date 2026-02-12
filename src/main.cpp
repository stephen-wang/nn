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
    auto dataSet = NNDatasetManager::loadCifar100();

    // By default, keep the run small (CNN backprop is currently disabled).
    // if (CIFAR100_CNN_MAX_TRAIN_SAMPLES > 0 &&
    //     static_cast<int>(dataSet.trainLabel_.size()) > CIFAR100_CNN_MAX_TRAIN_SAMPLES) {
    //     dataSet.trainLabel_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TRAIN_SAMPLES));
    //     dataSet.trainInput_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TRAIN_SAMPLES) *
    //                                static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
    // }
    // if (CIFAR100_CNN_MAX_TEST_SAMPLES > 0 &&
    //     static_cast<int>(dataSet.testLabel_.size()) > CIFAR100_CNN_MAX_TEST_SAMPLES) {
    //     dataSet.testLabel_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TEST_SAMPLES));
    //     dataSet.testInput_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TEST_SAMPLES) *
    //                               static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
    // }

    CNNConfigBuilder builder;
    auto configs =
        builder
            // Conv1: 3 -> 16 (3x3, stride 1, pad 1)
            .addConvolution(CIFAR100_CNN_IN_CHANNELS, CIFAR100_CNN_CONV1_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            // Conv2: 16 -> 32 (3x3, stride 1, pad 1)
            .addConvolution(CIFAR100_CNN_CONV1_OUT_CHANNELS, CIFAR100_CNN_CONV2_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            // Pool: 2x2, stride 2 (32x32 -> 16x16)
            .addMaxPooling(CIFAR100_CNN_POOLING_FILTER_SIZE, CIFAR100_CNN_POOLING_STRIDE_SIZE)
            // FC: 8192 -> 256 -> 100
            .addFullyConnected(CIFAR100_CNN_FC1_IN_SIZE, CIFAR100_CNN_FC1_OUT_SIZE)
            .addFullyConnected(CIFAR100_CNN_FC1_OUT_SIZE, CIFAR100_CNN_OUTPUT_SIZE)
            .build();

    auto cnn = CNN(configs);
    cnn.train(dataSet, CIFAR100_CNN_EPOCHS, CIFAR100_CNN_BATCH_SIZE, CIFAR100_CNN_LEARNING_RATE,
              CIFAR100_CNN_MOMENTUM);
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
