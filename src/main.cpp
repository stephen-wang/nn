#include "ArgHelper.h"
#include "CNN.h"
#include "CNNConfigBuilder.h"
#include "DNN.h"
#include "DefaultCNNConfig.h"
#include "DefaultDNNConfig.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <algorithm>
#include <iostream>
#include <vector>

static void startCnnTraining(const ArgHelper& argHelper) {
    auto dataSet = NNDatasetManager::loadCifar100();

    const int epochs = std::max(1, argHelper.intValue("--epochs", CIFAR100_CNN_EPOCHS));
    const int batchSize = std::max(1, argHelper.intValue("--batch-size", CIFAR100_CNN_BATCH_SIZE));
    const float lr = argHelper.floatValue("--learning-rate", CIFAR100_CNN_LEARNING_RATE);
    const float momentum = argHelper.floatValue("--momentum", CIFAR100_CNN_MOMENTUM);
    const float weightDecay = argHelper.floatValue("--weight-decay", CIFAR100_CNN_WEIGHT_DECAY);

    const int maxTrain = argHelper.intValue("--max-train-samples", CIFAR100_CNN_MAX_TRAIN_SAMPLES);
    const int maxTest = argHelper.intValue("--max-test-samples", CIFAR100_CNN_MAX_TEST_SAMPLES);

    // Keep the default CLI run reasonably small to iterate faster.
    if (maxTrain > 0 && static_cast<int>(dataSet.trainLabel_.size()) > maxTrain) {
        dataSet.trainLabel_.resize(static_cast<size_t>(maxTrain));
        dataSet.trainInput_.resize(static_cast<size_t>(maxTrain) *
                                   static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
    }
    if (maxTest > 0 && static_cast<int>(dataSet.testLabel_.size()) > maxTest) {
        dataSet.testLabel_.resize(static_cast<size_t>(maxTest));
        dataSet.testInput_.resize(static_cast<size_t>(maxTest) *
                                  static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
    }

    CNNConfigBuilder builder;
    auto configs =
        builder
            // Conv1: 3 -> 32 (3x3, stride 1, pad 1)
            .addConvolution(CIFAR100_CNN_IN_CHANNELS, CIFAR100_CNN_CONV1_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            .addBatchNorm(CIFAR100_CNN_CONV1_OUT_CHANNELS)
            // Conv2: 32 -> 64 (3x3, stride 1, pad 1)
            .addConvolution(CIFAR100_CNN_CONV1_OUT_CHANNELS, CIFAR100_CNN_CONV2_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            .addBatchNorm(CIFAR100_CNN_CONV2_OUT_CHANNELS)
            // Pool: 2x2, stride 2 (32x32 -> 16x16)
            .addMaxPooling(CIFAR100_CNN_POOLING_FILTER_SIZE, CIFAR100_CNN_POOLING_STRIDE_SIZE)
            // Conv3: 64 -> 128 (3x3, stride 1, pad 1)
            .addConvolution(CIFAR100_CNN_CONV2_OUT_CHANNELS, CIFAR100_CNN_CONV3_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            .addBatchNorm(CIFAR100_CNN_CONV3_OUT_CHANNELS)
            // Pool: 2x2, stride 2 (16x16 -> 8x8)
            .addMaxPooling(CIFAR100_CNN_POOLING_FILTER_SIZE, CIFAR100_CNN_POOLING_STRIDE_SIZE)
            // FC: 8192 -> 512 -> 100
            .addFullyConnected(CIFAR100_CNN_FC1_IN_SIZE, CIFAR100_CNN_FC1_OUT_SIZE)
            .addFullyConnected(CIFAR100_CNN_FC1_OUT_SIZE, CIFAR100_CNN_OUTPUT_SIZE)
            .build();

    auto cnn = CNN(configs);
    cnn.train(dataSet, epochs, batchSize, lr, momentum, weightDecay);
}

static void startDnnTraining(const ArgHelper& argHelper) {
    auto dataSet = NNDatasetManager::loadMnist();
    const int epochs = std::max(1, argHelper.intValue("--epochs", DEFAULT_DNN_EPOCHS));
    const int batchSize = std::max(1, argHelper.intValue("--batch-size", DEFAULT_DNN_BATCH_SIZE));
    const float lr = argHelper.floatValue("--learning-rate", DEFAULT_DNN_LEARNING_RATE);
    const float momentum = argHelper.floatValue("--momentum", DEFAULT_DNN_MOMENTUM);
    std::vector<int> cfg{DEFAULT_DNN_INPUT_SIZE, DEFAULT_DNN_HIDDEN1_SIZE, DEFAULT_DNN_HIDDEN2_SIZE,
                         DEFAULT_DNN_OUTPUT_SIZE};
    auto nn = DNN(cfg);
    nn.train(dataSet, epochs, batchSize, lr, momentum, nullptr, nullptr, nullptr, nullptr);
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
        startDnnTraining(argHelper);
    } else {
        startCnnTraining(argHelper);
    }

    return 0;
}
