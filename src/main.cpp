#include "ArgHelper.h"
#include "CNN.h"
#include "DNN.h"
#include "DefaultCNNConfig.h"
#include "DefaultDNNConfig.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <algorithm>
#include <iostream>
#include <vector>

static void startCnnTraining(const ArgHelper& argHelper) {
    const int epochs = std::max(1, argHelper.intValue("--epochs", CIFAR100_CNN_EPOCHS));
    const int maxEpoch = argHelper.intValue("--maxEpoch", -1);
    const int targetEpoch = maxEpoch > 0 ? maxEpoch : epochs;
    const int batchSize = std::max(1, argHelper.intValue("--batch-size", CIFAR100_CNN_BATCH_SIZE));
    const float lr = argHelper.floatValue("--learning-rate", CIFAR100_CNN_LEARNING_RATE);
    const float momentum = argHelper.floatValue("--momentum", CIFAR100_CNN_MOMENTUM);
    const float weightDecay = argHelper.floatValue("--weight-decay", CIFAR100_CNN_WEIGHT_DECAY);

    const int maxTrain = argHelper.intValue("--max-train-samples", CIFAR100_CNN_MAX_TRAIN_SAMPLES);
    const int maxTest = argHelper.intValue("--max-test-samples", CIFAR100_CNN_MAX_TEST_SAMPLES);
    const char* checkpointPathArg = argHelper.value("--cnn-checkpoint");
    const std::string checkpointPath =
        checkpointPathArg ? std::string(checkpointPathArg) : std::string("cnn_checkpoint.bin");
    const bool loadBeforeTrain = argHelper.has("--cnn-load") || maxEpoch > 0;

    auto dataSet = NNDatasetManager::prepareCifar100Dataset(maxTrain, maxTest);
    auto configs = NNDatasetManager::buildCifar100CnnConfigs();

    auto cnn = CNN(configs);
    cnn.configurePersistence(checkpointPath, loadBeforeTrain);
    cnn.train(dataSet, targetEpoch, batchSize, lr, momentum, weightDecay);
}

static void startDnnTraining(const ArgHelper& argHelper) {
    auto dataSet = NNDatasetManager::loadMnist();
    const int epochs = std::max(1, argHelper.intValue("--epochs", DEFAULT_DNN_EPOCHS));
    const int maxEpoch = argHelper.intValue("--maxEpoch", -1);
    const int targetEpoch = maxEpoch > 0 ? maxEpoch : epochs;
    const int batchSize = std::max(1, argHelper.intValue("--batch-size", DEFAULT_DNN_BATCH_SIZE));
    const float lr = argHelper.floatValue("--learning-rate", DEFAULT_DNN_LEARNING_RATE);
    const float momentum = argHelper.floatValue("--momentum", DEFAULT_DNN_MOMENTUM);
    const char* checkpointPathArg = argHelper.value("--dnn-checkpoint");
    const std::string checkpointPath =
        checkpointPathArg ? std::string(checkpointPathArg) : std::string("dnn_checkpoint.bin");
    const bool loadBeforeTrain = argHelper.has("--dnn-load") || maxEpoch > 0;
    std::vector<int> cfg{DEFAULT_DNN_INPUT_SIZE, DEFAULT_DNN_HIDDEN1_SIZE, DEFAULT_DNN_HIDDEN2_SIZE,
                         DEFAULT_DNN_OUTPUT_SIZE};
    auto dnn = DNN(cfg);
    dnn.configurePersistence(checkpointPath, loadBeforeTrain);
    dnn.train(dataSet, targetEpoch, batchSize, lr, momentum, nullptr, nullptr, nullptr, nullptr);
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
