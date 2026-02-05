#include "DNN.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <string>

#if defined(NN_ENABLE_GUI)
#include "NNGuiUtils.h"
#endif

const int INPUT_SIZE = 784; // 28x28 pixels
const int HIDDEN1_SIZE = 128;
const int HIDDEN2_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int EPOCHS = 9;
const int BATCH_SIZE = 16;
const float LEARNING_RATE = 0.005f;
const float MOMENTUM = 0.9f;

static bool hasArg(int argc, char** argv, const char* flag) {
    for (int i = 1; i < argc; ++i) {
        if (argv[i] && std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

int main(int argc, char** argv) {
#if defined(NN_ENABLE_GUI)
    const bool defaultGui =
#if defined(NN_DEFAULT_GUI)
        true;
#else
        false;
#endif

    const bool requestCli = hasArg(argc, argv, "--cli");
    const bool requestGui = hasArg(argc, argv, "--gui") || (defaultGui && !requestCli);
    if (requestGui) {
        return NNGuiUtils::RunTrainingGui();
    }
#else
    if (hasArg(argc, argv, "--gui")) {
        std::cerr << "GUI support is not compiled into this binary. Build with `make nn_gui`."
                  << std::endl;
        return 2;
    }
#endif

    if (hasArg(argc, argv, "--help") || hasArg(argc, argv, "-h")) {
        std::cout << "Usage: ./main [--gui] [--cli]\n"
                  << "  --gui  Run training GUI (requires nn_gui build)\n"
                  << "  --cli  Force CLI training mode (useful for ./nn_gui --cli)\n";
        return 0;
    }

    auto dataSet = NNDatasetManager::loadMnistDataset();
    std::vector<int> cfg{INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE};
    auto nn = DNN(cfg);
    nn.train(dataSet, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, nullptr, nullptr, nullptr,
             nullptr);

    return 0;
}
