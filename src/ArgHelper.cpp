#include "ArgHelper.h"

#include <iostream>

#if defined(NN_ENABLE_GUI)
#include "CNNGuiUtils.h"
#include "DNNGuiUtils.h"
#include "NNGuiUtils.h"
#include <string>

namespace {
std::string resolveCheckpointPath(const ArgHelper& args, const char* key, const char* fallback) {
    const char* value = args.value(key);
    return value ? std::string(value) : std::string(fallback);
}
}
#endif

int ArgHelper::maybeRunGui() const {
#if defined(NN_ENABLE_GUI)
    const bool defaultGui =
#if defined(NN_DEFAULT_GUI)
        true;
#else
        false;
#endif

    const GuiLaunchMode launchMode = guiLaunchMode(defaultGui);
    switch (launchMode) {
    case GuiLaunchMode::Infer:
        return NNGuiUtils::RunGui();
    case GuiLaunchMode::TrainDNN: {
        const int maxEpoch = intValue("--maxEpoch", -1);
        const bool loadBeforeTrain = has("--dnn-load") || maxEpoch > 0;
        return DNNGuiUtils::RunTrainingGui(
            resolveCheckpointPath(*this, "--dnn-checkpoint", "dnn_checkpoint.bin"),
            loadBeforeTrain, maxEpoch);
    }
    case GuiLaunchMode::TrainCNN: {
        const int maxEpoch = intValue("--maxEpoch", -1);
        const bool loadBeforeTrain = has("--cnn-load") || maxEpoch > 0;
        return CNNGuiUtils::RunTrainingGui(
            resolveCheckpointPath(*this, "--cnn-checkpoint", "cnn_checkpoint.bin"),
            loadBeforeTrain, maxEpoch);
    }
    case GuiLaunchMode::Invalid:
        std::cerr << "Invalid GUI mode. Use exactly one of: --inferMode, --trainMode dnn,"
                  << " or --trainMode cnn." << std::endl;
        return 2;
    case GuiLaunchMode::None:
        break;
    }

    return -1;
#else
    if (has("--gui") || has("--inferMode") || value("--trainMode")) {
        std::cerr << "GUI support is not compiled into this binary. Build with `make nn_gui`.";
        std::cerr << std::endl;
        return 2;
    }

    return -1;
#endif
}
