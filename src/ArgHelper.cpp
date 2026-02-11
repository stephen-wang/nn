#include "ArgHelper.h"

#include <iostream>

#if defined(NN_ENABLE_GUI)
#include "NNGuiUtils.h"
#endif

int ArgHelper::maybeRunGui(ModelType modelType) const {
#if defined(NN_ENABLE_GUI)
    const bool defaultGui =
#if defined(NN_DEFAULT_GUI)
        true;
#else
        false;
#endif

const bool requestGui = guiRequested(defaultGui);
    if (requestGui) {
        if (modelType == ModelType::CNN) {
            std::cerr << "GUI currently supports DNN only. Run without --gui or use --model dnn.";
            std::cerr << std::endl;
            return 2;
        }
        return NNGuiUtils::RunTrainingGui();
    }

    return -1;
#else
    if (has("--gui")) {
        std::cerr << "GUI support is not compiled into this binary. Build with `make nn_gui`.";
        std::cerr << std::endl;
        return 2;
    }

    return -1;
#endif
}
