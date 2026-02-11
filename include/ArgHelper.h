#pragma once

#include <ostream>
#include <string>
#include <string_view>

enum class ModelType {
    DNN,
    CNN,
};

class ArgHelper {
  public:
    ArgHelper(int argc, char** argv) : argc_(argc), argv_(argv) {}

    bool has(std::string_view flag) const {
        for (int i = 1; i < argc_; ++i) {
            if (argv_[i] && std::string_view(argv_[i]) == flag) {
                return true;
            }
        }
        return false;
    }

    const char* value(std::string_view key) const {
        const std::string keyEq = std::string(key) + "=";
        for (int i = 1; i < argc_; ++i) {
            if (!argv_[i])
                continue;

            const std::string arg = argv_[i];
            if (arg == key && i + 1 < argc_ && argv_[i + 1]) {
                return argv_[i + 1];
            }
            if (arg.rfind(keyEq, 0) == 0) {
                return argv_[i] + keyEq.size();
            }
        }
        return nullptr;
    }

    bool helpRequested() const { return has("--help") || has("-h"); }

    bool cliRequested() const { return has("--cli"); }

    bool guiRequested(bool defaultGui) const {
        return has("--gui") || (defaultGui && !cliRequested());
    }

    ModelType modelType() const {
        if (has("--cnn"))
            return ModelType::CNN;
        if (has("--dnn"))
            return ModelType::DNN;

        if (const char* model = value("--model")) {
            const std::string m = model;
            if (m == "cnn")
                return ModelType::CNN;
            if (m == "dnn")
                return ModelType::DNN;
        }

        return ModelType::DNN;
    }

    void printUsage(std::ostream& os, const char* program = "./main") const {
        os << "Usage: " << program << " [--gui] [--cli] [--model dnn|cnn]\n"
           << "  --gui  Run training GUI (requires nn_gui build)\n"
           << "  --cli  Force CLI training mode (useful for ./nn_gui --cli)\n"
           << "  --model dnn|cnn  Select network type (default: dnn)\n"
           << "  --dnn / --cnn    Short aliases for --model\n";
    }

    // Returns >= 0 when GUI handling is complete and main() should return that code.
    // Returns -1 when main() should continue in CLI mode.
    int maybeRunGui(ModelType modelType) const;

  private:
    int argc_;
    char** argv_;
};
