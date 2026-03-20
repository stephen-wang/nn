#include "NNGuiUtils.h"

#include "CNN.h"
#include "DNN.h"
#include "DefaultDNNConfig.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include "../third_party/imgui/backends/imgui_impl_glfw.h"
#include "../third_party/imgui/backends/imgui_impl_opengl3.h"
#include "../third_party/imgui/imgui.h"

#include <GLFW/glfw3.h>
#include <OpenGL/gl.h>

namespace {
constexpr int kPathBufferSize = 512;
constexpr int kMnistSide = 28;
constexpr int kCifarSide = 32;
constexpr int kCifarChannels = 3;

enum class GuiMode {
    DNN = 0,
    CNN = 1,
};

struct PredictionEntry {
    int index = -1;
    float score = 0.0f;
    std::string label;
};

struct BrowserState {
    GuiMode mode = GuiMode::DNN;
    char parameterPath[kPathBufferSize] = {};
    char dataPath[kPathBufferSize] = {};

    std::unique_ptr<DNN> dnn;
    std::unique_ptr<CNN> cnn;

    std::vector<NNMatrixPtr> dnnInputs;
    std::vector<std::vector<unsigned char>> dnnPreviewBytes;
    std::vector<int> dnnGroundTruth;

    NNMatrixPtrV cnnInputs;
    std::vector<std::vector<unsigned char>> cnnPreviewBytes;
    std::vector<std::string> cnnFineLabels;
    std::vector<int> cnnGroundTruth;

    bool modelLoaded = false;
    bool dataLoaded = false;
    int pageStart = 0;
    int selectedIndex = 0;
    bool scrollSelectionIntoView = false;
    int predictedIndex = -1;
    float predictedScore = 0.0f;
    std::vector<PredictionEntry> predictions;

    std::string modelStatus = "";
    std::string dataStatus = "";
    std::string predictStatus = "";
};

void copyStringToBuffer(char* buffer, size_t size, const std::string& value) {
    if (!buffer || size == 0) {
        return;
    }
    std::snprintf(buffer, size, "%s", value.c_str());
}

std::string trimTrailingWhitespace(std::string value) {
    while (!value.empty() &&
           (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
        value.pop_back();
    }
    return value;
}

#if defined(__APPLE__)
std::string chooseFileDialog(const char* prompt) {
    std::string script = "osascript -e 'POSIX path of (choose file with prompt \"" +
                         std::string(prompt) + "\")' 2>/dev/null";
    FILE* pipe = popen(script.c_str(), "r");
    if (!pipe) {
        return {};
    }
    char buffer[1024] = {};
    std::string result;
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    pclose(pipe);
    return trimTrailingWhitespace(result);
}
#else
std::string chooseFileDialog(const char* prompt) {
    (void) prompt;
    return {};
}
#endif

std::string guessMnistLabelPath(const std::string& imagePath) {
    std::string path = imagePath;
    const std::array<std::pair<const char*, const char*>, 2> replacements = {{
        {"train-images-idx3-ubyte", "train-labels-idx1-ubyte"},
        {"t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"},
    }};
    for (const auto& item : replacements) {
        const std::string from = item.first;
        const std::string to = item.second;
        const size_t pos = path.find(from);
        if (pos != std::string::npos) {
            path.replace(pos, from.size(), to);
            return path;
        }
    }
    return {};
}

std::vector<int> extractOneHotIndices(const NNMatrixPtrV& labels) {
    std::vector<int> indices;
    indices.reserve(labels.size());
    for (const auto& label : labels) {
        if (!label || label->getColSize() != 1 || label->getRowSize() <= 0) {
            indices.push_back(-1);
            continue;
        }
        indices.push_back(label->getIndexOfColMax(0));
    }
    return indices;
}

std::vector<int> loadMnistGroundTruth(const std::string& imagePath) {
    const std::string labelPath = guessMnistLabelPath(imagePath);
    if (labelPath.empty()) {
        return {};
    }
    auto labels = NNUtils::read_mnist_labels(labelPath);
    return extractOneHotIndices(labels);
}

std::vector<int> parseDnnCheckpointConfig(const std::string& filePath) {
    std::ifstream is(filePath, std::ios::binary);
    if (!is.is_open()) {
        throw std::runtime_error("Unable to open checkpoint: " + filePath);
    }

    auto skipSerializedMatrix = [&](const char* matrixName) {
        std::int32_t rows = 0;
        std::int32_t cols = 0;
        is.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        is.read(reinterpret_cast<char*>(&cols), sizeof(cols));
        if (!is.good() || rows <= 0 || cols <= 0) {
            throw std::runtime_error(std::string("Invalid DNN ") + matrixName + " matrix metadata");
        }

        const std::streamoff payloadBytes = static_cast<std::streamoff>(rows) *
                                            static_cast<std::streamoff>(cols) *
                                            static_cast<std::streamoff>(sizeof(float));
        is.seekg(payloadBytes, std::ios::cur);
        if (!is.good()) {
            throw std::runtime_error(std::string("Truncated DNN ") + matrixName +
                                     " matrix payload");
        }
    };

    char magic[8] = {};
    is.read(magic, sizeof(magic));
    constexpr char kMagic[8] = {'N', 'N', 'D', 'N', 'N', '1', '\0', '\0'};
    if (!is.good() || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) {
        throw std::runtime_error("Invalid DNN checkpoint header");
    }

    std::uint32_t version = 0;
    std::uint32_t layerCount = 0;
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    is.read(reinterpret_cast<char*>(&layerCount), sizeof(layerCount));
    if (!is.good() || (version != 1 && version != 2) || layerCount == 0) {
        throw std::runtime_error("Unsupported DNN checkpoint format");
    }

    if (version >= 2) {
        std::int32_t completedEpoch = 0;
        is.read(reinterpret_cast<char*>(&completedEpoch), sizeof(completedEpoch));
        if (!is.good()) {
            throw std::runtime_error("Corrupt DNN checkpoint");
        }
    }

    std::vector<int> cfg;
    cfg.reserve(static_cast<size_t>(layerCount) + 1u);
    for (std::uint32_t i = 0; i < layerCount; ++i) {
        std::int32_t inputSize = 0;
        std::int32_t outputSize = 0;
        is.read(reinterpret_cast<char*>(&inputSize), sizeof(inputSize));
        is.read(reinterpret_cast<char*>(&outputSize), sizeof(outputSize));
        if (!is.good() || inputSize <= 0 || outputSize <= 0) {
            throw std::runtime_error("Invalid DNN layer metadata");
        }
        if (cfg.empty()) {
            cfg.push_back(static_cast<int>(inputSize));
        }
        cfg.push_back(static_cast<int>(outputSize));

        skipSerializedMatrix("weight");
        skipSerializedMatrix("velocity weight");
        skipSerializedMatrix("bias");
        skipSerializedMatrix("velocity bias");
    }
    return cfg;
}

void resetPrediction(BrowserState& state) {
    state.predictedIndex = -1;
    state.predictedScore = 0.0f;
    state.predictions.clear();
    state.predictStatus.clear();
}

void resetData(BrowserState& state) {
    state.dnnInputs.clear();
    state.dnnPreviewBytes.clear();
    state.dnnGroundTruth.clear();
    state.cnnInputs.clear();
    state.cnnPreviewBytes.clear();
    state.cnnGroundTruth.clear();
    state.dataLoaded = false;
    state.pageStart = 0;
    state.selectedIndex = 0;
    state.scrollSelectionIntoView = false;
    resetPrediction(state);
}

void setModeDefaults(BrowserState& state, GuiMode mode) {
    state.mode = mode;
    resetData(state);
    state.modelLoaded = false;
    state.dnn.reset();
    state.cnn.reset();
    state.modelStatus.clear();
    state.dataStatus.clear();
    if (mode == GuiMode::DNN) {
        copyStringToBuffer(state.parameterPath, sizeof(state.parameterPath), "dnn_checkpoint.bin");
        copyStringToBuffer(state.dataPath, sizeof(state.dataPath),
                           "dataset/mnist/t10k-images-idx3-ubyte");
    } else {
        copyStringToBuffer(state.parameterPath, sizeof(state.parameterPath), "cnn_checkpoint.bin");
        copyStringToBuffer(state.dataPath, sizeof(state.dataPath),
                           "dataset/cifar-100/cifar-100-binary/test.bin");
    }
}

GLFWwindow* initWindow() {
    if (!glfwInit()) {
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1440, 960, "NN Inference Browser", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    return window;
}

int itemCount(const BrowserState& state) {
    if (state.mode == GuiMode::DNN) {
        return static_cast<int>(state.dnnInputs.size());
    }
    if (!state.cnnPreviewBytes.empty()) {
        return static_cast<int>(state.cnnPreviewBytes.size());
    }
    return static_cast<int>(state.cnnInputs.size()) / kCifarChannels;
}

std::vector<unsigned char> extractMnistPreview(const NNMatrix& matrix) {
    const int rows = matrix.getRowSize();
    std::vector<unsigned char> preview(static_cast<size_t>(rows), 0);
    for (int i = 0; i < rows; ++i) {
        const float v = matrix.get(i, 0);
        const float clamped = std::max(0.0f, std::min(255.0f, v));
        preview[static_cast<size_t>(i)] = static_cast<unsigned char>(clamped);
    }
    return preview;
}

void loadModel(BrowserState& state) {
    resetPrediction(state);
    state.modelLoaded = false;
    state.modelStatus = "Loading model ...";
    try {
        if (state.mode == GuiMode::DNN) {
            std::vector<int> cfg = parseDnnCheckpointConfig(state.parameterPath);
            auto model = std::make_unique<DNN>(cfg);
            if (!model->load(state.parameterPath)) {
                state.modelLoaded = false;
                state.modelStatus = "Failed to load DNN model.";
                return;
            }
            state.dnn = std::move(model);
            state.cnn.reset();
            state.modelLoaded = true;
            state.modelStatus = "Loaded NN model";
            return;
        }

        auto model = std::make_unique<CNN>(NNDatasetManager::buildCifar100CnnConfigs());
        if (!model->load(state.parameterPath)) {
            state.modelLoaded = false;
            state.modelStatus = "Failed to load CNN model.";
            return;
        }
        state.cnn = std::move(model);
        state.dnn.reset();
        state.modelLoaded = true;
        state.modelStatus = "Loaded NN model";
    } catch (const std::exception& ex) {
        state.modelLoaded = false;
        state.modelStatus = std::string("Model load error: ") + ex.what();
    } catch (...) {
        state.modelLoaded = false;
        state.modelStatus = "Unknown model load error.";
    }
}

void loadDnnData(BrowserState& state) {
    auto rawInputs = NNUtils::read_mnist_data(state.dataPath);
    state.dnnInputs.clear();
    state.dnnPreviewBytes.clear();
    state.dnnGroundTruth = loadMnistGroundTruth(state.dataPath);
    state.dnnInputs.reserve(rawInputs.size());
    state.dnnPreviewBytes.reserve(rawInputs.size());

    for (const auto& raw : rawInputs) {
        if (!raw) {
            continue;
        }
        state.dnnPreviewBytes.push_back(extractMnistPreview(*raw));
        auto normalized = std::make_shared<NNMatrix>(*raw);
        *normalized /= 255.0f;
        state.dnnInputs.push_back(std::move(normalized));
    }
}

void loadCnnData(BrowserState& state) {
    auto dataSet = NNDatasetManager::loadCifar100File(state.dataPath);
    state.cnnInputs = std::move(dataSet.trainInput_);
    state.cnnPreviewBytes = std::move(dataSet.trainPreviewBytes_);
    state.cnnGroundTruth = extractOneHotIndices(dataSet.trainLabel_);
    if (state.cnnFineLabels.empty()) {
        state.cnnFineLabels = NNDatasetManager::loadCifar100FineLabelNames();
    }
}

void loadData(BrowserState& state) {
    resetData(state);
    state.dataStatus = "Loading data ...";
    try {
        if (state.mode == GuiMode::DNN) {
            loadDnnData(state);
        } else {
            loadCnnData(state);
        }
        const int total = itemCount(state);
        state.dataLoaded = total > 0;
        state.pageStart = 0;
        state.selectedIndex = 0;
        state.scrollSelectionIntoView = state.dataLoaded;
        state.dataStatus = state.dataLoaded ? "Loaded test input data" : "No items loaded.";
    } catch (const std::exception& ex) {
        state.dataLoaded = false;
        state.dataStatus = std::string("Data load error: ") + ex.what();
    } catch (...) {
        state.dataLoaded = false;
        state.dataStatus = "Unknown data load error.";
    }
}

std::vector<PredictionEntry> buildPredictions(const NNMatrix& output, GuiMode mode,
                                              const std::vector<std::string>& cnnLabels) {
    std::vector<PredictionEntry> entries;
    if (output.getColSize() != 1 || output.getRowSize() <= 0) {
        return entries;
    }

    entries.reserve(static_cast<size_t>(output.getRowSize()));
    for (int i = 0; i < output.getRowSize(); ++i) {
        PredictionEntry entry;
        entry.index = i;
        entry.score = output.get(i, 0);
        if (mode == GuiMode::DNN) {
            entry.label = std::to_string(i);
        } else if (static_cast<size_t>(i) < cnnLabels.size()) {
            entry.label = cnnLabels[static_cast<size_t>(i)];
        } else {
            entry.label = std::string("class ") + std::to_string(i);
        }
        entries.push_back(std::move(entry));
    }

    std::sort(entries.begin(), entries.end(),
              [](const PredictionEntry& a, const PredictionEntry& b) { return a.score > b.score; });
    if (entries.size() > 5) {
        entries.resize(5);
    }
    return entries;
}

void predictSelected(BrowserState& state) {
    resetPrediction(state);
    if (!state.modelLoaded) {
        state.predictStatus = "Load a model first.";
        return;
    }
    if (!state.dataLoaded) {
        state.predictStatus = "Load test data first.";
        return;
    }
    if (state.selectedIndex < 0 || state.selectedIndex >= itemCount(state)) {
        state.predictStatus = "Select an input item first.";
        return;
    }

    try {
        if (state.mode == GuiMode::DNN) {
            if (!state.dnn) {
                state.predictStatus = "DNN model is not ready.";
                return;
            }
            NNMatrix output =
                state.dnn->infer(state.dnnInputs[static_cast<size_t>(state.selectedIndex)]);
            state.predictedIndex = output.getIndexOfColMax(0);
            state.predictedScore = output.get(state.predictedIndex, 0);
            state.predictions = buildPredictions(output, state.mode, state.cnnFineLabels);
            state.predictStatus = "Prediction finished.";
            return;
        }

        if (!state.cnn) {
            state.predictStatus = "CNN model is not ready.";
            return;
        }
        const int base = state.selectedIndex * kCifarChannels;
        if (base + kCifarChannels > static_cast<int>(state.cnnInputs.size())) {
            state.predictStatus = "Selected CNN sample is out of range.";
            return;
        }
        NNMatrixPtrV sample;
        sample.reserve(kCifarChannels);
        for (int c = 0; c < kCifarChannels; ++c) {
            sample.push_back(state.cnnInputs[static_cast<size_t>(base + c)]);
        }
        NNMatrix output = state.cnn->infer(sample);
        state.predictedIndex = output.getIndexOfColMax(0);
        state.predictedScore = output.get(state.predictedIndex, 0);
        state.predictions = buildPredictions(output, state.mode, state.cnnFineLabels);
        state.predictStatus = "Prediction finished.";
    } catch (const std::exception& ex) {
        state.predictStatus = std::string("Predict error: ") + ex.what();
    } catch (...) {
        state.predictStatus = "Unknown predict error.";
    }
}

void drawGrayscaleImage(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                        const std::vector<unsigned char>& image, int width, int height) {
    if (image.size() < static_cast<size_t>(width * height)) {
        drawList->AddText(origin, IM_COL32(220, 220, 220, 255), "No image");
        return;
    }

    const float scale = std::max(1.0f, std::floor(std::min(size.x / static_cast<float>(width),
                                                           size.y / static_cast<float>(height))));
    const float imgW = scale * static_cast<float>(width);
    const float imgH = scale * static_cast<float>(height);
    const float offsetX = std::floor(origin.x + (size.x - imgW) * 0.5f);
    const float offsetY = std::floor(origin.y + (size.y - imgH) * 0.5f);

    drawList->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const unsigned char value = image[static_cast<size_t>(y * width + x)];
            const ImU32 color = IM_COL32(value, value, value, 255);
            const float x0 = offsetX + static_cast<float>(x) * scale;
            const float y0 = offsetY + static_cast<float>(y) * scale;
            drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + scale, y0 + scale), color);
        }
    }
    drawList->AddRect(ImVec2(offsetX, offsetY), ImVec2(offsetX + imgW, offsetY + imgH),
                      IM_COL32(90, 100, 120, 255));
    drawList->PopClipRect();
}

void drawRgbImage(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                  const std::vector<unsigned char>& image, int side) {
    const int planeSize = side * side;
    if (image.size() < static_cast<size_t>(planeSize * kCifarChannels)) {
        drawList->AddText(origin, IM_COL32(220, 220, 220, 255), "No image");
        return;
    }

    const float scale = std::max(1.0f, std::floor(std::min(size.x / static_cast<float>(side),
                                                           size.y / static_cast<float>(side))));
    const float imgW = scale * static_cast<float>(side);
    const float imgH = scale * static_cast<float>(side);
    const float offsetX = std::floor(origin.x + (size.x - imgW) * 0.5f);
    const float offsetY = std::floor(origin.y + (size.y - imgH) * 0.5f);

    drawList->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);
    for (int y = 0; y < side; ++y) {
        for (int x = 0; x < side; ++x) {
            const int idx = y * side + x;
            const ImU32 color = IM_COL32(image[static_cast<size_t>(idx)],
                                         image[static_cast<size_t>(planeSize + idx)],
                                         image[static_cast<size_t>(2 * planeSize + idx)], 255);
            const float x0 = offsetX + static_cast<float>(x) * scale;
            const float y0 = offsetY + static_cast<float>(y) * scale;
            drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + scale, y0 + scale), color);
        }
    }
    drawList->AddRect(ImVec2(offsetX, offsetY), ImVec2(offsetX + imgW, offsetY + imgH),
                      IM_COL32(90, 100, 120, 255));
    drawList->PopClipRect();
}

void drawSelectedPreview(BrowserState& state) {
    ImVec2 origin = ImGui::GetCursorScreenPos();
    ImVec2 size = ImGui::GetContentRegionAvail();
    ImDrawList* drawList = ImGui::GetWindowDrawList();

    if (!state.dataLoaded || state.selectedIndex < 0 || state.selectedIndex >= itemCount(state)) {
        drawList->AddText(origin, IM_COL32(220, 220, 220, 255), "No input selected.");
        ImGui::InvisibleButton("preview_canvas", size);
        return;
    }

    if (state.mode == GuiMode::DNN) {
        drawGrayscaleImage(drawList, origin, size,
                           state.dnnPreviewBytes[static_cast<size_t>(state.selectedIndex)],
                           kMnistSide, kMnistSide);
    } else {
        ImVec2 cnnSize(size.x * 0.5f, size.y * 0.5f);
        ImVec2 cnnOrigin(origin.x + (size.x - cnnSize.x) * 0.5f,
                         origin.y + (size.y - cnnSize.y) * 0.5f);
        drawRgbImage(drawList, cnnOrigin, cnnSize,
                     state.cnnPreviewBytes[static_cast<size_t>(state.selectedIndex)], kCifarSide);
    }
    ImGui::InvisibleButton("preview_canvas", size);
}

std::string groundTruthLabel(const BrowserState& state) {
    if (!state.dataLoaded || state.selectedIndex < 0 || state.selectedIndex >= itemCount(state)) {
        return {};
    }
    if (state.mode == GuiMode::DNN) {
        if (static_cast<size_t>(state.selectedIndex) >= state.dnnGroundTruth.size()) {
            return {};
        }
        const int label = state.dnnGroundTruth[static_cast<size_t>(state.selectedIndex)];
        return label >= 0 ? std::to_string(label) : std::string();
    }
    if (static_cast<size_t>(state.selectedIndex) >= state.cnnGroundTruth.size()) {
        return {};
    }
    const int label = state.cnnGroundTruth[static_cast<size_t>(state.selectedIndex)];
    if (label < 0) {
        return {};
    }
    if (static_cast<size_t>(label) < state.cnnFineLabels.size()) {
        return state.cnnFineLabels[static_cast<size_t>(label)] + " (" + std::to_string(label) + ")";
    }
    return std::to_string(label);
}

} // namespace

int NNGuiUtils::RunGui() {
    GLFWwindow* window = initWindow();
    if (!window) {
        return 1;
    }

    const char* glsl_version = "#version 150";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    BrowserState state;
    setModeDefaults(state, GuiMode::DNN);

    while (!glfwWindowShouldClose(window)) {
        bool triggerLoadModel = false;
        bool triggerLoadData = false;

        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
        ImGui::Begin("NN Inference Browser", nullptr, flags);

        ImGui::BeginChild("Controls", ImVec2(430.0f, 0.0f), true);

        const char* modeItems[] = {"DNN", "CNN"};
        int modeIndex = (state.mode == GuiMode::DNN) ? 0 : 1;
        if (ImGui::Combo("Mode", &modeIndex, modeItems, IM_ARRAYSIZE(modeItems))) {
            setModeDefaults(state, modeIndex == 0 ? GuiMode::DNN : GuiMode::CNN);
        }

        ImGui::SeparatorText("Model");
        ImGui::InputText("Parameter file", state.parameterPath, IM_ARRAYSIZE(state.parameterPath));
        ImGui::SameLine();
        if (ImGui::Button("Browse##model")) {
            const std::string path = chooseFileDialog("Select model file");
            if (!path.empty()) {
                copyStringToBuffer(state.parameterPath, sizeof(state.parameterPath), path);
            }
        }
        if (ImGui::Button("Load model", ImVec2(-1.0f, 0.0f))) {
            state.modelStatus = "Loading model ...";
            triggerLoadModel = true;
        }
        ImGui::TextWrapped("%s", state.modelStatus.c_str());

        ImGui::SeparatorText("Data");
        ImGui::InputText("Test data file", state.dataPath, IM_ARRAYSIZE(state.dataPath));
        ImGui::SameLine();
        if (ImGui::Button("Browse##data")) {
            const std::string path = chooseFileDialog("Select test data file");
            if (!path.empty()) {
                copyStringToBuffer(state.dataPath, sizeof(state.dataPath), path);
            }
        }
        if (ImGui::Button("Load data", ImVec2(-1.0f, 0.0f))) {
            state.dataStatus = "Loading data ...";
            triggerLoadData = true;
        }
        ImGui::TextWrapped("%s", state.dataStatus.c_str());

        const int totalItems = itemCount(state);
        if (totalItems > 0) {
            ImGui::SeparatorText("Input items");
            ImGui::Text("Total items: %d", totalItems);

            const float listHeight =
                std::min(320.0f, ImGui::GetTextLineHeightWithSpacing() * 14.0f +
                                     ImGui::GetStyle().FramePadding.y * 2.0f);
            if (ImGui::BeginChild("InputItemsList", ImVec2(0.0f, listHeight), true)) {
                ImGuiListClipper clipper;
                clipper.Begin(totalItems);
                while (clipper.Step()) {
                    for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i) {
                        char itemLabel[64] = {};
                        std::snprintf(itemLabel, sizeof(itemLabel), "Item %d", i);
                        const bool isSelected = (state.selectedIndex == i);
                        if (ImGui::Selectable(itemLabel, isSelected)) {
                            state.selectedIndex = i;
                            resetPrediction(state);
                            state.scrollSelectionIntoView = true;
                        }
                        if (isSelected && state.scrollSelectionIntoView) {
                            ImGui::SetScrollHereY(0.25f);
                            state.scrollSelectionIntoView = false;
                        }
                    }
                }
            }
            ImGui::EndChild();
        }

        ImGui::SeparatorText("Predict");
        if (ImGui::Button("Predict", ImVec2(-1.0f, 0.0f))) {
            predictSelected(state);
        }
        if (!state.predictStatus.empty()) {
            ImGui::TextWrapped("%s", state.predictStatus.c_str());
        }
        if (state.predictedIndex >= 0) {
            const std::string gt = groundTruthLabel(state);
            if (!gt.empty()) {
                ImGui::TextWrapped("Ground truth: %s", gt.c_str());
            }
            if (state.mode == GuiMode::DNN) {
                ImGui::Text("Predicted digit: %d", state.predictedIndex);
            } else if (static_cast<size_t>(state.predictedIndex) < state.cnnFineLabels.size()) {
                ImGui::TextWrapped(
                    "Predicted class: %s (%d)",
                    state.cnnFineLabels[static_cast<size_t>(state.predictedIndex)].c_str(),
                    state.predictedIndex);
            } else {
                ImGui::Text("Predicted class: %d", state.predictedIndex);
            }
            ImGui::Text("Score: %.4f", state.predictedScore);
        } else {
            const std::string gt = groundTruthLabel(state);
            if (!gt.empty()) {
                ImGui::TextWrapped("Ground truth: %s", gt.c_str());
            }
        }

        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("PreviewPane", ImVec2(0.0f, 0.0f), true);
        ImGui::Text("Selected input preview");
        ImGui::Separator();
        ImGui::BeginChild("ImageArea", ImVec2(0.0f, 420.0f), true);
        drawSelectedPreview(state);
        ImGui::EndChild();

        ImGui::Spacing();
        ImGui::Text("Top predictions");
        ImGui::Separator();
        if (state.predictions.empty()) {
            ImGui::TextDisabled("Run prediction to see results.");
        } else {
            for (const auto& entry : state.predictions) {
                ImGui::BulletText("%s (%d): %.4f", entry.label.c_str(), entry.index, entry.score);
            }
        }
        ImGui::EndChild();

        ImGui::End();

        ImGui::Render();
        int display_w = 0;
        int display_h = 0;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);

        if (triggerLoadModel) {
            loadModel(state);
        }
        if (triggerLoadData) {
            loadData(state);
        }
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
