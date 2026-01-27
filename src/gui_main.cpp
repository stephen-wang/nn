#include "NNUtils.h"
#include "NeuralNetwork.h"

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
#include <array>
#include <cfloat>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <cmath>

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include <OpenGL/gl3.h>
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"

const char *MNIST_TRAIN_DATA_FILE = "mnist/train-images-idx3-ubyte";
const char *MNIST_TRAIN_LABEL_FILE = "mnist/train-labels-idx1-ubyte";
const char *MNISt_TEST_DATA_FILE = "mnist/t10k-images-idx3-ubyte";
const char *MNIST_TEST_LABEL_FILE = "mnist/t10k-labels-idx1-ubyte";

const int INPUT_SIZE = 784; // 28x28 pixels
const int HIDDEN1_SIZE = 128;
const int HIDDEN2_SIZE = 64;
const int OUTPUT_SIZE = 10;
const int EPOCHS = 9;
const int BATCH_SIZE = 16;
const float LEARNING_RATE = 0.005f;
const float MOMENTUM = 0.9f;

struct TrainingStats
{
    std::vector<float> loss;
    std::vector<float> acc;
    std::mutex mutex;
    std::atomic<int> currentEpoch{0};
    std::atomic<bool> done{false};
    std::atomic<int> activeLayer{-1};
    std::atomic<int> activePhase{0};
    std::vector<float> currentImage;
    int currentOutputIndex = -1;
    float currentOutputValue = 0.0f;
    std::atomic<bool> stop{false};
};

static GLFWwindow *initWindow()
{
    if (!glfwInit())
    {
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow *window = glfwCreateWindow(1000, 700, "NN Training - Dear ImGui", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

static void startTraining(TrainingStats &stats)
{
    std::cout << "Read train data from " << MNIST_TRAIN_DATA_FILE << std::endl;
    auto inputs = NNUtils::read_mnist_data(MNIST_TRAIN_DATA_FILE);
    NNUtils::normalizeMnistData(inputs);

    std::cout << "Read train label from " << MNIST_TRAIN_LABEL_FILE << std::endl;
    auto labels = NNUtils::read_mnist_labels(MNIST_TRAIN_LABEL_FILE);
    NNUtils::normalizeMnistLabel(labels);

    std::cout << "Read test data from " << MNISt_TEST_DATA_FILE << std::endl;
    auto testInputs = NNUtils::read_mnist_data(MNISt_TEST_DATA_FILE);
    NNUtils::normalizeMnistData(testInputs);

    std::cout << "Read test label from " << MNIST_TEST_LABEL_FILE << std::endl;
    auto testLabels = NNUtils::read_mnist_labels(MNIST_TEST_LABEL_FILE);
    NNUtils::normalizeMnistLabel(testLabels);

    std::vector<int> cfg{INPUT_SIZE, HIDDEN1_SIZE, HIDDEN2_SIZE, OUTPUT_SIZE};
    auto nn = NeuralNetwork(cfg);

    NeuralNetwork::TrainCallback callback = [&](int epoch, int totalEpochs, float loss, float accuracy)
    {
        std::lock_guard<std::mutex> lock(stats.mutex);
        stats.loss.push_back(loss);
        stats.acc.push_back(accuracy * 100.0f);
        stats.currentEpoch.store(epoch);
        if (epoch >= totalEpochs)
        {
            stats.done.store(true);
        }
    };

    NeuralNetwork::LayerCallback layerCallback = [&](int epoch, int batch, int layerIndex, NeuralNetwork::LayerPhase phase)
    {
        stats.activeLayer.store(layerIndex);
        stats.activePhase.store(static_cast<int>(phase));
    };

    NeuralNetwork::BatchCallback batchCallback = [&](int epoch, int batch, const NNMatrix &input, const NNMatrix &output)
    {
        std::lock_guard<std::mutex> lock(stats.mutex);
        if (batch % 10 == 0)
        {
            int rows = input.getRowSize();
            stats.currentImage.resize(static_cast<size_t>(rows));
            for (int i = 0; i < rows; ++i)
            {
                stats.currentImage[static_cast<size_t>(i)] = input.get(i, 0);
            }
            stats.currentOutputIndex = output.getIndexOfColMax(0);
            stats.currentOutputValue = output.get(stats.currentOutputIndex, 0);
        }
    };

    NeuralNetwork::StopCallback stopCallback = [&]()
    {
        return stats.stop.load();
    };

    nn.train(inputs, labels, testInputs, testLabels, EPOCHS, BATCH_SIZE, LEARNING_RATE, MOMENTUM, callback, layerCallback, batchCallback, stopCallback);
    stats.activeLayer.store(-1);
    stats.activePhase.store(static_cast<int>(NeuralNetwork::LayerPhase::Idle));
    stats.done.store(true);
}

static void drawInputImage(ImDrawList *drawList, const ImVec2 &origin, const ImVec2 &size, const std::vector<float> &image)
{
    const int width = 28;
    const int height = 28;
    if (image.size() < static_cast<size_t>(width * height))
    {
        drawList->AddText(origin, IM_COL32(200, 200, 210, 255), "Waiting for batch...");
        return;
    }

    const float maxScaleX = size.x / width;
    const float maxScaleY = size.y / height;
    float scale = std::min(maxScaleX, maxScaleY);
    scale = std::floor(scale);
    if (scale < 1.0f)
    {
        scale = 1.0f;
    }
    const float imgW = scale * width;
    const float imgH = scale * height;
    const float offsetX = origin.x + (size.x - imgW) * 0.5f;
    const float offsetY = origin.y + (size.y - imgH) * 0.5f;

    drawList->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const float value = image[static_cast<size_t>(y * width + x)];
            const int intensity = static_cast<int>(value * 255.0f);
            const ImU32 color = IM_COL32(intensity, intensity, intensity, 255);
            const float x0 = offsetX + x * scale;
            const float y0 = offsetY + y * scale;
            drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + scale, y0 + scale), color);
        }
    }

    drawList->AddRect(ImVec2(offsetX, offsetY), ImVec2(offsetX + imgW, offsetY + imgH), IM_COL32(80, 90, 110, 255));
    drawList->PopClipRect();
}

static ImU32 scaleColor(ImU32 color, float scale)
{
    ImColor c(color);
    ImVec4 v = c.Value;
    auto clamp01 = [](float value)
    {
        return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
    };
    v.x = clamp01(v.x * scale);
    v.y = clamp01(v.y * scale);
    v.z = clamp01(v.z * scale);
    return ImColor(v);
}

static void drawDnnTopology(ImDrawList *drawList, const ImVec2 &origin, const ImVec2 &size, int activeLayer, int activePhase)
{
    struct LayerConfig
    {
        int count;
        ImU32 nodeColor;
        ImU32 suspColor;
        const char *name;
        int weightRows;
    };

    const std::array<LayerConfig, 4> layers = {
        LayerConfig{16, IM_COL32(120, 200, 255, 255), IM_COL32(180, 220, 255, 255), "Input", INPUT_SIZE},
        LayerConfig{10, IM_COL32(140, 255, 180, 255), IM_COL32(190, 255, 210, 255), "Hidden 1", HIDDEN1_SIZE},
        LayerConfig{8, IM_COL32(255, 210, 120, 255), IM_COL32(255, 230, 180, 255), "Hidden 2", HIDDEN2_SIZE},
        LayerConfig{4, IM_COL32(255, 140, 140, 255), IM_COL32(255, 190, 190, 255), "Output", OUTPUT_SIZE}};

    const float leftPadding = 40.0f;
    const float rightPadding = 40.0f;
    const float topPadding = 40.0f;
    const float bottomPadding = 40.0f;
    const float nodeRadius = 8.0f;
    const float nodeBorder = 2.0f;
    const float suspRadius = 3.0f;
    const float suspOffset = 14.0f;
    const float labelOffset = 32.0f;
    const float ellipsisSpacing = 9.0f;
    const float ellipsisRadius = 3.0f;

    const float usableW = size.x - leftPadding - rightPadding;
    const float usableH = size.y - topPadding - bottomPadding;
    if (usableW <= 0.0f || usableH <= 0.0f)
    {
        return;
    }

    const float xStep = usableW / static_cast<float>(layers.size() - 1);
    std::vector<std::vector<ImVec2>> layerPositions;
    layerPositions.reserve(layers.size());

    ImFont *font = ImGui::GetFont();
    const float fontSize = ImGui::GetFontSize();

    int maxCount = 1;
    for (const auto &layer : layers)
    {
        if (layer.count > maxCount)
        {
            maxCount = layer.count;
        }
    }

    struct LayerLayout
    {
        float x;
        float top;
        float height;
    };

    std::vector<LayerLayout> layouts;
    layouts.reserve(layers.size());

    for (size_t i = 0; i < layers.size(); ++i)
    {
        const int count = layers[i].count;
        const float heightScale = static_cast<float>(count) / static_cast<float>(maxCount);
        const float layerHeight = usableH * heightScale;
        const float layerTop = origin.y + topPadding + (usableH - layerHeight) * 0.5f;
        const float yStep = count > 1 ? layerHeight / static_cast<float>(count - 1) : 0.0f;
        std::vector<ImVec2> positions;
        positions.reserve(static_cast<size_t>(count));

        const float x = origin.x + leftPadding + xStep * static_cast<float>(i);
        const float centerY = layerTop + layerHeight * 0.5f;
        const bool needsEllipsis = layers[i].weightRows > layers[i].count;
        const float ellipsisGap = needsEllipsis ? std::min(nodeRadius * 3.0f, layerHeight * 0.15f) : 0.0f;

        for (int j = 0; j < count; ++j)
        {
            float y = layerTop + yStep * static_cast<float>(j);
            if (needsEllipsis)
            {
                if (y >= centerY)
                {
                    y = std::min(layerTop + layerHeight, y + ellipsisGap);
                }
                else
                {
                    y = std::max(layerTop, y - ellipsisGap);
                }
            }
            positions.emplace_back(x, y);
        }

        layerPositions.emplace_back(std::move(positions));
        layouts.push_back({x, layerTop, layerHeight});

        char label[64];
        std::snprintf(label, sizeof(label), "%s (%d)", layers[i].name, layers[i].count);
        char rowsLabel[64];
        std::snprintf(rowsLabel, sizeof(rowsLabel), "W rows: %d", layers[i].weightRows);

        ImVec2 textSize = ImGui::CalcTextSize(label);
        ImVec2 rowsSize = ImGui::CalcTextSize(rowsLabel);
        float labelY = origin.y + topPadding - labelOffset - fontSize * 2.0f;
        ImVec2 textPos(x - textSize.x * 0.5f, labelY);
        ImVec2 rowsPos(x - rowsSize.x * 0.5f, labelY + fontSize + 2.0f);
        drawList->AddText(font, fontSize, textPos, IM_COL32(230, 230, 240, 255), label);
        drawList->AddText(font, fontSize, rowsPos, IM_COL32(200, 200, 210, 255), rowsLabel);
    }

    const ImU32 linkColorDim = IM_COL32(60, 60, 70, 120);
    const ImU32 linkColorActive = activePhase == static_cast<int>(NeuralNetwork::LayerPhase::Backward)
                                      ? IM_COL32(255, 190, 140, 200)
                                      : IM_COL32(140, 200, 255, 200);
    for (size_t i = 0; i + 1 < layerPositions.size(); ++i)
    {
        const auto &from = layerPositions[i];
        const auto &to = layerPositions[i + 1];
        const bool highlight = (static_cast<int>(i) == activeLayer) || (static_cast<int>(i + 1) == activeLayer);
        const ImU32 linkColor = highlight ? linkColorActive : linkColorDim;
        for (const auto &p : from)
        {
            for (const auto &q : to)
            {
                drawList->AddLine(p, q, linkColor, 1.0f);
            }
        }
    }

    for (size_t i = 0; i < layerPositions.size(); ++i)
    {
        const auto &positions = layerPositions[i];
        const bool highlight = (static_cast<int>(i) == activeLayer) && (activePhase != static_cast<int>(NeuralNetwork::LayerPhase::Idle));
        const ImU32 nodeColor = highlight ? layers[i].nodeColor : scaleColor(layers[i].nodeColor, 0.35f);
        const ImU32 suspColor = highlight ? layers[i].suspColor : scaleColor(layers[i].suspColor, 0.35f);
        for (const auto &p : positions)
        {
            const ImVec2 suspPoint(p.x, p.y - suspOffset);
            drawList->AddLine(suspPoint, p, IM_COL32(120, 120, 130, highlight ? 220 : 140), 1.0f);
            drawList->AddCircleFilled(suspPoint, suspRadius, suspColor);
            drawList->AddCircleFilled(p, nodeRadius, nodeColor);
            drawList->AddCircle(p, nodeRadius, IM_COL32(30, 30, 35, 255), nodeBorder);
        }

        if (layers[i].weightRows > layers[i].count)
        {
            const float centerY = layouts[i].top + layouts[i].height * 0.5f;
            const float x = layouts[i].x;
            const ImU32 ellipsisColor = nodeColor;
            for (int d = -1; d <= 1; ++d)
            {
                ImVec2 dotPos(x + static_cast<float>(d) * ellipsisSpacing, centerY);
                drawList->AddCircleFilled(dotPos, ellipsisRadius, ellipsisColor);
            }
        }
    }
}

int main(int argc, char **argv)
{
    GLFWwindow *window = initWindow();
    if (!window)
    {
        std::cerr << "Failed to initialize GLFW window" << std::endl;
        return 1;
    }

    const char *glsl_version = "#version 150";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    TrainingStats stats;
    std::thread trainingThread(startTraining, std::ref(stats));

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport *viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGuiWindowFlags mainFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoScrollWithMouse | ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        ImGui::Begin("NN Dashboard", nullptr, mainFlags);

        ImVec2 content = ImGui::GetContentRegionAvail();
        float topHeight = content.y * 0.25f;
        if (topHeight < 240.0f)
        {
            topHeight = 240.0f;
        }

        ImGui::BeginChild("TrainingTop", ImVec2(0.0f, topHeight), false);
        float totalW = ImGui::GetContentRegionAvail().x;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float colW = (totalW - spacing * 2.0f) / 3.0f;

        ImGui::BeginChild("TrainingLeft", ImVec2(colW, 0.0f), true);
        int epoch = stats.currentEpoch.load();
        ImGui::Text("Epoch: %d / %d", epoch, EPOCHS);
        ImGui::Text("Status: %s", stats.done.load() ? "Completed" : "Training...");

        {
            std::lock_guard<std::mutex> lock(stats.mutex);
            if (!stats.loss.empty())
            {
                ImGui::PlotLines("Loss", stats.loss.data(), static_cast<int>(stats.loss.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0, 120));
            }
            else
            {
                ImGui::Text("Loss: waiting for data...");
            }

            if (!stats.acc.empty())
            {
                ImGui::PlotLines("Accuracy (%)", stats.acc.data(), static_cast<int>(stats.acc.size()), 0, nullptr, 0.0f, 100.0f, ImVec2(0, 120));
            }
            else
            {
                ImGui::Text("Accuracy: waiting for data...");
            }
        }
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("TrainingMiddle", ImVec2(colW, 0.0f), true);
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 210, 120, 255));
        ImGui::SetWindowFontScale(1.6f);
        ImGui::Text("Training input");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();
        ImVec2 imgPos = ImGui::GetCursorScreenPos();
        ImVec2 imgSize = ImGui::GetContentRegionAvail();
        ImGui::InvisibleButton("batch_input_canvas", imgSize);
        ImDrawList *imgDraw = ImGui::GetWindowDrawList();
        imgDraw->AddRectFilled(imgPos, ImVec2(imgPos.x + imgSize.x, imgPos.y + imgSize.y), IM_COL32(18, 20, 26, 255));
        imgDraw->AddRect(imgPos, ImVec2(imgPos.x + imgSize.x, imgPos.y + imgSize.y), IM_COL32(70, 80, 90, 255));
        {
            std::lock_guard<std::mutex> lock(stats.mutex);
            drawInputImage(imgDraw, imgPos, imgSize, stats.currentImage);
        }
        ImGui::EndChild();

        ImGui::SameLine();
        ImGui::BeginChild("TrainingRight", ImVec2(colW, 0.0f), true);
        int outputIndex = -1;
        float outputValue = 0.0f;
        {
            std::lock_guard<std::mutex> lock(stats.mutex);
            outputIndex = stats.currentOutputIndex;
            outputValue = stats.currentOutputValue;
        }
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 210, 120, 255));
        ImGui::SetWindowFontScale(1.6f);
        if (outputIndex >= 0)
        {
            ImGui::Text("Training output: %d", outputIndex);
            ImGui::Text("Score: %.4f", outputValue);
        }
        else
        {
            ImGui::Text("Waiting for batch...");
        }
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();
        ImGui::EndChild();

        ImGui::EndChild();

        ImGui::Spacing();
        ImGui::BeginChild("Network Topology", ImVec2(0.0f, 0.0f), true);
        ImVec2 canvasPos = ImGui::GetCursorScreenPos();
        ImVec2 canvasSize = ImGui::GetContentRegionAvail();
        ImGui::InvisibleButton("network_canvas", canvasSize);
        ImDrawList *drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(20, 22, 28, 255));
        drawList->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y), IM_COL32(70, 80, 90, 255));
        int activeLayer = stats.activeLayer.load();
        int activePhase = stats.activePhase.load();
        drawDnnTopology(drawList, canvasPos, canvasSize, activeLayer, activePhase);
        ImGui::EndChild();

        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.12f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        glfwSwapBuffers(window);
    }

    stats.stop.store(true);

    if (trainingThread.joinable())
    {
        trainingThread.join();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
