#include "CNNGuiUtils.h"

#include "CNN.h"
#include "CNNConfigBuilder.h"
#include "DefaultCNNConfig.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"

#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>

namespace {
constexpr int kImageSide = 32;
constexpr int kInputSize = kImageSide * kImageSide;
constexpr int kInChannels = CIFAR100_CNN_IN_CHANNELS;
constexpr int kOutputSize = CIFAR100_CNN_OUTPUT_SIZE;

constexpr int kEpochs = CIFAR100_CNN_EPOCHS;
constexpr int kBatchSize = CIFAR100_CNN_BATCH_SIZE;
const float kLearningRate = CIFAR100_CNN_LEARNING_RATE;
const float kMomentum = CIFAR100_CNN_MOMENTUM;

constexpr int kFlattenSize = CIFAR100_CNN_FC1_IN_SIZE;
constexpr int kFcHiddenSize = CIFAR100_CNN_FC1_OUT_SIZE;

static float clampf(float value, float lo, float hi) {
    if (value < lo) {
        return lo;
    }
    if (value > hi) {
        return hi;
    }
    return value;
}

static ImU32 scaleColor(ImU32 color, float scale) {
    ImColor c(color);
    ImVec4 v = c.Value;
    auto clamp01 = [](float value) { return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value); };
    v.x = clamp01(v.x * scale);
    v.y = clamp01(v.y * scale);
    v.z = clamp01(v.z * scale);
    return ImColor(v);
}

static void drawCnnTopology(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                            int activeLayer, int activePhase) {
    struct Block {
        const char* name;
        ImU32 color;
    };

    // This view corresponds to the CIFAR-100 CNN architecture used in startTraining().
    const std::array<Block, 6> blocks = {
        Block{"Input (32x32x3)", IM_COL32(120, 200, 255, 255)},
        Block{"Conv1 (3->16)", IM_COL32(140, 255, 180, 255)},
        Block{"Conv2 (16->32)", IM_COL32(120, 255, 210, 255)},
        Block{"MaxPool (2x2)", IM_COL32(255, 210, 120, 255)},
        Block{"FC (8192->256)", IM_COL32(255, 160, 120, 255)},
        Block{"Output (100)", IM_COL32(255, 140, 140, 255)},
    };

    const float leftPadding = 50.0f;
    const float rightPadding = 50.0f;
    const float topPadding = 60.0f;
    const float bottomPadding = 60.0f;

    const float usableW = size.x - leftPadding - rightPadding;
    const float usableH = size.y - topPadding - bottomPadding;
    if (usableW <= 0.0f || usableH <= 0.0f) {
        return;
    }

    const float xStep = usableW / static_cast<float>(blocks.size() - 1);
    const float y = origin.y + topPadding + usableH * 0.45f;

    const float nodeR = 10.0f;
    const float nodeBorder = 2.0f;

    const ImU32 linkColorDim = IM_COL32(60, 60, 70, 160);
    const ImU32 linkColorActive = activePhase == static_cast<int>(CNN::LayerPhase::Backward)
                                      ? IM_COL32(255, 190, 140, 220)
                                      : IM_COL32(140, 200, 255, 220);

    // Map CNN layer indices (0..4) to blocks (Conv1=1, Conv2=2, Pool=3, FC=4, Output=5).
    auto blockIsActive = [&](int blockIndex) {
        if (activePhase == static_cast<int>(CNN::LayerPhase::Idle)) {
            return false;
        }
        if (blockIndex <= 0) {
            return false;
        }
        return activeLayer == (blockIndex - 1);
    };

    std::array<ImVec2, 6> centers;
    for (size_t i = 0; i < blocks.size(); ++i) {
        centers[i] = ImVec2(origin.x + leftPadding + static_cast<float>(i) * xStep, y);
    }

    for (size_t i = 0; i + 1 < centers.size(); ++i) {
        const bool highlight =
            blockIsActive(static_cast<int>(i)) || blockIsActive(static_cast<int>(i + 1));
        drawList->AddLine(centers[i], centers[i + 1], highlight ? linkColorActive : linkColorDim,
                          2.0f);
    }

    ImFont* font = ImGui::GetFont();
    const float fontSize = ImGui::GetFontSize();

    for (size_t i = 0; i < blocks.size(); ++i) {
        const bool highlight = blockIsActive(static_cast<int>(i));
        const ImU32 color = highlight ? blocks[i].color : scaleColor(blocks[i].color, 0.35f);
        drawList->AddCircleFilled(centers[i], nodeR, color);
        drawList->AddCircle(centers[i], nodeR, IM_COL32(30, 30, 35, 255), 0, nodeBorder);

        const char* label = blocks[i].name;
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        const float textX = centers[i].x - textSize.x * 0.5f;
        const float textY = centers[i].y + nodeR + 18.0f;
        drawList->AddText(font, fontSize, ImVec2(textX, textY), IM_COL32(230, 230, 240, 255),
                          label);
    }
}

} // namespace

struct CNNGuiUtils::TrainingStats {
    std::vector<float> loss;
    std::vector<float> acc;
    std::mutex mutex;
    std::atomic<int> currentEpoch{0};
    std::atomic<int> currentBatch{-1};
    std::atomic<int> totalBatches{0};
    std::atomic<float> batchLoss{NAN};
    std::atomic<float> epochLoss{NAN};
    std::atomic<float> batchAccuracy{NAN};
    std::atomic<float> epochAccuracy{NAN};
    std::atomic<bool> done{false};
    std::atomic<int> activeLayer{-1};
    std::atomic<int> activePhase{0};
    std::vector<float> currentImage;
    int currentOutputIndex = -1;
    float currentOutputValue = 0.0f;
    std::atomic<bool> stop{false};
};

GLFWwindow* CNNGuiUtils::initWindow() {
    if (!glfwInit()) {
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1000, 700, "CNN Training", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

void CNNGuiUtils::startTraining(TrainingStats& stats) {
    auto dataset = NNDatasetManager::loadCifar100();

    // Keep the GUI run small by default, consistent with main.cpp.
    if (CIFAR100_CNN_MAX_TRAIN_SAMPLES > 0 &&
        static_cast<int>(dataset.trainLabel_.size()) > CIFAR100_CNN_MAX_TRAIN_SAMPLES) {
        dataset.trainLabel_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TRAIN_SAMPLES));
        dataset.trainInput_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TRAIN_SAMPLES) *
                                   static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
    }
    if (CIFAR100_CNN_MAX_TEST_SAMPLES > 0 &&
        static_cast<int>(dataset.testLabel_.size()) > CIFAR100_CNN_MAX_TEST_SAMPLES) {
        dataset.testLabel_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TEST_SAMPLES));
        dataset.testInput_.resize(static_cast<size_t>(CIFAR100_CNN_MAX_TEST_SAMPLES) *
                                  static_cast<size_t>(CIFAR100_CNN_IN_CHANNELS));
    }

    CNNConfigBuilder builder;
    auto configs =
        builder
            .addConvolution(CIFAR100_CNN_IN_CHANNELS, CIFAR100_CNN_CONV1_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            .addBatchNorm(CIFAR100_CNN_CONV1_OUT_CHANNELS)
            .addConvolution(CIFAR100_CNN_CONV1_OUT_CHANNELS, CIFAR100_CNN_CONV2_OUT_CHANNELS,
                            CIFAR100_CNN_CONV_FILTER_SIZE, CIFAR100_CNN_CONV_STRIDE_SIZE,
                            CIFAR100_CNN_CONV_PADDING_SIZE)
            .addBatchNorm(CIFAR100_CNN_CONV2_OUT_CHANNELS)
            .addMaxPooling(CIFAR100_CNN_POOLING_FILTER_SIZE, CIFAR100_CNN_POOLING_STRIDE_SIZE)
            .addFullyConnected(kFlattenSize, kFcHiddenSize)
            .addFullyConnected(kFcHiddenSize, kOutputSize)
            .build();

    auto cnn = CNN(configs);

    CNN::TrainCallback callback = [&](int epoch, int totalEpochs, float loss, float accuracy) {
        std::lock_guard<std::mutex> lock(stats.mutex);
        stats.loss.push_back(loss);
        stats.acc.push_back(accuracy * 100.0f);
        stats.currentEpoch.store(epoch);
        stats.epochLoss.store(loss);
        stats.epochAccuracy.store(accuracy);
        if (epoch >= totalEpochs) {
            stats.done.store(true);
        }
    };

    CNN::LayerCallback layerCallback = [&](int epoch, int batch, int layerIndex,
                                           CNN::LayerPhase phase) {
        (void) epoch;
        (void) batch;
        stats.activeLayer.store(layerIndex);
        stats.activePhase.store(static_cast<int>(phase));
    };

    CNN::BatchCallback batchCallback = [&](int epoch, int batch, const NNMatrixPtrV& input,
                                           const NNMatrix& output) {
        (void) epoch;
        std::lock_guard<std::mutex> lock(stats.mutex);
        if (batch % 10 != 0) {
            return;
        }
        if (input.empty() || !input[0]) {
            return;
        }

        // Render RGB as a grayscale composite in the existing single-image pane.
        stats.currentImage.assign(static_cast<size_t>(kInputSize), 0.0f);
        if (static_cast<int>(input.size()) >= kInChannels && input[0] && input[1] && input[2] &&
            input[0]->getRowSize() == kImageSide && input[0]->getColSize() == kImageSide &&
            input[1]->getRowSize() == kImageSide && input[1]->getColSize() == kImageSide &&
            input[2]->getRowSize() == kImageSide && input[2]->getColSize() == kImageSide) {
            for (int r = 0; r < kImageSide; ++r) {
                for (int c = 0; c < kImageSide; ++c) {
                    const float v =
                        (input[0]->get(r, c) + input[1]->get(r, c) + input[2]->get(r, c)) / 3.0f;
                    stats.currentImage[static_cast<size_t>(r * kImageSide + c)] = v;
                }
            }
        } else {
            const auto& m = input[0];
            if (m->getRowSize() == kImageSide && m->getColSize() == kImageSide) {
                for (int r = 0; r < kImageSide; ++r) {
                    for (int c = 0; c < kImageSide; ++c) {
                        stats.currentImage[static_cast<size_t>(r * kImageSide + c)] = m->get(r, c);
                    }
                }
            }
        }

        stats.currentOutputIndex = output.getIndexOfColMax(0);
        stats.currentOutputValue = output.get(stats.currentOutputIndex, 0);
    };

    CNN::BatchStatsCallback batchStatsCallback = [&](int epoch, int totalEpochs, int batch,
                                                     int totalBatches, float batchLoss,
                                                     float epochLoss, float batchAccuracy) {
        (void) totalEpochs;
        const int prevEpoch = stats.currentEpoch.load();
        if (epoch != prevEpoch) {
            stats.epochAccuracy.store(NAN);
        }
        stats.currentEpoch.store(epoch);
        stats.currentBatch.store(batch);
        stats.totalBatches.store(totalBatches);
        stats.batchLoss.store(batchLoss);
        stats.epochLoss.store(epochLoss);
        stats.batchAccuracy.store(batchAccuracy);

        const float prevEpochAcc = stats.epochAccuracy.load();
        float runningEpochAcc = batchAccuracy;
        if (batch > 1 && std::isfinite(prevEpochAcc)) {
            runningEpochAcc = (prevEpochAcc * static_cast<float>(batch - 1) + batchAccuracy) /
                              static_cast<float>(batch);
        }
        stats.epochAccuracy.store(runningEpochAcc);
    };

    CNN::StopCallback stopCallback = [&]() { return stats.stop.load(); };

    cnn.train(dataset, kEpochs, kBatchSize, kLearningRate, kMomentum, CIFAR100_CNN_WEIGHT_DECAY,
              callback, layerCallback, batchCallback, stopCallback, batchStatsCallback);

    stats.activeLayer.store(-1);
    stats.activePhase.store(static_cast<int>(CNN::LayerPhase::Idle));
    stats.done.store(true);
}

void CNNGuiUtils::drawInputImage(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                                 const std::vector<float>& image) {
    const int width = kImageSide;
    const int height = kImageSide;
    if (image.size() < static_cast<size_t>(width * height)) {
        drawList->AddText(origin, IM_COL32(200, 200, 210, 255), "Waiting for batch...");
        return;
    }

    const float maxScaleX = size.x / width;
    const float maxScaleY = size.y / height;
    float scale = std::min(maxScaleX, maxScaleY);
    scale = std::floor(scale);
    if (scale < 1.0f) {
        scale = 1.0f;
    }
    const float imgW = scale * width;
    const float imgH = scale * height;
    const float offsetX = origin.x + (size.x - imgW) * 0.5f;
    const float offsetY = origin.y + (size.y - imgH) * 0.5f;

    drawList->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const float value = image[static_cast<size_t>(y * width + x)];
            const int intensity = static_cast<int>(clampf(value, 0.0f, 1.0f) * 255.0f);
            const ImU32 color = IM_COL32(intensity, intensity, intensity, 255);
            const float x0 = offsetX + x * scale;
            const float y0 = offsetY + y * scale;
            drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + scale, y0 + scale), color);
        }
    }

    drawList->AddRect(ImVec2(offsetX, offsetY), ImVec2(offsetX + imgW, offsetY + imgH),
                      IM_COL32(80, 90, 110, 255));
    drawList->PopClipRect();
}

int CNNGuiUtils::RunTrainingGui() {
    GLFWwindow* window = initWindow();
    if (!window) {
        std::cerr << "Failed to initialize GLFW window" << std::endl;
        return 1;
    }

    const char* glsl_version = "#version 150";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    TrainingStats stats;
    std::thread trainingThread(startTraining, std::ref(stats));

    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGuiViewport* viewport = ImGui::GetMainViewport();
        ImGui::SetNextWindowPos(viewport->Pos);
        ImGui::SetNextWindowSize(viewport->Size);
        ImGuiWindowFlags mainFlags =
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse |
            ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoNavFocus;
        ImGui::Begin("NN Dashboard", nullptr, mainFlags);

        ImVec2 content = ImGui::GetContentRegionAvail();
        float topHeight = content.y * 0.20f;
        if (topHeight < 220.0f) {
            topHeight = 220.0f;
        }
        if (topHeight > content.y - 220.0f) {
            topHeight = std::max(0.0f, content.y - 220.0f);
        }

        ImGui::BeginChild("TrainingTop", ImVec2(0.0f, topHeight), false);
        float totalW = ImGui::GetContentRegionAvail().x;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float colW = (totalW - spacing * 2.0f) / 3.0f;

        ImGui::BeginChild("TrainingLeft", ImVec2(colW, 0.0f), true);
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 210, 120, 255));
        ImGui::SetWindowFontScale(1.6f);
        ImGui::Text("Progress");
        ImGui::SetWindowFontScale(1.0f);
        ImGui::PopStyleColor();
        ImGui::Spacing();

        const int epoch = stats.currentEpoch.load();
        const int batch = stats.currentBatch.load();
        const int totalBatches = stats.totalBatches.load();
        const float batchLoss = stats.batchLoss.load();
        const float epochLoss = stats.epochLoss.load();
        const float batchAcc = stats.batchAccuracy.load();
        const float epochAcc = stats.epochAccuracy.load();

        if (epoch > 0) {
            ImGui::Text("Epoch: %d/%d", epoch, kEpochs);
        } else {
            ImGui::Text("Epoch: .../%d", kEpochs);
        }

        if (batch > 0 && totalBatches > 0) {
            ImGui::Text("Batch: %d/%d", batch, totalBatches);
        } else {
            ImGui::Text("Batch: .../...");
        }

        if (std::isfinite(batchLoss)) {
            ImGui::Text("Batch Loss: %.6f", batchLoss);
        } else {
            ImGui::Text("Batch Loss: ...");
        }

        if (std::isfinite(epochLoss)) {
            ImGui::Text("Epoc Loss: %.6f", epochLoss);
        } else {
            ImGui::Text("Epoc Loss: ...");
        }

        if (std::isfinite(batchAcc)) {
            ImGui::Text("Batch Accuracy: %.2f%%", batchAcc * 100.0f);
        } else {
            ImGui::Text("Batch Accuracy: ...");
        }

        if (std::isfinite(epochAcc)) {
            ImGui::Text("Epoc Accuracy: %.2f%%", epochAcc * 100.0f);
        } else {
            ImGui::Text("Epoc Accuracy: ...");
        }

        ImGui::Text("Status: %s", stats.done.load() ? "Done" : "Training");
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
        ImDrawList* imgDraw = ImGui::GetWindowDrawList();
        imgDraw->AddRectFilled(imgPos, ImVec2(imgPos.x + imgSize.x, imgPos.y + imgSize.y),
                               IM_COL32(18, 20, 26, 255));
        {
            std::lock_guard<std::mutex> lock(stats.mutex);
            drawInputImage(imgDraw, imgPos, imgSize, stats.currentImage);
        }
        imgDraw->AddRect(imgPos, ImVec2(imgPos.x + imgSize.x, imgPos.y + imgSize.y),
                         IM_COL32(70, 80, 90, 255));
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
        if (outputIndex >= 0) {
            ImGui::Text("Training output: %d", outputIndex);
            ImGui::Text("Score: %.4f", outputValue);
        } else {
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
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        drawList->AddRectFilled(canvasPos,
                                ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                                IM_COL32(20, 22, 28, 255));
        drawList->AddRect(canvasPos, ImVec2(canvasPos.x + canvasSize.x, canvasPos.y + canvasSize.y),
                          IM_COL32(70, 80, 90, 255));
        int activeLayer = stats.activeLayer.load();
        int activePhase = stats.activePhase.load();
        drawCnnTopology(drawList, canvasPos, canvasSize, activeLayer, activePhase);
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

    if (trainingThread.joinable()) {
        trainingThread.join();
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
