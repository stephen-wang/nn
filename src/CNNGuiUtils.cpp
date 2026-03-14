#include "CNNGuiUtils.h"

#include "CNN.h"
#include "DefaultCNNConfig.h"
#include "NNDatasetManager.h"
#include "NNUtils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define GL_SILENCE_DEPRECATION
#define GLFW_INCLUDE_NONE
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include "imgui.h"

#include <GLFW/glfw3.h>
#include <OpenGL/gl3.h>

namespace {
void configureGuiFonts() {
    ImGuiIO& io = ImGui::GetIO();
    const ImWchar* chineseGlyphRanges = io.Fonts->GetGlyphRangesChineseFull();

    const std::array<const char*, 5> fontCandidates = {
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Songti.ttc",
    };

    ImFontConfig fontConfig;
    fontConfig.OversampleH = 2;
    fontConfig.OversampleV = 2;
    fontConfig.PixelSnapH = true;

    for (const char* fontPath : fontCandidates) {
        if (!std::filesystem::exists(fontPath)) {
            continue;
        }
        if (ImFont* font =
                io.Fonts->AddFontFromFileTTF(fontPath, 18.0f, &fontConfig, chineseGlyphRanges)) {
            io.FontDefault = font;
            return;
        }
    }

    io.Fonts->AddFontDefault();
}

inline float toDisplayUnit(float value) {
    if (value < 0.0f || value > 1.0f) {
        value = (value + 1.0f) * 0.5f;
    }
    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}
constexpr int kImageSide = 32;
constexpr int kInputSize = kImageSide * kImageSide;
constexpr int kInChannels = CIFAR100_CNN_IN_CHANNELS;
constexpr float kOperationFrameHeight = 285.0f;

constexpr int kEpochs = CIFAR100_CNN_EPOCHS;
constexpr int kBatchSize = CIFAR100_CNN_BATCH_SIZE;
const float kLearningRate = CIFAR100_CNN_LEARNING_RATE;
const float kMomentum = CIFAR100_CNN_MOMENTUM;

static std::string translateCifar100FineLabelToChinese(const std::string& label) {
    static const std::array<std::pair<const char*, const char*>, 100> kLabelMap = {{
        {"apple", "苹果"},
        {"aquarium_fish", "观赏鱼"},
        {"baby", "婴儿"},
        {"bear", "熊"},
        {"beaver", "海狸"},
        {"bed", "床"},
        {"bee", "蜜蜂"},
        {"beetle", "甲虫"},
        {"bicycle", "自行车"},
        {"bottle", "瓶子"},
        {"bowl", "碗"},
        {"boy", "男孩"},
        {"bridge", "桥"},
        {"bus", "公交车"},
        {"butterfly", "蝴蝶"},
        {"camel", "骆驼"},
        {"can", "罐头"},
        {"castle", "城堡"},
        {"caterpillar", "毛毛虫"},
        {"cattle", "牛"},
        {"chair", "椅子"},
        {"chimpanzee", "黑猩猩"},
        {"clock", "时钟"},
        {"cloud", "云"},
        {"cockroach", "蟑螂"},
        {"couch", "沙发"},
        {"crab", "螃蟹"},
        {"crocodile", "鳄鱼"},
        {"cup", "杯子"},
        {"dinosaur", "恐龙"},
        {"dolphin", "海豚"},
        {"elephant", "大象"},
        {"flatfish", "比目鱼"},
        {"forest", "森林"},
        {"fox", "狐狸"},
        {"girl", "女孩"},
        {"hamster", "仓鼠"},
        {"house", "房子"},
        {"kangaroo", "袋鼠"},
        {"keyboard", "键盘"},
        {"lamp", "台灯"},
        {"lawn_mower", "割草机"},
        {"leopard", "豹"},
        {"lion", "狮子"},
        {"lizard", "蜥蜴"},
        {"lobster", "龙虾"},
        {"man", "男人"},
        {"maple_tree", "枫树"},
        {"motorcycle", "摩托车"},
        {"mountain", "山"},
        {"mouse", "老鼠"},
        {"mushroom", "蘑菇"},
        {"oak_tree", "橡树"},
        {"orange", "橙子"},
        {"orchid", "兰花"},
        {"otter", "水獭"},
        {"palm_tree", "棕榈树"},
        {"pear", "梨"},
        {"pickup_truck", "皮卡车"},
        {"pine_tree", "松树"},
        {"plain", "平原"},
        {"plate", "盘子"},
        {"poppy", "罂粟花"},
        {"porcupine", "豪猪"},
        {"possum", "负鼠"},
        {"rabbit", "兔子"},
        {"raccoon", "浣熊"},
        {"ray", "鳐鱼"},
        {"road", "道路"},
        {"rocket", "火箭"},
        {"rose", "玫瑰"},
        {"sea", "海洋"},
        {"seal", "海豹"},
        {"shark", "鲨鱼"},
        {"shrew", "鼩鼱"},
        {"skunk", "臭鼬"},
        {"skyscraper", "摩天大楼"},
        {"snail", "蜗牛"},
        {"snake", "蛇"},
        {"spider", "蜘蛛"},
        {"squirrel", "松鼠"},
        {"streetcar", "有轨电车"},
        {"sunflower", "向日葵"},
        {"sweet_pepper", "甜椒"},
        {"table", "桌子"},
        {"tank", "坦克"},
        {"telephone", "电话"},
        {"television", "电视"},
        {"tiger", "老虎"},
        {"tractor", "拖拉机"},
        {"train", "火车"},
        {"trout", "鳟鱼"},
        {"tulip", "郁金香"},
        {"turtle", "乌龟"},
        {"wardrobe", "衣柜"},
        {"whale", "鲸鱼"},
        {"willow_tree", "柳树"},
        {"wolf", "狼"},
        {"woman", "女人"},
        {"worm", "蠕虫"},
    }};

    for (const auto& item : kLabelMap) {
        if (label == item.first) {
            return item.second;
        }
    }
    return label;
}

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

static void drawSplitSquare(ImDrawList* drawList, const ImVec2& topLeft, float side, ImU32 color,
                            ImU32 borderColor, int splitCount) {
    const ImVec2 bottomRight(topLeft.x + side, topLeft.y + side);
    drawList->AddRectFilled(topLeft, bottomRight, color, 2.0f);
    drawList->AddRect(topLeft, bottomRight, borderColor, 2.0f, 0, 1.5f);

    if (splitCount > 1) {
        const float step = side / static_cast<float>(splitCount);
        for (int i = 1; i < splitCount; ++i) {
            const float x = topLeft.x + step * static_cast<float>(i);
            const float y = topLeft.y + step * static_cast<float>(i);
            drawList->AddLine(ImVec2(x, topLeft.y), ImVec2(x, bottomRight.y), borderColor, 1.0f);
            drawList->AddLine(ImVec2(topLeft.x, y), ImVec2(bottomRight.x, y), borderColor, 1.0f);
        }
    }
}

static void drawDottedLine(ImDrawList* drawList, const ImVec2& start, const ImVec2& end,
                           ImU32 color, float thickness, float dotLength, float gapLength) {
    const float dx = end.x - start.x;
    const float dy = end.y - start.y;
    const float length = std::sqrt(dx * dx + dy * dy);
    if (length <= 0.5f) {
        return;
    }

    const float ux = dx / length;
    const float uy = dy / length;
    const float patternLength = std::max(0.1f, dotLength + gapLength);
    for (float d = 0.0f; d < length; d += patternLength) {
        const float d1 = std::min(d + dotLength, length);
        const ImVec2 p0(start.x + ux * d, start.y + uy * d);
        const ImVec2 p1(start.x + ux * d1, start.y + uy * d1);
        drawList->AddLine(p0, p1, color, thickness);
    }
}

static void drawSigmaGlyph(ImDrawList* drawList, const ImVec2& topLeft, float side, ImU32 color) {
    const float pad = side * 0.22f;
    const float left = topLeft.x + pad;
    const float right = topLeft.x + side - pad;
    const float top = topLeft.y + pad;
    const float bottom = topLeft.y + side - pad;
    const float midY = (top + bottom) * 0.5f;
    const float thickness = std::max(1.1f, side * 0.08f);

    drawList->AddLine(ImVec2(right, top), ImVec2(left, top), color, thickness);
    drawList->AddLine(ImVec2(left, top), ImVec2(right - side * 0.18f, midY), color, thickness);
    drawList->AddLine(ImVec2(right - side * 0.18f, midY), ImVec2(left, bottom), color, thickness);
    drawList->AddLine(ImVec2(left, bottom), ImVec2(right, bottom), color, thickness);
}

static void drawInputLayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color) {
    constexpr float side = 62.0f;
    constexpr float diagonalStep = 5.0f;
    const ImU32 borderColor = IM_COL32(30, 30, 35, 255);

    // Draw back-to-front so square1 is on top of square2, and square2 on top of square3.
    for (int layer = 2; layer >= 0; --layer) {
        const float dx = static_cast<float>(layer - 1) * diagonalStep;
        const float dy = static_cast<float>(layer - 1) * diagonalStep;
        const ImVec2 topLeft(center.x - side * 0.5f + dx, center.y - side * 0.5f + dy);
        drawSplitSquare(drawList, topLeft, side, color, borderColor, 4);
    }
}

static float stackDepthStepForChannels(int channels) {
    if (channels <= 4) {
        return 4.0f;
    }
    if (channels <= 16) {
        return 2.0f;
    }
    return 1.2f;
}

static ImVec2 drawFeatureMapStack(ImDrawList* drawList, const ImVec2& center, float side,
                                  int channels, float depthStep, ImU32 color, ImU32 borderColor) {
    const int safeChannels = std::max(1, channels);
    const float startOffset = -0.5f * depthStep * static_cast<float>(safeChannels - 1);

    for (int layer = safeChannels - 1; layer >= 0; --layer) {
        const float offset = startOffset + depthStep * static_cast<float>(layer);
        const ImVec2 topLeft(center.x - side * 0.5f + offset, center.y - side * 0.5f + offset);
        const bool isFrontLayer = (layer == safeChannels - 1);
        drawSplitSquare(drawList, topLeft, side, color, borderColor, isFrontLayer ? 32 : 0);
    }

    const float frontOffset = startOffset + depthStep * static_cast<float>(safeChannels - 1);
    return ImVec2(center.x - side * 0.5f + frontOffset, center.y - side * 0.5f + frontOffset);
}

static ImVec2 drawFeatureMapStackWithFrontSplit(ImDrawList* drawList, const ImVec2& center,
                                                float side, int channels, float depthStep,
                                                ImU32 color, ImU32 borderColor,
                                                int frontSplitCount) {
    const int safeChannels = std::max(1, channels);
    const float startOffset = -0.5f * depthStep * static_cast<float>(safeChannels - 1);

    for (int layer = safeChannels - 1; layer >= 0; --layer) {
        const float offset = startOffset + depthStep * static_cast<float>(layer);
        const ImVec2 topLeft(center.x - side * 0.5f + offset, center.y - side * 0.5f + offset);
        const bool isFrontLayer = (layer == safeChannels - 1);
        drawSplitSquare(drawList, topLeft, side, color, borderColor,
                        isFrontLayer ? frontSplitCount : 0);
    }

    const float frontOffset = startOffset + depthStep * static_cast<float>(safeChannels - 1);
    return ImVec2(center.x - side * 0.5f + frontOffset, center.y - side * 0.5f + frontOffset);
}

static float topologyActivityScale(ImU32 topologyColor) {
    const ImVec4 v = ImColor(topologyColor).Value;
    const float maxComponent = std::max(v.x, std::max(v.y, v.z));
    return maxComponent < 0.5f ? 0.38f : 1.0f;
}

static void drawConvLayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 leftColor,
                               ImU32 rightColor, ImU32 frameFillColor, ImU32 frameBorderColor,
                               ImU32 kernelColor, ImU32 kernelBorderColor, int inputChannels,
                               int outputChannels, const char* kernelSymbol,
                               ImU32 kernelSymbolColor, bool rightToLeftFlow,
                               bool anchorOutputToFront, bool showOutputSquares,
                               bool showConnectionLine, double animTimeSec,
                               double scanStepsPerSecond) {
    constexpr float operationFrameWidth = 200.0f;
    constexpr float operationFrameHeight = 285.0f;
    constexpr float mapSide = 72.0f;
    constexpr float kernelCells = 9.0f;
    constexpr float kernelScale = 0.5f;
    constexpr int kernelSplitCount = 3;
    const ImU32 borderColor = IM_COL32(30, 30, 35, 255);

    const ImVec2 frameTopLeft(center.x - operationFrameWidth * 0.5f,
                              center.y - operationFrameHeight * 0.5f + 6.0f);
    const ImVec2 frameBottomRight(frameTopLeft.x + operationFrameWidth,
                                  frameTopLeft.y + operationFrameHeight);
    drawList->AddRectFilled(frameTopLeft, frameBottomRight, frameFillColor, 6.0f);
    drawList->AddRect(frameTopLeft, frameBottomRight, frameBorderColor, 6.0f, 0, 1.2f);

    const float leftAnchorX = frameTopLeft.x + 94.0f;
    const float rightAnchorX = frameBottomRight.x - 94.0f;
    const float centerAnchorX = (frameTopLeft.x + frameBottomRight.x) * 0.5f;
    const float drawCenterY = center.y + 8.0f;
    const bool singleVisibleStack = !showOutputSquares;
    const ImVec2 inputCenter(singleVisibleStack ? centerAnchorX
                                                : (rightToLeftFlow ? rightAnchorX : leftAnchorX),
                             drawCenterY);

    const float inputDepthStep = stackDepthStepForChannels(inputChannels);
    const float outputDepthStep = stackDepthStepForChannels(outputChannels);
    const float inputStartOffset =
        -0.5f * inputDepthStep * static_cast<float>(std::max(1, inputChannels) - 1);
    const float outputStartOffset =
        -0.5f * outputDepthStep * static_cast<float>(std::max(1, outputChannels) - 1);
    const float outputFrontOffset =
        outputStartOffset + outputDepthStep * static_cast<float>(std::max(1, outputChannels) - 1);
    const float outputAnchorOffset = anchorOutputToFront ? outputFrontOffset : outputStartOffset;

    constexpr float kTopSquareGap = 34.0f;
    ImVec2 outputCenter(centerAnchorX, drawCenterY);
    if (!rightToLeftFlow) {
        const float inputTopSquareRightEdge =
            inputCenter.x - mapSide * 0.5f + inputStartOffset + mapSide;
        const float outputTopSquareLeftEdge = inputTopSquareRightEdge + kTopSquareGap;
        outputCenter =
            ImVec2(outputTopSquareLeftEdge - outputAnchorOffset + mapSide * 0.5f, center.y + 8.0f);
    } else {
        const float inputTopSquareLeftEdge = inputCenter.x - mapSide * 0.5f + inputStartOffset;
        const float outputTopSquareRightEdge = inputTopSquareLeftEdge - kTopSquareGap;
        outputCenter =
            ImVec2(outputTopSquareRightEdge - mapSide * 0.5f - outputAnchorOffset, center.y + 8.0f);
    }

    drawFeatureMapStack(drawList, inputCenter, mapSide, inputChannels, inputDepthStep, leftColor,
                        borderColor);
    if (showOutputSquares) {
        drawFeatureMapStack(drawList, outputCenter, mapSide, outputChannels, outputDepthStep,
                            rightColor, borderColor);
    }

    const ImVec2 inputBaseTopLeft(inputCenter.x - mapSide * 0.5f, inputCenter.y - mapSide * 0.5f);
    const ImVec2 inputTopLayerTopLeft(inputBaseTopLeft.x + inputStartOffset,
                                      inputBaseTopLeft.y + inputStartOffset);
    const ImVec2 outputBaseTopLeft(outputCenter.x - mapSide * 0.5f,
                                   outputCenter.y - mapSide * 0.5f);
    const ImVec2 outputTopLayerTopLeft(outputBaseTopLeft.x + outputAnchorOffset,
                                       outputBaseTopLeft.y + outputAnchorOffset);

    const float cellSize = mapSide / 32.0f;
    const float kernelSide = kernelCells * cellSize * kernelScale;
    const int positionsPerAxis =
        std::max(1, static_cast<int>(std::floor(32.0f - kernelCells * kernelScale + 1.0f)));
    const int totalPositions = positionsPerAxis * positionsPerAxis;
    const float stepDurationSec = 1.0f / static_cast<float>(std::max(0.1, scanStepsPerSecond));
    constexpr float kLineVisibleSec = 1.0f;
    constexpr float kPostMoveStopSec = 0.25f;
    const float movePhaseSec = 2.0f * stepDurationSec + kPostMoveStopSec;
    const float cycleSec = kLineVisibleSec + movePhaseSec;
    const float cycleIdF = std::floor(static_cast<float>(animTimeSec) / cycleSec);
    const int cycleId = static_cast<int>(cycleIdF);
    const float cycleTime = static_cast<float>(animTimeSec) - cycleIdF * cycleSec;
    const int baseStep = ((cycleId * 2) % totalPositions + totalPositions) % totalPositions;

    int stepOffset = 0;
    bool showDottedLine = false;
    if (cycleTime < kLineVisibleSec) {
        showDottedLine = true;
    } else {
        const float moveTime = cycleTime - kLineVisibleSec;
        if (moveTime >= stepDurationSec) {
            stepOffset = 1;
        }
        if (moveTime >= 2.0f * stepDurationSec) {
            stepOffset = 2;
        }
    }

    const int wrappedIndex = (baseStep + stepOffset) % totalPositions;
    const int scanRow = wrappedIndex / positionsPerAxis;
    const int scanCol = wrappedIndex % positionsPerAxis;

    constexpr float kKernelInset = 1.0f;
    const float kernelMinX = inputTopLayerTopLeft.x + kKernelInset;
    const float kernelMinY = inputTopLayerTopLeft.y + kKernelInset;
    const float kernelMaxX = inputTopLayerTopLeft.x + mapSide - kernelSide - kKernelInset;
    const float kernelMaxY = inputTopLayerTopLeft.y + mapSide - kernelSide - kKernelInset;
    const float kernelX = clampf(inputTopLayerTopLeft.x + static_cast<float>(scanCol) * cellSize,
                                 kernelMinX, kernelMaxX);
    const float kernelY = clampf(inputTopLayerTopLeft.y + static_cast<float>(scanRow) * cellSize,
                                 kernelMinY, kernelMaxY);
    const ImVec2 kernelTopLeft(kernelX, kernelY);
    drawSplitSquare(drawList, kernelTopLeft, kernelSide, kernelColor, kernelBorderColor,
                    kernelSplitCount);
    if (kernelSymbol != nullptr) {
        drawSigmaGlyph(drawList, kernelTopLeft, kernelSide, kernelSymbolColor);
    }

    if (showConnectionLine && showDottedLine) {
        const float outNormRow = positionsPerAxis > 1 ? static_cast<float>(scanRow) /
                                                            static_cast<float>(positionsPerAxis - 1)
                                                      : 0.0f;
        const float outNormCol = positionsPerAxis > 1 ? static_cast<float>(scanCol) /
                                                            static_cast<float>(positionsPerAxis - 1)
                                                      : 0.0f;
        const int outRowRaw = static_cast<int>(std::round(outNormRow * 31.0f));
        const int outColRaw = static_cast<int>(std::round(outNormCol * 31.0f));
        const int outRow = outRowRaw < 0 ? 0 : (outRowRaw > 31 ? 31 : outRowRaw);
        const int outCol = outColRaw < 0 ? 0 : (outColRaw > 31 ? 31 : outColRaw);
        const ImVec2 kernelCenter(kernelTopLeft.x + kernelSide * 0.5f,
                                  kernelTopLeft.y + kernelSide * 0.5f);
        const ImVec2 outputCellCenter(
            outputTopLayerTopLeft.x + (static_cast<float>(outCol) + 0.5f) * cellSize,
            outputTopLayerTopLeft.y + (static_cast<float>(outRow) + 0.5f) * cellSize);
        constexpr float kEndCircleRadius = 2.2f;
        const float clampedEndX =
            clampf(outputCellCenter.x, outputTopLayerTopLeft.x + kEndCircleRadius,
                   outputTopLayerTopLeft.x + mapSide - kEndCircleRadius);
        const float clampedEndY =
            clampf(outputCellCenter.y, outputTopLayerTopLeft.y + kEndCircleRadius,
                   outputTopLayerTopLeft.y + mapSide - kEndCircleRadius);
        const ImVec2 clampedOutputCellCenter(clampedEndX, clampedEndY);
        const ImU32 dottedColor = scaleColor(kernelBorderColor, 1.35f);
        drawDottedLine(drawList, kernelCenter, clampedOutputCellCenter, dottedColor, 1.6f, 4.0f,
                       3.2f);
        drawList->AddCircleFilled(clampedOutputCellCenter, kEndCircleRadius, dottedColor, 12);
    }
}

static void drawConv1LayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color,
                                double animTimeSec) {
    const float activityScale = topologyActivityScale(color);
    const ImU32 leftBigColor = scaleColor(IM_COL32(110, 225, 150, 245), activityScale);
    const ImU32 rightBigColor = scaleColor(IM_COL32(255, 185, 105, 245), activityScale);
    const ImU32 smallKernelColor = scaleColor(IM_COL32(255, 255, 180, 255), activityScale);
    const ImU32 smallKernelBorderColor = scaleColor(IM_COL32(255, 255, 255, 255), activityScale);
    const ImU32 kernelSymbolColor = scaleColor(IM_COL32(20, 35, 95, 255), activityScale);
    drawConvLayerGlyph(drawList, center, leftBigColor, rightBigColor, IM_COL32(28, 32, 40, 180),
                       IM_COL32(80, 90, 110, 220), smallKernelColor, smallKernelBorderColor, 3, 32,
                       "Σ", kernelSymbolColor, false, false, false, false, animTimeSec, 11.0);
}

static void drawConv2LayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color,
                                double animTimeSec) {
    const float activityScale = topologyActivityScale(color);
    const ImU32 leftBigColor = scaleColor(IM_COL32(105, 160, 255, 245), activityScale);
    const ImU32 rightBigColor = scaleColor(IM_COL32(205, 120, 255, 245), activityScale);
    const ImU32 smallKernelColor = scaleColor(IM_COL32(255, 200, 255, 255), activityScale);
    const ImU32 smallKernelBorderColor = scaleColor(IM_COL32(210, 255, 255, 255), activityScale);
    const ImU32 kernelSymbolColor = scaleColor(IM_COL32(60, 30, 110, 255), activityScale);
    drawConvLayerGlyph(drawList, center, leftBigColor, rightBigColor, IM_COL32(24, 30, 46, 185),
                       IM_COL32(90, 110, 150, 220), smallKernelColor, smallKernelBorderColor, 32,
                       64, "Σ", kernelSymbolColor, false, false, false, false, animTimeSec, 8.5);
}

static void drawConv3LayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color,
                                double animTimeSec) {
    const float activityScale = topologyActivityScale(color);
    const ImU32 leftBigColor = scaleColor(IM_COL32(135, 145, 255, 245), activityScale);
    const ImU32 rightBigColor = scaleColor(IM_COL32(170, 120, 255, 245), activityScale);
    const ImU32 smallKernelColor = scaleColor(IM_COL32(240, 210, 255, 255), activityScale);
    const ImU32 smallKernelBorderColor = scaleColor(IM_COL32(220, 240, 255, 255), activityScale);
    const ImU32 kernelSymbolColor = scaleColor(IM_COL32(40, 30, 95, 255), activityScale);
    drawConvLayerGlyph(drawList, center, leftBigColor, rightBigColor, IM_COL32(25, 29, 44, 185),
                       IM_COL32(86, 102, 150, 220), smallKernelColor, smallKernelBorderColor, 32,
                       32, "Σ", kernelSymbolColor, true, false, false, false, animTimeSec, 7.0);
}

static void drawMaxPoolLayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color,
                                  double animTimeSec, bool rightToLeftFlow,
                                  bool anchorOutputToFront, float topSquareGap,
                                  bool showOutputSquares, bool showConnectionLine) {
    const float activityScale = topologyActivityScale(color);
    const ImU32 leftBigColor = scaleColor(IM_COL32(255, 205, 120, 245), activityScale);
    const ImU32 rightBigColor = scaleColor(IM_COL32(255, 165, 95, 245), activityScale);
    const ImU32 windowColor = scaleColor(IM_COL32(245, 245, 255, 245), activityScale);
    const ImU32 windowBorderColor = scaleColor(IM_COL32(255, 255, 255, 255), activityScale);
    const ImU32 maxCellColor = scaleColor(IM_COL32(255, 255, 150, 255), activityScale);
    const ImU32 mapColor = scaleColor(IM_COL32(255, 250, 235, 255), activityScale);

    constexpr float frameWidth = 200.0f;
    constexpr float frameHeight = kOperationFrameHeight;
    constexpr float mapSide = 72.0f;
    constexpr int inChannels = 32;
    constexpr int outChannels = 32;
    constexpr int inputCellsPerAxis = 32;
    constexpr int outputCellsPerAxis = 16;
    constexpr int poolWindowCells = 2;
    constexpr float scanStepsPerSecond = 4.0f;

    const ImU32 borderColor = IM_COL32(30, 30, 35, 255);
    const ImVec2 frameTopLeft(center.x - frameWidth * 0.5f, center.y - frameHeight * 0.5f + 6.0f);
    const ImVec2 frameBottomRight(frameTopLeft.x + frameWidth, frameTopLeft.y + frameHeight);
    drawList->AddRectFilled(frameTopLeft, frameBottomRight, IM_COL32(38, 33, 24, 185), 6.0f);
    drawList->AddRect(frameTopLeft, frameBottomRight, IM_COL32(140, 118, 85, 220), 6.0f, 0, 1.2f);

    const float leftAnchorX = frameTopLeft.x + 94.0f;
    const float rightAnchorX = frameBottomRight.x - 94.0f;
    const float centerAnchorX = (frameTopLeft.x + frameBottomRight.x) * 0.5f;
    const float drawCenterY = center.y + 8.0f;
    const bool singleVisibleStack = !showOutputSquares;
    const ImVec2 inputCenter(singleVisibleStack ? centerAnchorX
                                                : (rightToLeftFlow ? rightAnchorX : leftAnchorX),
                             drawCenterY);
    const float inputDepthStep = stackDepthStepForChannels(inChannels);
    const float inputStartOffset =
        -0.5f * inputDepthStep * static_cast<float>(std::max(1, inChannels) - 1);

    const float outputDepthStep = stackDepthStepForChannels(outChannels);
    const float outputStartOffset =
        -0.5f * outputDepthStep * static_cast<float>(std::max(1, outChannels) - 1);
    const float outputFrontOffset =
        outputStartOffset + outputDepthStep * static_cast<float>(std::max(1, outChannels) - 1);
    const float outputAnchorOffset = anchorOutputToFront ? outputFrontOffset : outputStartOffset;
    ImVec2 outputCenter(centerAnchorX, drawCenterY);
    if (!rightToLeftFlow) {
        const float inputTopSquareRightEdge =
            inputCenter.x - mapSide * 0.5f + inputStartOffset + mapSide;
        const float outputTopSquareLeftEdge = inputTopSquareRightEdge + topSquareGap;
        outputCenter =
            ImVec2(outputTopSquareLeftEdge - outputAnchorOffset + mapSide * 0.5f, center.y + 8.0f);
    } else {
        const float inputTopSquareLeftEdge = inputCenter.x - mapSide * 0.5f + inputStartOffset;
        const float outputTopSquareRightEdge = inputTopSquareLeftEdge - topSquareGap;
        outputCenter =
            ImVec2(outputTopSquareRightEdge - mapSide * 0.5f - outputAnchorOffset, center.y + 8.0f);
    }

    drawFeatureMapStackWithFrontSplit(drawList, inputCenter, mapSide, inChannels, inputDepthStep,
                                      leftBigColor, borderColor, inputCellsPerAxis);
    if (showOutputSquares) {
        drawFeatureMapStackWithFrontSplit(drawList, outputCenter, mapSide, outChannels,
                                          outputDepthStep, rightBigColor, borderColor,
                                          outputCellsPerAxis);
    }

    const ImVec2 inputTopLayerTopLeft(inputCenter.x - mapSide * 0.5f + inputStartOffset,
                                      inputCenter.y - mapSide * 0.5f + inputStartOffset);
    const ImVec2 outputTopLayerTopLeft(outputCenter.x - mapSide * 0.5f + outputAnchorOffset,
                                       outputCenter.y - mapSide * 0.5f + outputAnchorOffset);

    const float inputCellSize = mapSide / static_cast<float>(inputCellsPerAxis);
    const float outputCellSize = mapSide / static_cast<float>(outputCellsPerAxis);

    const int positionsPerAxis = outputCellsPerAxis;
    const int totalPositions = positionsPerAxis * positionsPerAxis;
    const int animIndex = static_cast<int>(std::floor(animTimeSec * scanStepsPerSecond));
    const int wrappedIndex = ((animIndex % totalPositions) + totalPositions) % totalPositions;
    const int poolRow = wrappedIndex / positionsPerAxis;
    const int poolCol = wrappedIndex % positionsPerAxis;

    const ImVec2 poolWindowRawTopLeft(
        inputTopLayerTopLeft.x + static_cast<float>(poolCol * poolWindowCells) * inputCellSize,
        inputTopLayerTopLeft.y + static_cast<float>(poolRow * poolWindowCells) * inputCellSize);
    const float poolWindowRawSide = static_cast<float>(poolWindowCells) * inputCellSize;
    const float poolWindowSide = std::max(14.0f, poolWindowRawSide) * (4.0f / 3.0f);
    const ImVec2 poolWindowTopLeft(
        poolWindowRawTopLeft.x + (poolWindowRawSide - poolWindowSide) * 0.5f,
        poolWindowRawTopLeft.y + (poolWindowRawSide - poolWindowSide) * 0.5f);
    drawSplitSquare(drawList, poolWindowTopLeft, poolWindowSide, windowColor, windowBorderColor,
                    poolWindowCells);

    const int maxCellIndex = (wrappedIndex + poolRow) % 4;
    const int maxCellRow = maxCellIndex / 2;
    const int maxCellCol = maxCellIndex % 2;
    const float maxCellSide = poolWindowSide * 0.5f;
    const ImVec2 maxCellTopLeft(poolWindowTopLeft.x + static_cast<float>(maxCellCol) * maxCellSide,
                                poolWindowTopLeft.y + static_cast<float>(maxCellRow) * maxCellSide);
    drawList->AddRectFilled(maxCellTopLeft,
                            ImVec2(maxCellTopLeft.x + maxCellSide, maxCellTopLeft.y + maxCellSide),
                            maxCellColor, 0.0f);

    const ImU32 sumLabelColor = scaleColor(IM_COL32(30, 40, 120, 255), activityScale);
    const char* sumText = "max";
    ImFont* font = ImGui::GetFont();
    const float sumFontSize = std::min(ImGui::GetFontSize() * 0.75f, poolWindowSide * 0.52f);
    const ImVec2 sumTextSize = font->CalcTextSizeA(sumFontSize, FLT_MAX, 0.0f, sumText);
    const ImVec2 sumTextPos(poolWindowTopLeft.x + (poolWindowSide - sumTextSize.x) * 0.5f,
                            poolWindowTopLeft.y + (poolWindowSide - sumTextSize.y) * 0.5f);
    drawList->AddText(font, sumFontSize, sumTextPos, sumLabelColor, sumText);

    if (showOutputSquares) {
        const ImVec2 outputCellTopLeft(
            outputTopLayerTopLeft.x + static_cast<float>(poolCol) * outputCellSize,
            outputTopLayerTopLeft.y + static_cast<float>(poolRow) * outputCellSize);
        drawList->AddRectFilled(
            outputCellTopLeft,
            ImVec2(outputCellTopLeft.x + outputCellSize, outputCellTopLeft.y + outputCellSize),
            mapColor, 0.0f);

        if (showConnectionLine) {
            const ImVec2 startPoint(maxCellTopLeft.x + inputCellSize * 0.5f,
                                    maxCellTopLeft.y + inputCellSize * 0.5f);
            const ImVec2 endPoint(outputCellTopLeft.x + outputCellSize * 0.5f,
                                  outputCellTopLeft.y + outputCellSize * 0.5f);
            drawDottedLine(drawList, startPoint, endPoint, scaleColor(windowBorderColor, 1.25f),
                           1.4f, 3.8f, 2.8f);
            drawList->AddCircleFilled(endPoint, 2.0f, scaleColor(windowBorderColor, 1.25f), 12);
        }
    }
}

static void drawFCLayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color,
                             int rightCircleCount, int leftDotCount, ImU32 rightCircleBaseColor,
                             ImU32 leftDotBaseColor, ImU32 lineBaseColor,
                             bool pruneOneThirdConnections, float leftDotHeightScale) {
    const float activityScale = topologyActivityScale(color);
    const ImU32 leftDotColor = scaleColor(leftDotBaseColor, activityScale);
    const ImU32 rightCircleColor = scaleColor(rightCircleBaseColor, activityScale);
    const ImU32 lineColor = scaleColor(lineBaseColor, activityScale);
    const ImU32 borderColor = IM_COL32(30, 30, 35, 255);

    constexpr float leftDotRadius = 2.1f;
    constexpr float rightCircleRadius = 3.0f;

    constexpr float frameWidth = 170.0f;
    constexpr float frameHeight = kOperationFrameHeight;
    const ImVec2 frameTopLeft(center.x - frameWidth * 0.5f, center.y - frameHeight * 0.5f + 6.0f);
    const ImVec2 frameBottomRight(frameTopLeft.x + frameWidth, frameTopLeft.y + frameHeight);
    drawList->AddRectFilled(frameTopLeft, frameBottomRight, IM_COL32(26, 30, 36, 180), 6.0f);
    drawList->AddRect(frameTopLeft, frameBottomRight, IM_COL32(90, 100, 115, 220), 6.0f, 0, 1.2f);

    const float innerTop = frameTopLeft.y + 14.0f;
    const float innerBottom = frameBottomRight.y - 14.0f;
    const float innerSpan = innerBottom - innerTop;
    const float leftSpan = innerSpan * clampf(leftDotHeightScale, 0.2f, 1.0f);
    const float leftInnerTop = innerTop + (innerSpan - leftSpan) * 0.5f;
    const float leftInnerBottom = leftInnerTop + leftSpan;

    auto buildColumn = [&](float x, int count, float top, float bottom) {
        std::vector<ImVec2> nodes;
        nodes.reserve(static_cast<size_t>(count));
        const int safeCount = std::max(1, count);
        const float dy = safeCount > 1 ? (bottom - top) / static_cast<float>(safeCount - 1) : 0.0f;
        for (int i = 0; i < safeCount; ++i) {
            nodes.emplace_back(x, top + static_cast<float>(i) * dy);
        }
        return nodes;
    };

    const float leftColumnX = frameTopLeft.x + 44.0f;      // output dots
    const float rightColumnX = frameBottomRight.x - 44.0f; // input circles

    const std::vector<ImVec2> leftNodes =
        buildColumn(leftColumnX, leftDotCount, leftInnerTop, leftInnerBottom);
    const std::vector<ImVec2> rightNodes =
        buildColumn(rightColumnX, rightCircleCount, innerTop, innerBottom);

    for (size_t rightIndex = 0; rightIndex < rightNodes.size(); ++rightIndex) {
        for (size_t leftIndex = 0; leftIndex < leftNodes.size(); ++leftIndex) {
            const unsigned int hash = (static_cast<unsigned int>(rightIndex + 1) * 73856093u) ^
                                      (static_cast<unsigned int>(leftIndex + 1) * 19349663u) ^
                                      0x9e3779b9u;
            if (pruneOneThirdConnections && hash % 2u == 0u) {
                continue;
            }
            drawList->AddLine(rightNodes[rightIndex], leftNodes[leftIndex], lineColor, 0.75f);
        }
    }

    for (const ImVec2& p : leftNodes) {
        drawList->AddCircleFilled(p, leftDotRadius, leftDotColor);
        drawList->AddCircle(p, leftDotRadius, borderColor, 0, 1.0f);
    }
    for (const ImVec2& p : rightNodes) {
        drawList->AddCircleFilled(p, rightCircleRadius, rightCircleColor);
        drawList->AddCircle(p, rightCircleRadius, borderColor, 0, 1.0f);
    }
}

static void drawOutputLayerGlyph(ImDrawList* drawList, const ImVec2& center, ImU32 color) {
    constexpr float frameWidth = 85.0f;
    constexpr float frameHeight = kOperationFrameHeight;
    constexpr float dotRadius = 3.1f;
    constexpr int dotCount = 10;

    const float activityScale = topologyActivityScale(color);
    const ImU32 dotColor = scaleColor(IM_COL32(255, 145, 145, 245), activityScale);
    const ImU32 borderColor = IM_COL32(30, 30, 35, 255);

    const ImVec2 frameTopLeft(center.x - frameWidth * 0.5f, center.y - frameHeight * 0.5f + 6.0f);
    const ImVec2 frameBottomRight(frameTopLeft.x + frameWidth, frameTopLeft.y + frameHeight);
    drawList->AddRectFilled(frameTopLeft, frameBottomRight, IM_COL32(34, 26, 28, 180), 6.0f);
    drawList->AddRect(frameTopLeft, frameBottomRight, IM_COL32(128, 90, 95, 220), 6.0f, 0, 1.2f);

    const float columnX = center.x;
    const float innerTop = frameTopLeft.y + 18.0f;
    const float innerBottom = frameBottomRight.y - 18.0f;
    const float dy =
        dotCount > 1 ? (innerBottom - innerTop) / static_cast<float>(dotCount - 1) : 0.0f;

    for (int i = 0; i < dotCount; ++i) {
        const ImVec2 p(columnX, innerTop + static_cast<float>(i) * dy);
        drawList->AddCircleFilled(p, dotRadius, dotColor);
        drawList->AddCircle(p, dotRadius, borderColor, 0, 1.0f);
    }
}

static void drawCnnTopology(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                            int activeLayer, int activePhase, double conv1AnimTimeSec) {
    (void) activeLayer;
    (void) activePhase;

    struct Block {
        const char* name;
        ImU32 color;
    };

    // This view corresponds to the CIFAR-100 CNN architecture used in startTraining().
    const std::array<Block, 9> blocks = {
        Block{"Input (32x32x3)", IM_COL32(180, 240, 255, 255)},
        Block{"Cov1 (32x32x32)", IM_COL32(140, 255, 180, 255)},
        Block{"Conv2 (32x32X64)", IM_COL32(120, 255, 210, 255)},
        Block{"MaxPool1 (16x16x64)", IM_COL32(255, 210, 120, 255)},
        Block{"Conv3 (16x16x128)", IM_COL32(150, 210, 255, 255)},
        Block{"MaxPool2 (8x8x128)", IM_COL32(255, 190, 110, 255)},
        Block{"FC1 (8192->256)", IM_COL32(255, 160, 120, 255)},
        Block{"FC2 (256->100)", IM_COL32(255, 170, 145, 255)},
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

    const float yUpper = origin.y + topPadding + usableH * 0.18f;
    const float yLower = origin.y + topPadding + usableH * 0.84f;

    const ImU32 linkColorDim = IM_COL32(60, 60, 70, 160);
    const ImU32 linkColorActive = IM_COL32(140, 200, 255, 220);

    constexpr std::array<int, 9> kHighlightSequence = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    constexpr double kHighlightDurationSec = 1.0;
    int activeBlockIndex = -1;
    if (conv1AnimTimeSec >= 0.0) {
        const int seqIndex =
            static_cast<int>(std::floor(conv1AnimTimeSec / kHighlightDurationSec)) %
            static_cast<int>(kHighlightSequence.size());
        activeBlockIndex = kHighlightSequence[static_cast<size_t>(seqIndex)];
    }

    auto blockIsActive = [&](int blockIndex) { return blockIndex == activeBlockIndex; };

    std::array<ImVec2, 9> centers;
    const float rowLeft = origin.x + leftPadding;
    const float rowRight = origin.x + leftPadding + usableW;

    auto distributeRow = [&](int startIndex, int endIndex, float rowY, bool rightToLeft) {
        const int count = endIndex - startIndex + 1;
        if (count <= 0) {
            return;
        }
        const float step = count > 1 ? (rowRight - rowLeft) / static_cast<float>(count - 1) : 0.0f;
        for (int i = 0; i < count; ++i) {
            const float x = rightToLeft ? (rowRight - static_cast<float>(i) * step)
                                        : (rowLeft + static_cast<float>(i) * step);
            centers[static_cast<size_t>(startIndex + i)] = ImVec2(x, rowY);
        }
    };

    // Upper line: Input/Conv1/Conv2/MaxPool1
    distributeRow(0, 3, yUpper, false);
    // Lower line (right-to-left): Conv3/MaxPool2/FC1/FC2/Output
    distributeRow(4, 8, yLower, true);

    // Shift selected groups left while preserving current intra-group distances.
    constexpr float upperGroupShiftX = 120.0f;
    for (int i = 1; i <= 3; ++i) {
        centers[static_cast<size_t>(i)].x -= upperGroupShiftX;
    }
    constexpr float inputSide = 62.0f;
    constexpr float inputDiagonalStep = 5.0f;
    constexpr float outputFrameWidth = 85.0f;
    const float inputLeftEdgeX = centers[0].x - inputSide * 0.5f - inputDiagonalStep;
    const float outputCenterX = inputLeftEdgeX + outputFrameWidth * 0.5f;
    const float row2RightAnchorX = centers[3].x;
    const float row2EqualStep = (row2RightAnchorX - outputCenterX) / 4.0f;
    for (int i = 0; i <= 4; ++i) {
        centers[static_cast<size_t>(4 + i)].x =
            row2RightAnchorX - row2EqualStep * static_cast<float>(i);
    }

    for (size_t i = 0; i + 1 < centers.size(); ++i) {
        const bool highlight =
            blockIsActive(static_cast<int>(i)) || blockIsActive(static_cast<int>(i + 1));
        const ImU32 lineColor = highlight ? linkColorActive : linkColorDim;
        if (std::fabs(centers[i].y - centers[i + 1].y) < 0.5f) {
            drawList->AddLine(centers[i], centers[i + 1], lineColor, 2.0f);
        } else {
            const float midX = (centers[i].x + centers[i + 1].x) * 0.5f;
            drawList->AddLine(centers[i], ImVec2(midX, centers[i].y), lineColor, 2.0f);
            drawList->AddLine(ImVec2(midX, centers[i].y), ImVec2(midX, centers[i + 1].y), lineColor,
                              2.0f);
            drawList->AddLine(ImVec2(midX, centers[i + 1].y), centers[i + 1], lineColor, 2.0f);
        }
    }

    ImFont* font = ImGui::GetFont();
    const float fontSize = ImGui::GetFontSize();
    const float maxGlyphHalfHeight = 145.0f;
    const float titleYUpper = yUpper - maxGlyphHalfHeight - fontSize - 2.0f;
    const float titleYLower = yLower - maxGlyphHalfHeight - fontSize - 6.0f;

    for (size_t i = 0; i < blocks.size(); ++i) {
        const bool highlight = blockIsActive(static_cast<int>(i));
        const ImU32 color = highlight ? blocks[i].color : scaleColor(blocks[i].color, 0.35f);
        const double glyphAnimTimeSec = conv1AnimTimeSec >= 0.0 ? conv1AnimTimeSec : 0.0;
        if (i == 0) {
            drawInputLayerGlyph(drawList, centers[i], color);
        } else if (i == 1) {
            drawConv1LayerGlyph(drawList, centers[i], color, glyphAnimTimeSec);
        } else if (i == 2) {
            drawConv2LayerGlyph(drawList, centers[i], color, glyphAnimTimeSec);
        } else if (i == 3) {
            drawMaxPoolLayerGlyph(drawList, centers[i], color, glyphAnimTimeSec, false, false,
                                  28.0f, false, false);
        } else if (i == 4) {
            drawConv3LayerGlyph(drawList, centers[i], color, glyphAnimTimeSec);
        } else if (i == 5) {
            drawMaxPoolLayerGlyph(drawList, centers[i], color, glyphAnimTimeSec, true, false, 34.0f,
                                  false, false);
        } else if (i == 6) {
            drawFCLayerGlyph(drawList, centers[i], color, 64, 32, IM_COL32(255, 175, 120, 245),
                             IM_COL32(130, 220, 255, 245), IM_COL32(210, 200, 180, 120), true,
                             2.0f / 3.0f);
        } else if (i == 7) {
            drawFCLayerGlyph(drawList, centers[i], color, 32, 16, IM_COL32(185, 255, 150, 245),
                             IM_COL32(245, 180, 255, 245), IM_COL32(190, 210, 220, 120), false,
                             4.0f / 9.0f);
        } else {
            drawOutputLayerGlyph(drawList, centers[i], color);
        }

        const char* label = blocks[i].name;
        const ImVec2 textSize = ImGui::CalcTextSize(label);
        float textX = centers[i].x - textSize.x * 0.5f;
        if (i == 0) {
            textX += 46.0f;
        }
        const float textY = i <= 3 ? titleYUpper : titleYLower;
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
    std::vector<std::string> cifarFineLabelNames;
    std::vector<std::string> cifarFineLabelNamesZh;
    int currentOutputIndex = -1;
    float currentOutputValue = 0.0f;
    std::string currentOutputLabel;
    std::string currentOutputLabelZh;
    std::atomic<bool> stop{false};
    std::string checkpointFilePath;
    bool loadCheckpointBeforeTrain = false;
    int maxEpoch = kEpochs;
};

GLFWwindow* CNNGuiUtils::initWindow() {
    if (!glfwInit()) {
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    GLFWwindow* window = glfwCreateWindow(1400, 1080, "CNN Training", nullptr, nullptr);
    if (!window) {
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    return window;
}

void CNNGuiUtils::startTraining(TrainingStats& stats) {
    {
        std::lock_guard<std::mutex> lock(stats.mutex);
        try {
            stats.cifarFineLabelNames = NNDatasetManager::loadCifar100FineLabelNames();
            stats.cifarFineLabelNamesZh.clear();
            stats.cifarFineLabelNamesZh.reserve(stats.cifarFineLabelNames.size());
            for (const auto& name : stats.cifarFineLabelNames) {
                stats.cifarFineLabelNamesZh.push_back(translateCifar100FineLabelToChinese(name));
            }
        } catch (...) {
            stats.cifarFineLabelNames.clear();
            stats.cifarFineLabelNamesZh.clear();
        }
    }

    auto dataset = NNDatasetManager::prepareCifar100Dataset(CIFAR100_CNN_MAX_TRAIN_SAMPLES,
                                                            CIFAR100_CNN_MAX_TEST_SAMPLES);
    auto configs = NNDatasetManager::buildCifar100CnnConfigs();

    auto cnn = CNN(configs);
    cnn.configurePersistence(stats.checkpointFilePath, stats.loadCheckpointBeforeTrain);

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

        // Keep RGB channels for colorful CIFAR preview.
        stats.currentImage.assign(static_cast<size_t>(kInputSize * kInChannels), 0.0f);
        if (static_cast<int>(input.size()) >= kInChannels && input[0] && input[1] && input[2] &&
            input[0]->getRowSize() == kImageSide && input[0]->getColSize() == kImageSide &&
            input[1]->getRowSize() == kImageSide && input[1]->getColSize() == kImageSide &&
            input[2]->getRowSize() == kImageSide && input[2]->getColSize() == kImageSide) {
            for (int r = 0; r < kImageSide; ++r) {
                for (int c = 0; c < kImageSide; ++c) {
                    const size_t base = static_cast<size_t>((r * kImageSide + c) * kInChannels);
                    stats.currentImage[base] = toDisplayUnit(input[0]->get(r, c));
                    stats.currentImage[base + 1] = toDisplayUnit(input[1]->get(r, c));
                    stats.currentImage[base + 2] = toDisplayUnit(input[2]->get(r, c));
                }
            }
        } else {
            stats.currentImage.assign(static_cast<size_t>(kInputSize), 0.0f);
            const auto& m = input[0];
            if (m->getRowSize() == kImageSide && m->getColSize() == kImageSide) {
                for (int r = 0; r < kImageSide; ++r) {
                    for (int c = 0; c < kImageSide; ++c) {
                        stats.currentImage[static_cast<size_t>(r * kImageSide + c)] =
                            toDisplayUnit(m->get(r, c));
                    }
                }
            }
        }

        stats.currentOutputIndex = output.getIndexOfColMax(0);
        stats.currentOutputValue = output.get(stats.currentOutputIndex, 0);
        if (stats.currentOutputIndex >= 0 &&
            static_cast<size_t>(stats.currentOutputIndex) < stats.cifarFineLabelNames.size()) {
            stats.currentOutputLabel =
                stats.cifarFineLabelNames[static_cast<size_t>(stats.currentOutputIndex)];
            if (static_cast<size_t>(stats.currentOutputIndex) <
                stats.cifarFineLabelNamesZh.size()) {
                stats.currentOutputLabelZh =
                    stats.cifarFineLabelNamesZh[static_cast<size_t>(stats.currentOutputIndex)];
            } else {
                stats.currentOutputLabelZh.clear();
            }
        } else {
            stats.currentOutputLabel.clear();
            stats.currentOutputLabelZh.clear();
        }
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

    cnn.train(dataset, stats.maxEpoch, kBatchSize, kLearningRate, kMomentum,
              CIFAR100_CNN_WEIGHT_DECAY, callback, layerCallback, batchCallback, stopCallback,
              batchStatsCallback);

    stats.activeLayer.store(-1);
    stats.activePhase.store(static_cast<int>(CNN::LayerPhase::Idle));
    stats.done.store(true);
}

void CNNGuiUtils::drawInputImage(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                                 const std::vector<float>& image) {
    const int width = kImageSide;
    const int height = kImageSide;
    const size_t grayPixelCount = static_cast<size_t>(width * height);
    const size_t rgbPixelCount = grayPixelCount * static_cast<size_t>(kInChannels);
    const bool hasRgb = image.size() >= rgbPixelCount;
    if (!hasRgb && image.size() < grayPixelCount) {
        drawList->AddText(origin, IM_COL32(200, 200, 210, 255), "Waiting for batch...");
        return;
    }

    const float maxScaleX = size.x / width;
    const float maxScaleY = size.y / height;
    const float fitScale = std::min(maxScaleX, maxScaleY);
    const float scale = fitScale < 2.0f ? fitScale : 2.0f;
    const float imgW = scale * width;
    const float imgH = scale * height;
    const float offsetX = origin.x + (size.x - imgW) * 0.5f;
    const float offsetY = origin.y + (size.y - imgH) * 0.5f;

    drawList->PushClipRect(origin, ImVec2(origin.x + size.x, origin.y + size.y), true);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            ImU32 color = IM_COL32(0, 0, 0, 255);
            if (hasRgb) {
                const size_t base = static_cast<size_t>((y * width + x) * kInChannels);
                const int r = static_cast<int>(clampf(image[base], 0.0f, 1.0f) * 255.0f);
                const int g = static_cast<int>(clampf(image[base + 1], 0.0f, 1.0f) * 255.0f);
                const int b = static_cast<int>(clampf(image[base + 2], 0.0f, 1.0f) * 255.0f);
                color = IM_COL32(r, g, b, 255);
            } else {
                const float value = image[static_cast<size_t>(y * width + x)];
                const int intensity = static_cast<int>(clampf(value, 0.0f, 1.0f) * 255.0f);
                color = IM_COL32(intensity, intensity, intensity, 255);
            }
            const float x0 = offsetX + x * scale;
            const float y0 = offsetY + y * scale;
            drawList->AddRectFilled(ImVec2(x0, y0), ImVec2(x0 + scale, y0 + scale), color);
        }
    }

    drawList->AddRect(ImVec2(offsetX, offsetY), ImVec2(offsetX + imgW, offsetY + imgH),
                      IM_COL32(80, 90, 110, 255));
    drawList->PopClipRect();
}

int CNNGuiUtils::RunTrainingGui(const std::string& checkpointFilePath, bool loadBeforeTrain,
                                int maxEpoch) {
    GLFWwindow* window = initWindow();
    if (!window) {
        std::cerr << "Failed to initialize GLFW window" << std::endl;
        return 1;
    }

    const char* glsl_version = "#version 150";
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    configureGuiFonts();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    TrainingStats stats;
    stats.checkpointFilePath = checkpointFilePath;
    stats.loadCheckpointBeforeTrain = loadBeforeTrain || maxEpoch > 0;
    stats.maxEpoch = maxEpoch > 0 ? maxEpoch : kEpochs;
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
            ImGui::Text("Epoch: %d/%d", epoch, stats.maxEpoch);
        } else {
            ImGui::Text("Epoch: .../%d", stats.maxEpoch);
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
        std::string outputLabel;
        std::string outputLabelZh;
        {
            std::lock_guard<std::mutex> lock(stats.mutex);
            outputIndex = stats.currentOutputIndex;
            outputValue = stats.currentOutputValue;
            outputLabel = stats.currentOutputLabel;
            outputLabelZh = stats.currentOutputLabelZh;
        }
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_Text, IM_COL32(255, 210, 120, 255));
        ImGui::SetWindowFontScale(1.6f);
        if (outputIndex >= 0) {
            ImGui::Text("Output Label: %s {%d}",
                        outputLabel.empty() ? "Unknown" : outputLabel.c_str(), outputIndex);
            ImGui::Text("中文标签: %s", outputLabelZh.empty() ? "未知" : outputLabelZh.c_str());
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
        const bool done = stats.done.load();
        if (done) {
            activeLayer = -1;
            activePhase = static_cast<int>(CNN::LayerPhase::Idle);
        }
        const double conv1AnimTimeSec = done ? -1.0 : ImGui::GetTime();
        drawCnnTopology(drawList, canvasPos, canvasSize, activeLayer, activePhase,
                        conv1AnimTimeSec);
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
