#pragma once

#include <string>
#include <vector>

struct GLFWwindow;
struct ImDrawList;
struct ImVec2;

class DNNGuiUtils {
  public:
    static int RunTrainingGui(const std::string& checkpointFilePath = "dnn_checkpoint.bin",
                              bool loadBeforeTrain = false);

  private:
    struct TrainingStats;

    static GLFWwindow* initWindow();
    static void startTraining(TrainingStats& stats);
    static void drawInputImage(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                               const std::vector<float>& image);
};
