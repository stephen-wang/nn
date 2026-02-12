#pragma once

#include <vector>

struct GLFWwindow;
struct ImDrawList;
struct ImVec2;

class DNNGuiUtils {
  public:
    static int RunTrainingGui();

  private:
    struct TrainingStats;

    static GLFWwindow* initWindow();
    static void startTraining(TrainingStats& stats);
    static void drawInputImage(ImDrawList* drawList, const ImVec2& origin, const ImVec2& size,
                               const std::vector<float>& image);
};
