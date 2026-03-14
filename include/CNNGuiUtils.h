#pragma once

#ifndef GL_SILENCE_DEPRECATION
#define GL_SILENCE_DEPRECATION
#endif

#include <OpenGL/gl.h>
#include <string>
#include <vector>

struct GLFWwindow;
struct ImDrawList;
struct ImVec2;

class CNNGuiUtils {
  public:
    static int RunTrainingGui(const std::string& checkpointFilePath = "cnn_checkpoint.bin",
                              bool loadBeforeTrain = false, int maxEpoch = -1);

  private:
    struct TrainingStats;

    static GLFWwindow* initWindow();
    static void startTraining(TrainingStats& stats);
};
