#pragma once

#include "NNDataset.h"

#include <string>
#include <vector>

class NNDatasetManager {
  private:
    static const std::string TAG;

  public:
    static NNDataset loadMnist();
    static NNDataset loadCifar100();
    static std::vector<std::string> loadCifar100CoarseLabelNames();
    static std::vector<std::string> loadCifar100FineLabelNames();

  private:
    static void readCifar100BinaryFile(const std::string& filePath, NNMatrixPtrV& data,
                                       NNMatrixPtrV& labels);
    static std::vector<std::string> readCifar100LabelNamesFromFile(const std::string& filePath);
};
