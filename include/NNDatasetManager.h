#pragma once

#include "CNNConfig.h"
#include "NNDataset.h"

#include <string>
#include <vector>

class NNDatasetManager {
  private:
    static const std::string TAG;

  public:
    static NNDataset loadMnist();
    static NNDataset loadCifar100();
    static NNDataset prepareCifar100Dataset(int maxTrainSamples, int maxTestSamples);
    static std::vector<CNNConfigPtr> buildCifar100CnnConfigs();
    static std::vector<std::string> loadCifar100CoarseLabelNames();
    static std::vector<std::string> loadCifar100FineLabelNames();

  private:
    static void readCifar100BinaryFile(const std::string& filePath, NNMatrixPtrV& data,
                                       NNMatrixPtrV& labels,
                                       std::vector<std::vector<unsigned char>>& previewBytes);
    static std::vector<std::string> readCifar100LabelNamesFromFile(const std::string& filePath);
};
