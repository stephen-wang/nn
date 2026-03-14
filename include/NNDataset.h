#pragma once

#include "NNMatrix.h"

#include <string>
#include <vector>

class NNDataset {
  public:
    NNDataset(std::string datasetLabel, NNMatrixPtrV trainInput, NNMatrixPtrV trainLabel,
              NNMatrixPtrV testInput, NNMatrixPtrV testLabel,
              std::vector<std::vector<unsigned char>> trainPreviewBytes = {},
              std::vector<std::vector<unsigned char>> testPreviewBytes = {});

    NNDataset(const NNDataset& other) = delete;
    NNDataset(NNDataset&& other) noexcept;

    virtual ~NNDataset() {
        trainInput_.clear();
        trainLabel_.clear();
        testInput_.clear();
        testLabel_.clear();
        trainPreviewBytes_.clear();
        testPreviewBytes_.clear();
    }
    int getTrainInputSize() const { return trainInput_.size(); }
    int getTestInputSize() const { return testInput_.size(); }
    NNMatrixPtr getTrainDataAt(int index) const;
    NNMatrixPtr getTrainLabeltaAt(int index) const;
    NNMatrixPtr getTestDataAt(int index) const;
    NNMatrixPtr getTestLabeltaAt(int index) const;

  public:
    std::string datasetLabel_;
    NNMatrixPtrV trainInput_;
    NNMatrixPtrV trainLabel_;
    NNMatrixPtrV testInput_;
    NNMatrixPtrV testLabel_;
    std::vector<std::vector<unsigned char>> trainPreviewBytes_;
    std::vector<std::vector<unsigned char>> testPreviewBytes_;
};
