#pragma once

#include "NNMatrix.h"

#include <string>
#include <vector>

class NNDataSet {
  public:
    NNDataSet(const std::string& datasetLabel, const NNMatrixPtrVector& trainInput,
              const NNMatrixPtrVector& trainLabel, const NNMatrixPtrVector& testInput,
              const NNMatrixPtrVector& testLabel);

    NNDataSet(const NNDataSet& other) = delete;
    NNDataSet(NNDataSet&& other) noexcept;

    virtual ~NNDataSet() {
        trainInput_.clear();
        trainLabel_.clear();
        testInput_.clear();
        testLabel_.clear();
    }
    int getTrainInputSize() const { return trainInput_.size(); }
    int getTestInputSize() const { return testInput_.size(); }
    NNMatrixPtr getTrainDataAt(int index) const;
    NNMatrixPtr getTrainLabeltaAt(int index) const;
    NNMatrixPtr getTestDataAt(int index) const;
    NNMatrixPtr getTestLabeltaAt(int index) const;

  public:
    std::string datasetLabel_;
    NNMatrixPtrVector trainInput_;
    NNMatrixPtrVector trainLabel_;
    NNMatrixPtrVector testInput_;
    NNMatrixPtrVector testLabel_;
};
