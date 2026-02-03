#include "NNDataSet.h"

#include <stdexcept>
#include <utility>

NNDataSet::NNDataSet(std::string datasetLabel, NNMatrixPtrVector trainInput,
                     NNMatrixPtrVector trainLabel, NNMatrixPtrVector testInput,
                     NNMatrixPtrVector testLabel)
    : datasetLabel_(std::move(datasetLabel)), trainInput_(std::move(trainInput)),
      trainLabel_(std::move(trainLabel)), testInput_(std::move(testInput)),
      testLabel_(std::move(testLabel)) {
}

NNDataSet::NNDataSet(NNDataSet&& other) noexcept {
    datasetLabel_ = std::move(other.datasetLabel_);
    trainInput_ = std::move(other.trainInput_);
    trainLabel_ = std::move(other.trainLabel_);
    testInput_ = std::move(other.testInput_);
    testLabel_ = std::move(other.testLabel_);
}

NNMatrixPtr NNDataSet::getTrainDataAt(int index) const {
    if (index < 0 || index >= trainInput_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return trainInput_[index];
}

NNMatrixPtr NNDataSet::getTrainLabeltaAt(int index) const {
    if (index < 0 || index >= trainLabel_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return trainLabel_[index];
}

NNMatrixPtr NNDataSet::getTestDataAt(int index) const {
    if (index < 0 || index >= testInput_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return testInput_[index];
}

NNMatrixPtr NNDataSet::getTestLabeltaAt(int index) const {
    if (index < 0 || index >= testLabel_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return testLabel_[index];
}