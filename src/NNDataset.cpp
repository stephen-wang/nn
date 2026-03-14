#include "NNDataset.h"

#include <stdexcept>
#include <utility>

NNDataset::NNDataset(std::string datasetLabel, NNMatrixPtrV trainInput, NNMatrixPtrV trainLabel,
                     NNMatrixPtrV testInput, NNMatrixPtrV testLabel,
                     std::vector<std::vector<unsigned char>> trainPreviewBytes,
                     std::vector<std::vector<unsigned char>> testPreviewBytes)
    : datasetLabel_(std::move(datasetLabel)), trainInput_(std::move(trainInput)),
      trainLabel_(std::move(trainLabel)), testInput_(std::move(testInput)),
      testLabel_(std::move(testLabel)), trainPreviewBytes_(std::move(trainPreviewBytes)),
      testPreviewBytes_(std::move(testPreviewBytes)) {
}

NNDataset::NNDataset(NNDataset&& other) noexcept {
    datasetLabel_ = std::move(other.datasetLabel_);
    trainInput_ = std::move(other.trainInput_);
    trainLabel_ = std::move(other.trainLabel_);
    testInput_ = std::move(other.testInput_);
    testLabel_ = std::move(other.testLabel_);
    trainPreviewBytes_ = std::move(other.trainPreviewBytes_);
    testPreviewBytes_ = std::move(other.testPreviewBytes_);
}

NNMatrixPtr NNDataset::getTrainDataAt(int index) const {
    if (index < 0 || index >= trainInput_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return trainInput_[index];
}

NNMatrixPtr NNDataset::getTrainLabeltaAt(int index) const {
    if (index < 0 || index >= trainLabel_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return trainLabel_[index];
}

NNMatrixPtr NNDataset::getTestDataAt(int index) const {
    if (index < 0 || index >= testInput_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return testInput_[index];
}

NNMatrixPtr NNDataset::getTestLabeltaAt(int index) const {
    if (index < 0 || index >= testLabel_.size()) {
        throw std::out_of_range("Index out of range");
    }
    return testLabel_[index];
}