#pragma once

#include "NNMatrix.h"
#include "nnlog/nnlog.h"

#include <cassert>
#include <cstdint>
#include <vector>

#define LOG NNLOG_INFO((TAG).c_str())

class NNUtils {
  private:
    enum : std::uint16_t {
        MNIST_IMAGE_MAGIC = 2051,
        MNIST_LABEL_MAGIC = 2049,
    };

    static const std::string TAG;
    static uint32_t swap_endian(uint32_t val);

  public:
    struct ShuffleSampleInfo {
        int inChannelSize = 1;
        int sampleCount = 0;
    };

    // Integer ceiling division for non-negative values.
    // Returns ceil(numer / denom). denom must be > 0.
    static int ceilDiv(int numer, int denom) {
        assert(denom > 0);
        assert(numer >= 0);
        return (numer + denom - 1) / denom;
    }

    static std::vector<NNMatrixPtr> read_mnist_data(const std::string& filePath);
    static std::vector<NNMatrixPtr> read_mnist_labels(const std::string& filePath);
    static void shuffle(std::vector<NNMatrixPtr>& input, std::vector<NNMatrixPtr>& label);

    // Shuffles by *sample* while keeping per-sample multi-channel inputs contiguous.
    // Example: RGB stored as 3 matrices per sample will be shuffled as a unit.
    // Returns inferred (channelsPerSample, sampleCount) used for batching.
    static ShuffleSampleInfo shuffleSamples(std::vector<NNMatrixPtr>& input,
                                            std::vector<NNMatrixPtr>& label);
    static std::vector<NNMatrixPtr> getBatch(std::vector<NNMatrixPtr>& input, int batchNo,
                                             int batchSize);

    // Flattens each matrix (column-major as stored) into a single column vector and concatenates
    // them in order.
    // Returns nullptr if any input is null/empty/invalid.
    static NNMatrixPtr flattenAndConcat(const NNMatrixPtrV& mats);

    static float random(float a, float b);
    static float xavierInit(int inputSize, int outputSize);
    static void normalizeMnistData(std::vector<NNMatrixPtr>& data);
    static void normalizeMnistLabel(std::vector<NNMatrixPtr>& labels);
};