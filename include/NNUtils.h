#pragma once

#include <vector>
#include "NNMatrix.h"

#include "nnlog/nnlog.h"

#define LOG NNLOG_INFO((TAG).c_str())

class NNUtils
{
private:
    enum
    {
        MNIST_IMAGE_MAGIC = 2051,
        MNIST_LABEL_MAGIC = 2049,
    };

    static const std::string TAG;
    static uint32_t swap_endian(uint32_t val);

public:
    static std::vector<NNMatrixPtr> read_mnist_data(const std::string &filePath);
    static std::vector<NNMatrixPtr> read_mnist_labels(const std::string &filePath);
    static void shuffle(std::vector<NNMatrixPtr> &input, std::vector<NNMatrixPtr> &label);
    static std::vector<NNMatrixPtr> getBatch(std::vector<NNMatrixPtr> &input, int batchNo, int batchSize);
    static float random(float a, float b);
    static float xavierInit(int inputSize, int outputSize);
    static void normalizeMnistData(std::vector<NNMatrixPtr> &data);
    static void normalizeMnistLabel(std::vector<NNMatrixPtr> &labels);
};