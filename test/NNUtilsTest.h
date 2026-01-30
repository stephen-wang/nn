#include "../include/NNUtils.h"

#include "gtest/gtest.h"
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

static NNMatrixPtr makeScalarMatrix(float value) {
    auto m = std::make_shared<NNMatrix>(1, 1, 0.0f);
    m->set(0, 0, value);
    return m;
}

TEST(NNUtilsTest, XavierInitRange) {
    int inputSize = 4;
    int outputSize = 6;
    float limit = std::sqrt(6.0f / float(inputSize + outputSize));
    for (int i = 0; i < 100; i++) {
        float val = NNUtils::xavierInit(inputSize, outputSize);
        EXPECT_GE(val, -limit);
        EXPECT_LE(val, limit);
    }
}

TEST(NNUtilsTest, NormalizeMnistData) {
    std::vector<NNMatrixPtr> data;
    auto m = std::make_shared<NNMatrix>(2, 1, 0.0f);
    m->set(0, 0, 0.0f);
    m->set(1, 0, 255.0f);
    data.push_back(m);

    NNUtils::normalizeMnistData(data);

    ASSERT_FLOAT_EQ(0.0f, data[0]->get(0, 0));
    ASSERT_FLOAT_EQ(1.0f, data[0]->get(1, 0));
}

TEST(NNUtilsTest, NormalizeMnistLabel) {
    std::vector<NNMatrixPtr> labels;
    auto m = std::make_shared<NNMatrix>(3, 1, 0.0f);
    m->set(0, 0, 0.0f);
    m->set(1, 0, 0.5f);
    m->set(2, 0, 1e-6f);
    labels.push_back(m);

    NNUtils::normalizeMnistLabel(labels);

    ASSERT_FLOAT_EQ(0.0f, labels[0]->get(0, 0));
    ASSERT_FLOAT_EQ(1.0f, labels[0]->get(1, 0));
    ASSERT_FLOAT_EQ(0.0f, labels[0]->get(2, 0));
}

TEST(NNUtilsTest, GetBatch) {
    std::vector<NNMatrixPtr> data;
    for (int i = 0; i < 5; i++) {
        data.push_back(makeScalarMatrix(static_cast<float>(i)));
    }

    auto batch = NNUtils::getBatch(data, 1, 2);
    ASSERT_EQ(2u, batch.size());
    ASSERT_FLOAT_EQ(2.0f, batch[0]->get(0, 0));
    ASSERT_FLOAT_EQ(3.0f, batch[1]->get(0, 0));
}

TEST(NNUtilsTest, GetBatchOutOfRangeReturnsEmpty) {
    std::vector<NNMatrixPtr> data;
    for (int i = 0; i < 3; i++) {
        data.push_back(makeScalarMatrix(static_cast<float>(i)));
    }

    auto batch = NNUtils::getBatch(data, 5, 2);
    ASSERT_TRUE(batch.empty());
}

TEST(NNUtilsTest, GetBatchPartialLastBatch) {
    std::vector<NNMatrixPtr> data;
    for (int i = 0; i < 5; i++) {
        data.push_back(makeScalarMatrix(static_cast<float>(i)));
    }

    auto batch = NNUtils::getBatch(data, 2, 2);
    ASSERT_EQ(1u, batch.size());
    ASSERT_FLOAT_EQ(4.0f, batch[0]->get(0, 0));
}

TEST(NNUtilsTest, ShuffleKeepsPairs) {
    std::vector<NNMatrixPtr> inputs;
    std::vector<NNMatrixPtr> labels;
    for (int i = 0; i < 10; i++) {
        inputs.push_back(makeScalarMatrix(static_cast<float>(i)));
        labels.push_back(makeScalarMatrix(static_cast<float>(i)));
    }

    NNUtils::shuffle(inputs, labels);

    ASSERT_EQ(inputs.size(), labels.size());
    for (int i = 0; i < inputs.size(); i++) {
        ASSERT_FLOAT_EQ(inputs[i]->get(0, 0), labels[i]->get(0, 0));
    }
}

TEST(NNUtilsTest, ShuffleEmptyVectors) {
    std::vector<NNMatrixPtr> inputs;
    std::vector<NNMatrixPtr> labels;

    NNUtils::shuffle(inputs, labels);

    ASSERT_TRUE(inputs.empty());
    ASSERT_TRUE(labels.empty());
}

TEST(NNUtilsTest, ShuffleSingleElement) {
    std::vector<NNMatrixPtr> inputs;
    std::vector<NNMatrixPtr> labels;

    inputs.push_back(makeScalarMatrix(1.0f));
    labels.push_back(makeScalarMatrix(1.0f));

    NNUtils::shuffle(inputs, labels);

    ASSERT_EQ(1u, inputs.size());
    ASSERT_EQ(1u, labels.size());
    ASSERT_FLOAT_EQ(1.0f, inputs[0]->get(0, 0));
    ASSERT_FLOAT_EQ(1.0f, labels[0]->get(0, 0));
}

static void write_be32(std::ofstream& ofs, std::uint32_t value) {
    unsigned char bytes[4];
    bytes[0] = static_cast<unsigned char>((value >> 24) & 0xFF);
    bytes[1] = static_cast<unsigned char>((value >> 16) & 0xFF);
    bytes[2] = static_cast<unsigned char>((value >> 8) & 0xFF);
    bytes[3] = static_cast<unsigned char>(value & 0xFF);
    ofs.write(reinterpret_cast<char*>(bytes), 4);
}

TEST(NNUtilsTest, ReadMnistData) {
    auto tempPath = std::filesystem::temp_directory_path() / "mnist_test_images.idx3-ubyte";

    {
        std::ofstream ofs(tempPath, std::ios::binary);
        ASSERT_TRUE(ofs.is_open());

        write_be32(ofs, 2051); // magic
        write_be32(ofs, 1);    // numImages
        write_be32(ofs, 2);    // rows
        write_be32(ofs, 2);    // cols

        unsigned char pixels[4] = {0, 128, 255, 64};
        ofs.write(reinterpret_cast<char*>(pixels), 4);
    }

    auto images = NNUtils::read_mnist_data(tempPath.string());
    ASSERT_EQ(1u, images.size());
    ASSERT_EQ(4, images[0]->getRowSize());
    ASSERT_EQ(1, images[0]->getColSize());

    ASSERT_FLOAT_EQ(0.0f, images[0]->get(0, 0));
    ASSERT_FLOAT_EQ(128.0f, images[0]->get(1, 0));
    ASSERT_FLOAT_EQ(255.0f, images[0]->get(2, 0));
    ASSERT_FLOAT_EQ(64.0f, images[0]->get(3, 0));

    std::error_code ec;
    std::filesystem::remove(tempPath, ec);
}

TEST(NNUtilsTest, ReadMnistLabels) {
    auto tempPath = std::filesystem::temp_directory_path() / "mnist_test_labels.idx1-ubyte";

    {
        std::ofstream ofs(tempPath, std::ios::binary);
        ASSERT_TRUE(ofs.is_open());

        write_be32(ofs, 2049); // magic
        write_be32(ofs, 2);    // numLabels

        unsigned char labels[2] = {3, 7};
        ofs.write(reinterpret_cast<char*>(labels), 2);
    }

    auto labels = NNUtils::read_mnist_labels(tempPath.string());
    ASSERT_EQ(2u, labels.size());

    ASSERT_EQ(10, labels[0]->getRowSize());
    ASSERT_EQ(1, labels[0]->getColSize());
    ASSERT_FLOAT_EQ(1.0f, labels[0]->get(3, 0));
    ASSERT_FLOAT_EQ(0.0f, labels[0]->get(7, 0));

    ASSERT_FLOAT_EQ(1.0f, labels[1]->get(7, 0));
    ASSERT_FLOAT_EQ(0.0f, labels[1]->get(3, 0));

    std::error_code ec;
    std::filesystem::remove(tempPath, ec);
}

TEST(NNUtilsTest, ReadMnistDataInvalidMagic) {
    auto tempPath = std::filesystem::temp_directory_path() / "mnist_test_bad_images.idx3-ubyte";

    {
        std::ofstream ofs(tempPath, std::ios::binary);
        ASSERT_TRUE(ofs.is_open());

        write_be32(ofs, 9999); // wrong magic
        write_be32(ofs, 1);    // numImages
        write_be32(ofs, 2);    // rows
        write_be32(ofs, 2);    // cols

        unsigned char pixels[4] = {0, 1, 2, 3};
        ofs.write(reinterpret_cast<char*>(pixels), 4);
    }

    EXPECT_THROW(NNUtils::read_mnist_data(tempPath.string()), std::runtime_error);

    std::error_code ec;
    std::filesystem::remove(tempPath, ec);
}

TEST(NNUtilsTest, ReadMnistLabelsInvalidMagic) {
    auto tempPath = std::filesystem::temp_directory_path() / "mnist_test_bad_labels.idx1-ubyte";

    {
        std::ofstream ofs(tempPath, std::ios::binary);
        ASSERT_TRUE(ofs.is_open());

        write_be32(ofs, 9999); // wrong magic
        write_be32(ofs, 1);    // numLabels

        unsigned char label = 1;
        ofs.write(reinterpret_cast<char*>(&label), 1);
    }

    EXPECT_THROW(NNUtils::read_mnist_labels(tempPath.string()), std::runtime_error);

    std::error_code ec;
    std::filesystem::remove(tempPath, ec);
}

TEST(NNUtilsTest, ReadMnistDataMissingFile) {
    auto tempPath = std::filesystem::temp_directory_path() / "mnist_test_missing_images.idx3-ubyte";
    std::filesystem::remove(tempPath);
    EXPECT_THROW(NNUtils::read_mnist_data(tempPath.string()), std::runtime_error);
}

TEST(NNUtilsTest, ReadMnistLabelsMissingFile) {
    auto tempPath = std::filesystem::temp_directory_path() / "mnist_test_missing_labels.idx1-ubyte";
    std::filesystem::remove(tempPath);
    EXPECT_THROW(NNUtils::read_mnist_labels(tempPath.string()), std::runtime_error);
}
