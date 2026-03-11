#pragma once

#include "../include/DNN.h"

#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
NNMatrixPtr makeVec4(float a, float b, float c, float d) {
    auto x = std::make_shared<NNMatrix>(4, 1, 0.0f);
    x->set(0, 0, a);
    x->set(1, 0, b);
    x->set(2, 0, c);
    x->set(3, 0, d);
    return x;
}

NNMatrixPtr makeOneHot2Label(int cls) {
    auto y = std::make_shared<NNMatrix>(2, 1, 0.0f);
    y->set(cls, 0, 1.0f);
    return y;
}

std::vector<char> readAllFileBytes(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    EXPECT_TRUE(ifs.is_open());
    if (!ifs.is_open()) {
        return {};
    }
    return std::vector<char>((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
}
} // namespace

TEST(DNNSerializationTest, SaveLoadRoundTripProducesIdenticalStateFile) {
    std::vector<int> cfg{4, 3, 2};
    DNN modelA(cfg);

    NNMatrixPtrV trainX{makeVec4(0.1f, 0.2f, 0.3f, 0.4f), makeVec4(0.5f, 0.4f, 0.3f, 0.2f)};
    NNMatrixPtrV trainY{makeOneHot2Label(0), makeOneHot2Label(1)};
    NNDataset ds("toy-dnn", trainX, trainY, trainX, trainY);

    modelA.train(ds, 1, 2, 0.01f, 0.9f, nullptr, nullptr, nullptr, nullptr, nullptr);

    const auto dir = std::filesystem::temp_directory_path();
    const auto fileA = dir / "dnn_save_load_A.bin";
    const auto fileB = dir / "dnn_save_load_B.bin";

    ASSERT_TRUE(modelA.save(fileA.string()));

    DNN modelB(cfg);
    ASSERT_TRUE(modelB.load(fileA.string()));
    ASSERT_TRUE(modelB.save(fileB.string()));

    auto bytesA = readAllFileBytes(fileA);
    auto bytesB = readAllFileBytes(fileB);
    ASSERT_EQ(bytesA, bytesB);

    std::error_code ec;
    std::filesystem::remove(fileA, ec);
    std::filesystem::remove(fileB, ec);
}

TEST(DNNSerializationTest, LoadFailsForMismatchedArchitecture) {
    std::vector<int> cfgA{4, 3, 2};
    DNN modelA(cfgA);

    const auto file = std::filesystem::temp_directory_path() / "dnn_mismatch_arch.bin";
    ASSERT_TRUE(modelA.save(file.string()));

    std::vector<int> cfgMismatch{4, 4, 2};
    DNN modelMismatch(cfgMismatch);
    ASSERT_FALSE(modelMismatch.load(file.string()));

    std::error_code ec;
    std::filesystem::remove(file, ec);
}
