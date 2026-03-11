#pragma once

#include "../include/CNN.h"
#include "../include/CNNConfigBuilder.h"

#include "gtest/gtest.h"
#include <filesystem>
#include <fstream>
#include <vector>

namespace {
NNMatrixPtr makeChannel4x4(float base) {
    auto m = std::make_shared<NNMatrix>(4, 4, 0.0f);
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            m->set(r, c, base + static_cast<float>(r * 4 + c) * 0.01f);
        }
    }
    return m;
}

NNMatrixPtr makeOneHot2(int cls) {
    auto y = std::make_shared<NNMatrix>(2, 1, 0.0f);
    y->set(cls, 0, 1.0f);
    return y;
}

std::vector<CNNConfigPtr> makeToyConfig() {
    return CNNConfigBuilder()
        .addConvolution(1, 2, 3, 1, 1)
        .addMaxPooling(2, 2)
        .addFullyConnected(8, 4)
        .addFullyConnected(4, 2)
        .build();
}

std::vector<char> readAllBytes(const std::filesystem::path& path) {
    std::ifstream ifs(path, std::ios::binary);
    EXPECT_TRUE(ifs.is_open());
    if (!ifs.is_open()) {
        return {};
    }
    return std::vector<char>((std::istreambuf_iterator<char>(ifs)),
                             std::istreambuf_iterator<char>());
}
} // namespace

TEST(CNNSerializationTest, SaveLoadRoundTripProducesIdenticalStateFile) {
    auto config = makeToyConfig();
    CNN modelA(config);

    NNMatrixPtrV trainX{makeChannel4x4(0.10f), makeChannel4x4(0.20f)};
    NNMatrixPtrV trainY{makeOneHot2(0), makeOneHot2(1)};
    NNDataset ds("toy", trainX, trainY, trainX, trainY);

    modelA.train(ds, 1, 1, 0.01f, 0.9f, 0.0f);

    const auto dir = std::filesystem::temp_directory_path();
    const auto fileA = dir / "cnn_save_load_A.bin";
    const auto fileB = dir / "cnn_save_load_B.bin";

    ASSERT_TRUE(modelA.save(fileA.string()));

    CNN modelB(config);
    ASSERT_TRUE(modelB.load(fileA.string()));
    ASSERT_TRUE(modelB.save(fileB.string()));

    auto bytesA = readAllBytes(fileA);
    auto bytesB = readAllBytes(fileB);
    ASSERT_EQ(bytesA, bytesB);

    std::error_code ec;
    std::filesystem::remove(fileA, ec);
    std::filesystem::remove(fileB, ec);
}

TEST(CNNSerializationTest, LoadFailsForMismatchedArchitecture) {
    auto configA = makeToyConfig();
    CNN modelA(configA);

    const auto file = std::filesystem::temp_directory_path() / "cnn_mismatch_arch.bin";
    ASSERT_TRUE(modelA.save(file.string()));

    auto mismatchedConfig =
        CNNConfigBuilder().addConvolution(1, 1, 3, 1, 1).addFullyConnected(16, 2).build();
    CNN modelMismatch(mismatchedConfig);
    ASSERT_FALSE(modelMismatch.load(file.string()));

    std::error_code ec;
    std::filesystem::remove(file, ec);
}
