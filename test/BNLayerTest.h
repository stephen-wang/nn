#pragma once

#include "../include/BNLayer.h"

#include "gtest/gtest.h"
#include <cmath>
#include <memory>
#include <vector>

namespace {
NNMatrixPtr makeScalarChannel(float value) {
    auto m = std::make_shared<NNMatrix>(1, 1, 0.0f);
    m->set(0, 0, value);
    return m;
}

NNMatrixPtr makeRowChannel(float a, float b) {
    auto m = std::make_shared<NNMatrix>(1, 2, 0.0f);
    m->set(0, 0, a);
    m->set(0, 1, b);
    return m;
}
} // namespace

TEST(BNLayerTest, ForwardBatchMatchesPopulationNormalization) {
    BNLayer layer(1, 0.0f, 0.1f);

    std::vector<NNMatrixPtrV> batch{
        NNMatrixPtrV{makeRowChannel(1.0f, 3.0f)},
        NNMatrixPtrV{makeRowChannel(5.0f, 7.0f)},
    };

    const auto out = layer.forwardBatch(batch, true);
    ASSERT_EQ(out.size(), batch.size());
    ASSERT_EQ(out[0].size(), 1U);
    ASSERT_EQ(out[1].size(), 1U);

    const float invStd = 1.0f / std::sqrt(5.0f);
    EXPECT_NEAR(out[0][0]->get(0, 0), (1.0f - 4.0f) * invStd, 1e-5f);
    EXPECT_NEAR(out[0][0]->get(0, 1), (3.0f - 4.0f) * invStd, 1e-5f);
    EXPECT_NEAR(out[1][0]->get(0, 0), (5.0f - 4.0f) * invStd, 1e-5f);
    EXPECT_NEAR(out[1][0]->get(0, 1), (7.0f - 4.0f) * invStd, 1e-5f);
}

TEST(BNLayerTest, ForwardBatchIsStableForLargeOffsetInputs) {
    BNLayer layer(1, 0.0f, 0.1f);

    std::vector<NNMatrixPtrV> batch{
        NNMatrixPtrV{makeScalarChannel(10000000.0f)},
        NNMatrixPtrV{makeScalarChannel(10000001.0f)},
        NNMatrixPtrV{makeScalarChannel(9999999.0f)},
        NNMatrixPtrV{makeScalarChannel(10000002.0f)},
    };

    const auto out = layer.forwardBatch(batch, true);
    ASSERT_EQ(out.size(), batch.size());

    const float invStd = 1.0f / std::sqrt(1.25f);
    const std::vector<float> expected{
        -0.5f * invStd,
        0.5f * invStd,
        -1.5f * invStd,
        1.5f * invStd,
    };

    for (size_t i = 0; i < expected.size(); ++i) {
        ASSERT_EQ(out[i].size(), 1U);
        ASSERT_NE(out[i][0], nullptr);
        EXPECT_NEAR(out[i][0]->get(0, 0), expected[i], 1e-4f);
    }
}
