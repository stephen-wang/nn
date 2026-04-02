#pragma once

#include "../include/NNMatrix.h"

#include "gtest/gtest.h"
#include <functional>

bool isEqual(const std::vector<float>& A, const std::vector<float>& B) {
    return std::equal(A.begin(), A.end(), B.begin(),
                      [](float x, float y) { return std::fabs(x - y) < 1e-6f; });
}

TEST(NNMatrixTest, ConstructorTest) {
    NNMatrix matrix(3, 5, 1.0f);
    ASSERT_FLOAT_EQ(1.0f, matrix.get(2, 2));
    ASSERT_EQ(3, matrix.getRowSize());
    ASSERT_EQ(5, matrix.getColSize());

    std::vector<float> expectedCol{1.0f, 1.0f, 1.0f};
    std::vector<float> expectedRow{1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    bool rowCheckResult = isEqual(expectedRow, matrix.getRow(1));
    ASSERT_TRUE(rowCheckResult);

    bool colCheckResult = isEqual(expectedCol, matrix.getCol(4));
    ASSERT_TRUE(colCheckResult);
}

TEST(NNMatrixTest, OperatorTest) {
    NNMatrix matrix(3, 3, 2.0f);
    ASSERT_FLOAT_EQ(2.0f, matrix.get(1, 2));

    // failed += due to mismatched metric
    matrix += NNMatrix(3, 2, 1.0f);
    ASSERT_FLOAT_EQ(2.0f, matrix.get(1, 2));

    // successful +=
    matrix += NNMatrix(3, 3, 3.0f);
    ASSERT_FLOAT_EQ(5.0f, matrix.get(1, 2));

    // successful -=
    matrix -= NNMatrix(3, 3, 10.0f);
    ASSERT_FLOAT_EQ(-5.0f, matrix.get(1, 0));
    ASSERT_FLOAT_EQ(-5.0f, matrix.get(0, 2));

    // successful *=
    matrix *= 10.0f;
    ASSERT_FLOAT_EQ(-50.0f, matrix.get(0, 0));

    // successful /=
    matrix /= -5.0f;
    ASSERT_FLOAT_EQ(10.0f, matrix.get(2, 2));

    // dotProduct
    //  ｜ 10 10 10 ｜    ｜1 2｜
    //  ｜ 10 10 10 ｜  X ｜2 4｜
    //  ｜ 10 10 10 ｜    ｜3 6｜
    NNMatrix matrix2(3, 2);
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 2; j++) {
            matrix2.set(i, j, (i + 1) * (j + 1));
        }
    }

    auto result = matrix.dotProduct(matrix2);
    ASSERT_EQ(3, result.getRowSize());
    ASSERT_EQ(2, result.getColSize());
    ASSERT_FLOAT_EQ(60.0f, result.get(0, 0));
    ASSERT_FLOAT_EQ(120.0f, result.get(0, 1));

    // elementProduct
    NNMatrix matrix3(2, 2, 2.0f);
    result = matrix3.elementProduct(NNMatrix(2, 2, 3.0f));
    ASSERT_FLOAT_EQ(result.get(0, 0), 6.0f);
    ASSERT_FLOAT_EQ(result.get(1, 1), 6.0f);
}

TEST(NNMatrixTest, MaxElementTest) {
    NNMatrix matrix(3, 1);
    matrix.set(0, 0, 1.0f);
    matrix.set(1, 0, 3.0f);
    matrix.set(2, 0, 100.f);

    ASSERT_EQ(2, matrix.getIndexOfColMax(0));
    ASSERT_FLOAT_EQ(100.0f, matrix.getColMax(0));
}

TEST(NNMatrixTest, StatisticsTest) {
    NNMatrix matrix(2, 2, 0.0f);
    matrix.set(0, 0, 1.0f);
    matrix.set(0, 1, 3.0f);
    matrix.set(1, 0, 5.0f);
    matrix.set(1, 1, 7.0f);

    EXPECT_FLOAT_EQ(4.0f, matrix.mean());
    EXPECT_NEAR(std::sqrt(5.0f), matrix.std(), 1e-6f);

    const auto normalized = matrix.normalized();
    const float invStd = 1.0f / std::sqrt(5.0f);
    EXPECT_NEAR((1.0f - 4.0f) * invStd, normalized.get(0, 0), 1e-6f);
    EXPECT_NEAR((3.0f - 4.0f) * invStd, normalized.get(0, 1), 1e-6f);
    EXPECT_NEAR((5.0f - 4.0f) * invStd, normalized.get(1, 0), 1e-6f);
    EXPECT_NEAR((7.0f - 4.0f) * invStd, normalized.get(1, 1), 1e-6f);
}

TEST(NNMatrixTest, StatisticsStayStableForLargeOffsetValues) {
    NNMatrix matrix(2, 2, 0.0f);
    matrix.set(0, 0, 123456.5f);
    matrix.set(0, 1, 123457.5f);
    matrix.set(1, 0, 123455.5f);
    matrix.set(1, 1, 123458.5f);

    EXPECT_NEAR(123457.0f, matrix.mean(), 1e-3f);
    EXPECT_NEAR(std::sqrt(1.25f), matrix.std(), 1e-4f);

    const auto normalized = matrix.normalized();
    const float invStd = 1.0f / std::sqrt(1.25f);
    EXPECT_NEAR(-0.5f * invStd, normalized.get(0, 0), 1e-4f);
    EXPECT_NEAR(0.5f * invStd, normalized.get(0, 1), 1e-4f);
    EXPECT_NEAR(-1.5f * invStd, normalized.get(1, 0), 1e-4f);
    EXPECT_NEAR(1.5f * invStd, normalized.get(1, 1), 1e-4f);
}

TEST(NNMatrixTest, StatisticsCacheInvalidatesAfterMutation) {
    NNMatrix matrix(2, 2, 1.0f);

    EXPECT_FLOAT_EQ(1.0f, matrix.mean());
    EXPECT_FLOAT_EQ(0.0f, matrix.std());

    matrix.set(0, 0, 5.0f);
    EXPECT_FLOAT_EQ(2.0f, matrix.mean());
    EXPECT_NEAR(std::sqrt(3.0f), matrix.std(), 1e-6f);

    float* raw = matrix.data();
    ASSERT_NE(raw, nullptr);
    raw[1] = 9.0f;

    EXPECT_FLOAT_EQ(4.0f, matrix.mean());
    EXPECT_NEAR(std::sqrt(11.0f), matrix.std(), 1e-6f);
}

TEST(NNMatrixTest, ExplicitNormalizationHelpersTest) {
    NNMatrix matrix(2, 2, 0.0f);
    matrix.set(0, 0, 1.0f);
    matrix.set(0, 1, 3.0f);
    matrix.set(1, 0, 5.0f);
    matrix.set(1, 1, 7.0f);

    EXPECT_EQ(4, matrix.elementCount());
    EXPECT_NEAR(20.0, matrix.squaredDiffSum(2.0), 1e-9);

    const auto normalized = matrix.normalized(4.0, 1.0 / std::sqrt(5.0));
    const float invStd = 1.0f / std::sqrt(5.0f);
    EXPECT_NEAR((1.0f - 4.0f) * invStd, normalized.get(0, 0), 1e-6f);
    EXPECT_NEAR((3.0f - 4.0f) * invStd, normalized.get(0, 1), 1e-6f);
    EXPECT_NEAR((5.0f - 4.0f) * invStd, normalized.get(1, 0), 1e-6f);
    EXPECT_NEAR((7.0f - 4.0f) * invStd, normalized.get(1, 1), 1e-6f);
}
