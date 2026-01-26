#pragma once

#include <functional>
#include "gtest/gtest.h"
#include "../include/NNMatrix.h"

bool isEqual(const std::vector<float> &A, const std::vector<float> &B) {
  return std::equal(A.begin(), A.end(), B.begin(), [](float x, float y) {
    return std::fabs(x - y) < 1e-6f;
  });
}

TEST(NNMatrixTest, ConstructorTest) {
  NNMatrix matrix(3, 5, 1.0f);
  ASSERT_FLOAT_EQ(1.0f, matrix.get(2, 2));
  ASSERT_EQ(3, matrix.getRowSize());
  ASSERT_EQ(5, matrix.getColSize());

  std::vector<float> expectedCol {1.0f, 1.0f, 1.0f};
  std::vector<float> expectedRow {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

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
  for (int i = 0; i < 3; i++ ) {
    for (int j = 0; j < 2; j++) {
      matrix2.set(i, j, (i+1) * (j+1));
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


