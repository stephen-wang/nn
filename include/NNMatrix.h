#pragma once

#include <assert.h>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using NNVector = std::vector<float>;
using MatrixFunc = std::function<float(float)>;

class NNMatrix : public std::enable_shared_from_this<NNMatrix> {
  public:
    NNMatrix(int row, int col, float defaultValue = 0.0f);
    NNMatrix(const NNMatrix& other);
    // NNMatrix(NNMatrix &&other);
    virtual ~NNMatrix();
    int getColSize() const { return col_; }
    int getRowSize() const { return row_; }
    NNVector getRow(int row) const;
    NNVector getCol(int col) const;
    void set(int i, int j, float elemValue);
    float get(int i, int j) const;
    int elementCount() const { return row_ * col_; }
    float mean() const;
    float std() const;
    NNMatrix normalized(float eps = 1e-6f) const;
    NNMatrix normalized(double mean, double invStd) const;
    double squaredDiffSum(double mean) const;
    NNMatrix operator-(const NNMatrix& other);
    NNMatrix& operator-=(const NNMatrix& other);
    NNMatrix& operator+=(const NNMatrix& other);
    NNMatrix& operator/=(float ratio);
    NNMatrix& operator*=(float ratio);
    NNMatrix& operator=(const NNMatrix& other);
    NNMatrix& operator=(NNMatrix&& other) noexcept;
    NNMatrix& flatten() noexcept;
    NNMatrix dotProduct(const NNMatrix& other);
    NNMatrix elementProduct(const NNMatrix& other);
    NNMatrix applyFunction(const MatrixFunc& func);
    inline bool hasSameDimension(const NNMatrix& other) const noexcept {
        return row_ == other.row_ && col_ == other.col_;
    }
    void applyFunctionInplace(const MatrixFunc& func);

    int getIndexOfColMax(int col) const;
    float getColMax(int col) const;

    // Returns the maximum value within a square region starting at (i, j).
    // - (i, j) is the top-left element (row, col).
    // - The requested region size is stride x stride.
    // - If the region exceeds matrix bounds, it is clamped to the valid area.
    // Preconditions: stride > 0, 0 <= i < row, 0 <= j < col.
    float getRegionMax(int i, int j, int stride);
    void dump(bool showFullLine = false, int lineSize = -1, bool dumpToFile = false) const;
    void toOneHot();

    float* data() {
        invalidateStatisticsCache();
        return mem_;
    }
    const float* data() const { return mem_; }

  private:
    void invalidateStatisticsCache();
    void updateStatisticsCache() const;

    const std::string TAG = "NNMatrix";
    const int MAX_DUMP_LINE_SIZE = 28;
    float* mem_ = nullptr;
    int row_ = 0;
    int col_ = 0;
    mutable bool statisticsCacheValid_ = false;
    mutable float cachedMean_ = 0.0f;
    mutable float cachedStandardDeviation_ = 0.0f;
};

using NNMatrixPtr = std::shared_ptr<NNMatrix>;
using NNMatrixPtrV = std::vector<NNMatrixPtr>;
using NNMatrixPtrVV = std::vector<NNMatrixPtrV>;
