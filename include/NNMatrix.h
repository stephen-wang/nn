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
    NNMatrix operator-(const NNMatrix& other);
    NNMatrix& operator-=(const NNMatrix& other);
    NNMatrix& operator+=(const NNMatrix& other);
    NNMatrix& operator/=(float ratio);
    NNMatrix& operator*=(float ratio);
    NNMatrix& operator=(const NNMatrix& other);
    NNMatrix& operator=(NNMatrix&& other) noexcept;
    NNMatrix dotProduct(const NNMatrix& other);
    NNMatrix elementProduct(const NNMatrix& other);
    NNMatrix applyFunction(const MatrixFunc& func);
    int getIndexOfColMax(int col) const;
    float getColMax(int col) const;
    void dump(bool showFullLine = false, int lineSize = -1, bool dumpToFile = false) const;
    void toOneHot();

  private:
    const std::string TAG = "NNMatrix";
    const int MAX_DUMP_LINE_SIZE = 28;
    float* mem_ = nullptr;
    int row_ = 0;
    int col_ = 0;
};

using NNMatrixPtr = std::shared_ptr<NNMatrix>;
