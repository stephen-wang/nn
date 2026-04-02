#include "NNMatrix.h"

#include "NNUtils.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#if defined(NN_ENABLE_OMP)
#include <omp.h>
#endif

NNMatrix::NNMatrix(int row, int col, float defaultValue) {
    if (row <= 0 || col <= 0) {
        LOG << "Invalid row/col " << row << "/" << col << std::endl;
        return;
    }

    row_ = row;
    col_ = col;
    auto elemCount = static_cast<size_t>(row) * static_cast<size_t>(col);
    mem_ = new float[elemCount];
    std::fill_n(mem_, elemCount, defaultValue);
}

NNMatrix::NNMatrix(const NNMatrix& other) : std::enable_shared_from_this<NNMatrix>(other) {
    row_ = other.row_;
    col_ = other.col_;
    auto elemCount = static_cast<size_t>(row_) * static_cast<size_t>(col_);
    mem_ = new float[elemCount];
    memcpy(mem_, other.mem_, elemCount * sizeof(float));
    statisticsCacheValid_ = other.statisticsCacheValid_;
    cachedMean_ = other.cachedMean_;
    cachedStandardDeviation_ = other.cachedStandardDeviation_;
}

NNMatrix::~NNMatrix() {
    if (mem_ != nullptr) {
        row_ = 0;
        col_ = 0;
        delete[] mem_;
        mem_ = nullptr;
    }
}

NNMatrix& NNMatrix::operator=(const NNMatrix& other) {
    if (this == &other) {
        return *this;
    }

    if (mem_ != nullptr) {
        delete[] mem_;
        mem_ = nullptr;
    }

    row_ = other.row_;
    col_ = other.col_;
    if (row_ > 0 && col_ > 0) {
        auto elemCount = static_cast<size_t>(row_) * static_cast<size_t>(col_);
        mem_ = new float[elemCount];
        memcpy(mem_, other.mem_, elemCount * sizeof(float));
    }

    statisticsCacheValid_ = other.statisticsCacheValid_;
    cachedMean_ = other.cachedMean_;
    cachedStandardDeviation_ = other.cachedStandardDeviation_;

    return *this;
}

NNMatrix& NNMatrix::operator=(NNMatrix&& other) noexcept {
    if (this == &other) {
        return *this;
    }

    if (mem_ != nullptr) {
        delete[] mem_;
        mem_ = nullptr;
    }

    mem_ = other.mem_;
    row_ = other.row_;
    col_ = other.col_;
    statisticsCacheValid_ = other.statisticsCacheValid_;
    cachedMean_ = other.cachedMean_;
    cachedStandardDeviation_ = other.cachedStandardDeviation_;
    other.mem_ = nullptr;
    other.row_ = 0;
    other.col_ = 0;
    other.statisticsCacheValid_ = false;
    other.cachedMean_ = 0.0f;
    other.cachedStandardDeviation_ = 0.0f;

    return *this;
}

NNMatrix& NNMatrix::flatten() noexcept {
    if (mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return *this;
    }

    row_ *= col_;
    col_ = 1;
    return *this;
}

NNVector NNMatrix::getRow(int row) const {
    std::vector<float> ret;
    if (row >= row_) {
        return ret;
    }

    for (int i = 0; i < col_; i++) {
        ret.push_back(mem_[row * col_ + i]);
    }

    return ret;
}

NNVector NNMatrix::getCol(int col) const {
    std::vector<float> ret;
    if (col >= col_) {
        return ret;
    }

    for (int i = 0; i < row_; i++) {
        ret.push_back(mem_[i * col_ + col]);
    }

    return ret;
}

void NNMatrix::set(int i, int j, float elemValue) {
    assert(i >= 0 && j >= 0);
    assert(i < row_ && j < col_);
    mem_[i * col_ + j] = elemValue;
    invalidateStatisticsCache();
}

float NNMatrix::get(int i, int j) const {
    assert(i >= 0 && j >= 0);
    assert(i < row_ && j < col_);
    return mem_[i * col_ + j];
}

float NNMatrix::mean() const {
    updateStatisticsCache();
    return cachedMean_;
}

float NNMatrix::std() const {
    updateStatisticsCache();
    return cachedStandardDeviation_;
}

NNMatrix NNMatrix::normalized(float eps) const {
    NNMatrix ret(row_, col_);
    if (mem_ == nullptr || ret.mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return ret;
    }

    updateStatisticsCache();

    const int total = row_ * col_;
    const double mu = static_cast<double>(cachedMean_);
    const double sigma = static_cast<double>(cachedStandardDeviation_);
    const double denom = std::max(sigma, static_cast<double>(eps));
    float* out = ret.mem_;

    for (int idx = 0; idx < total; ++idx) {
        out[idx] = static_cast<float>((static_cast<double>(mem_[idx]) - mu) / denom);
    }

    return ret;
}

NNMatrix NNMatrix::normalized(double mean, double invStd) const {
    NNMatrix ret(row_, col_);
    if (mem_ == nullptr || ret.mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return ret;
    }

    const int total = row_ * col_;
    float* out = ret.data();
    for (int idx = 0; idx < total; ++idx) {
        out[idx] = static_cast<float>((static_cast<double>(mem_[idx]) - mean) * invStd);
    }

    return ret;
}

double NNMatrix::squaredDiffSum(double mean) const {
    if (mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return 0.0;
    }

    updateStatisticsCache();
    const double matrixMean = static_cast<double>(cachedMean_);
    const double matrixStd = static_cast<double>(cachedStandardDeviation_);
    const double meanDiff = matrixMean - mean;
    return static_cast<double>(elementCount()) * (matrixStd * matrixStd + meanDiff * meanDiff);
}

NNMatrix NNMatrix::dotProduct(const NNMatrix& other) {
    assert(other.row_ == col_);

    NNMatrix ret(row_, other.col_);
    if (mem_ == nullptr || other.mem_ == nullptr || ret.mem_ == nullptr || row_ <= 0 || col_ <= 0 ||
        other.col_ <= 0) {
        return ret;
    }

    const int m = row_;
    const int n = other.col_;
    const int kDim = col_;

    // Compute C = A(m x kDim) * B(kDim x n). Matrices are stored row-major.
    // We use loop order i -> j -> k so B is accessed row-wise (contiguous) in the inner loop.
    // ret is constructed with defaultValue=0, so we can accumulate directly into its buffer.

    float* out = ret.mem_;
    const float* a = mem_;
    const float* b = other.mem_;

#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++) {
#else
    for (int i = 0; i < m; i++) {
#endif
        float* outRow = out + i * n;
        const float* aRow = a + i * kDim;
        for (int j = 0; j < kDim; j++) {
            const float aVal = aRow[j];
            const float* bRow = b + j * n;
            // outRow[kk] += A(i, j) * B(j, kk)
            for (int kk = 0; kk < n; kk++) {
                outRow[kk] += aVal * bRow[kk];
            }
        }
    }

    return ret;
}

NNMatrix NNMatrix::elementProduct(const NNMatrix& other) {
    assert(row_ == other.row_);
    assert(col_ == other.col_);

    NNMatrix ret(row_, col_);
    const int total = row_ * col_;
    const float* a = mem_;
    const float* b = other.mem_;
    float* out = ret.mem_;
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; idx++) {
        out[idx] = a[idx] * b[idx];
    }
#else
    for (int idx = 0; idx < total; idx++) {
        out[idx] = a[idx] * b[idx];
    }
#endif

    return ret;
}

NNMatrix& NNMatrix::operator+=(const NNMatrix& other) {
    if (row_ != other.row_ || col_ != other.col_) {
        LOG << "mismatched matrix size" << std::endl;
        return *this;
    }

    const int total = row_ * col_;
    float* dst = mem_;
    const float* src = other.mem_;
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; idx++) {
        dst[idx] += src[idx];
    }
#else
    for (int idx = 0; idx < total; idx++) {
        dst[idx] += src[idx];
    }
#endif
    invalidateStatisticsCache();
    return *this;
}

NNMatrix NNMatrix::operator-(const NNMatrix& other) {
    if (row_ != other.row_ || col_ != other.col_) {
        LOG << "mismatched matrix size" << std::endl;
        return *this;
    }

    NNMatrix ret(row_, col_);

    const int total = row_ * col_;
    const float* a = mem_;
    const float* b = other.mem_;
    float* out = ret.mem_;
    for (int idx = 0; idx < total; idx++) {
        out[idx] = a[idx] - b[idx];
    }
    return ret;
}

NNMatrix& NNMatrix::operator-=(const NNMatrix& other) {
    if (row_ != other.row_ || col_ != other.col_) {
        LOG << "mismatched matrix size" << std::endl;
        return *this;
    }

    const int total = row_ * col_;
    float* dst = mem_;
    const float* src = other.mem_;
    for (int idx = 0; idx < total; idx++) {
        dst[idx] -= src[idx];
    }
    invalidateStatisticsCache();
    return *this;
}

NNMatrix& NNMatrix::operator*=(float ratio) {
    const int total = row_ * col_;
    for (int idx = 0; idx < total; idx++) {
        mem_[idx] *= ratio;
    }
    invalidateStatisticsCache();
    return *this;
}

NNMatrix& NNMatrix::operator/=(float ratio) {
    if (fabs(ratio) < 1e-6) {
        LOG << "cannot divide by 0" << std::endl;
        return *this;
    }

    const int total = row_ * col_;
    for (int idx = 0; idx < total; idx++) {
        mem_[idx] /= ratio;
    }
    invalidateStatisticsCache();
    return *this;
}

NNMatrix NNMatrix::applyFunction(const MatrixFunc& func) {
    NNMatrix ret(row_, col_);
    if (mem_ == nullptr || ret.mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return ret;
    }

    const int total = row_ * col_;
    const float* src = mem_;
    float* dst = ret.mem_;
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; idx++) {
        dst[idx] = func(src[idx]);
    }
#else
    for (int idx = 0; idx < total; idx++) {
        dst[idx] = func(src[idx]);
    }
#endif
    return ret;
}

void NNMatrix::toOneHot() {
    if (mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return;
    }
    const int total = row_ * col_;
    constexpr float threshold = 1e-5f;
    for (int idx = 0; idx < total; idx++) {
        mem_[idx] = (mem_[idx] > threshold) ? 1.0f : 0.0f;
    }
    invalidateStatisticsCache();
}

void NNMatrix::applyFunctionInplace(const MatrixFunc& func) {
    if (!func) {
        return;
    }
    if (mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        return;
    }

    const int total = row_ * col_;
    float* dataPtr = mem_;
#if defined(NN_ENABLE_OMP)
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < total; ++idx) {
        dataPtr[idx] = func(dataPtr[idx]);
    }
#else
    for (int idx = 0; idx < total; ++idx) {
        dataPtr[idx] = func(dataPtr[idx]);
    }
#endif
    invalidateStatisticsCache();
}

void NNMatrix::invalidateStatisticsCache() {
    statisticsCacheValid_ = false;
}

void NNMatrix::updateStatisticsCache() const {
    if (statisticsCacheValid_) {
        return;
    }
    if (mem_ == nullptr || row_ <= 0 || col_ <= 0) {
        cachedMean_ = 0.0f;
        cachedStandardDeviation_ = 0.0f;
        statisticsCacheValid_ = true;
        return;
    }

    const int total = row_ * col_;
    double sum = 0.0;
    for (int idx = 0; idx < total; ++idx) {
        sum += static_cast<double>(mem_[idx]);
    }
    const double mu = sum / static_cast<double>(total);

    double squaredDiffSum = 0.0;
    for (int idx = 0; idx < total; ++idx) {
        const double diff = static_cast<double>(mem_[idx]) - mu;
        squaredDiffSum += diff * diff;
    }

    cachedMean_ = static_cast<float>(mu);
    cachedStandardDeviation_ =
        static_cast<float>(std::sqrt(squaredDiffSum / static_cast<double>(total)));
    statisticsCacheValid_ = true;
}

void NNMatrix::dump(bool showFullLine, int lineSize, bool dumpToFile) const {
    int numToDump = col_ * row_;
    std::stringstream ss;

    int i = 0;
    if (lineSize == -1) {
        lineSize = col_;
    }

    std::ofstream ofs;
    if (dumpToFile) {
        ofs.open("matrix.dump", std::ios::app);
    }
    ss << "Matrix (" << row_ << "x" << col_ << ")" << std::endl;
    while (i < numToDump) {
        if (lineSize > MAX_DUMP_LINE_SIZE) {
            int curPos = i % lineSize;
            if (showFullLine || curPos < MAX_DUMP_LINE_SIZE) {
                ss << std::setprecision(3) << mem_[i] << "\t";
            } else if (curPos == MAX_DUMP_LINE_SIZE) {
                ss << "......" << std::endl;
            }
        } else {
            if (lineSize == 1 || (i > 0 && (i % lineSize == 0))) {
                ss << std::endl;
            }
            ss << std::setprecision(3) << mem_[i] << "\t";
        }

        i++;
    }

    ss << std::endl;
    LOG << ss.str();
    if (dumpToFile) {
        ofs << ss.str();
        ofs.close();
    }
}

float NNMatrix::getColMax(int col) const {
    assert(col < col_);
    const float* ptr = mem_ + col;
    float ret = *ptr;
    for (int i = 1; i < row_; i++) {
        ptr += col_;
        const float v = *ptr;
        if (ret < v) {
            ret = v;
        }
    }

    return ret;
}

int NNMatrix::getIndexOfColMax(int col) const {
    assert(col < col_);
    const float* ptr = mem_ + col;
    float maxVal = *ptr;
    int ret = 0;
    for (int i = 1; i < row_; i++) {
        ptr += col_;
        const float v = *ptr;
        if (maxVal < v) {
            maxVal = v;
            ret = i;
        }
    }

    return ret;
}

float NNMatrix::getRegionMax(int i, int j, int stride) {
    assert(mem_ != nullptr);
    assert(stride > 0);
    assert(i >= 0 && j >= 0);
    assert(i < row_ && j < col_);

    const int endRow = std::min(i + stride, row_);
    const int endCol = std::min(j + stride, col_);

    float maxVal = mem_[i * col_ + j];
    for (int r = i; r < endRow; r++) {
        const float* cur = mem_ + r * col_ + j;
        for (int c = j; c < endCol; c++) {
            const float v = cur[c - j];
            if (v > maxVal) {
                maxVal = v;
            }
        }
    }

    return maxVal;
}
