#include "NNMatrix.h"
#include "NNUtils.h"
#include <sstream>
#include <fstream>
#include <iomanip>

NNMatrix::NNMatrix(int row, int col, float defaultValue) {
    if (row <= 0 || col <= 0) {
        LOG << "Invalid row/col " << row << "/" << col << std::endl;
        return;
    }

    row_ = row;
    col_ = col;
    auto memSize = row * col;
    mem_ = new float[memSize];
    std::fill_n(mem_, memSize, defaultValue);
}

NNMatrix::NNMatrix(const NNMatrix &other) {
    row_ = other.row_;
    col_ = other.col_;
    auto memSize = row_ * col_ * sizeof(float);
    mem_ = new float[memSize];
    memcpy(mem_, other.mem_, memSize);
}

NNMatrix::~NNMatrix() {
    if (mem_ != nullptr) {
        row_ = 0;
        col_ = 0;
        delete [] mem_;
        mem_ = nullptr;
    }
}

NNMatrix& NNMatrix::operator =(const NNMatrix &other) {
    if (this == &other) {
        return *this;
    }

    if (mem_ != nullptr) {
        delete [] mem_;
        mem_ = nullptr;
    }

    row_ = other.row_;
    col_ = other.col_;
    if (row_ > 0 && col_ > 0) {
        auto memSize = row_ * col_ * sizeof(float);
        mem_ = new float[memSize];
        memcpy(mem_, other.mem_, memSize);
    }

    return *this;
}

NNMatrix& NNMatrix::operator =(NNMatrix &&other) {
    if (this == &other) {
        return *this;
    }
    
    if (mem_ != nullptr) {
        delete [] mem_;
        mem_ = nullptr;
    }

    mem_ = other.mem_;
    row_ = other.row_;
    col_ = other.col_;
    other.mem_ = nullptr;
    other.row_ = 0;
    other.col_ = 0;

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
        ret.push_back(mem_[i * col_ + col ]);
    }

    return ret;
}

void NNMatrix::set(int i, int j, float elemValue) {
    assert(i >= 0 && j >= 0);
    assert(i < row_ && j < col_);
    mem_[i * col_ + j] = elemValue;
}

float NNMatrix::get(int i, int j) const {
    assert(i >= 0 && j >= 0);
    assert(i < row_ && j < col_);
    return mem_[i * col_ + j];
}

NNMatrix NNMatrix::dotProduct(const NNMatrix& other) {
    assert(other.row_ == col_);

    NNMatrix ret(row_, other.col_);
    for (int i = 0; i < row_; i++) {
        for (int k = 0; k < other.col_; k++) {
            float elem = 0.0f;
            for (int j = 0; j < col_; j++) {
                elem += mem_[i * col_ + j] * other.mem_[j * other.col_ + k];
            }
            ret.set(i, k, elem);
        }
    }

    return ret;
}

NNMatrix NNMatrix::elementProduct(const NNMatrix& other) {
    assert(row_ == other.row_);
    assert(col_ == other.col_);

    NNMatrix ret(row_, col_);
    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            ret.set(i, j, mem_[i * col_ + j] * other.mem_[i * col_ + j]);
        }
    }

    return ret;
}

NNMatrix& NNMatrix::operator +=(const NNMatrix& other) {
    if (row_ != other.row_ || col_ != other.col_) {
        LOG << "mismatched matrix size" << std::endl;
        return *this;
    }

    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            mem_[i * col_ + j] += other.get(i, j);
        }
    }
    return *this;
}

NNMatrix NNMatrix::operator - (const NNMatrix& other) {
    if (row_ != other.row_ || col_ != other.col_) {
        LOG << "mismatched matrix size" << std::endl;
        return *this;
    }

    NNMatrix ret(row_, col_);
    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            ret.set(i, j, get(i, j) - other.get(i, j));
        }
    }
    return ret;
}


NNMatrix& NNMatrix::operator -= (const NNMatrix& other) {
    if (row_ != other.row_ || col_ != other.col_) {
        LOG << "mismatched matrix size" << std::endl;
        return *this;
    }

    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            mem_[i * col_ + j] -= other.get(i, j);
        }
    }
    return *this;
}

NNMatrix& NNMatrix::operator *= (float ratio) {
    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            mem_[i * col_ + j] *= ratio;
        }
    }
    return *this;
}

NNMatrix& NNMatrix::operator /= (float ratio) {
    if (fabs(ratio) < 1e-6) {
        LOG << "cannot divide by 0" << std::endl;
        return *this;
    }

    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            mem_[i * col_ + j] /= ratio;
        }
    }
    return *this;
}

NNMatrix NNMatrix::applyFunction(const MatrixFunc &func) {
    NNMatrix ret(row_, col_);
    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            ret.set(i, j, func(mem_[i * col_ + j]));
        }
    }
    return ret;
}

void NNMatrix::toOneHot() {
    for (int i = 0; i < row_; i++) {
        for (int j = 0; j < col_; j++) {
            float rawVal = mem_[i * col_ + j];
            mem_[i * col_ + j] = (rawVal > 1e-5 ? 1.0f : 0.0f);
        }
    }
}

void NNMatrix::dump(bool showFullLine, int lineSize, bool dumpToFile) const {
    int numToDump = col_ * row_;
    std::stringstream ss;
    
    int i = 0;
    if (lineSize == -1) {
        lineSize = col_;
    }

    // if (lineSize == 1) {
    //     lineSize = row_;
    // }

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
    float ret = mem_[col];
    for (int i = 1; i < row_; i++) {
        if (ret < mem_[i * col_ + col]) {
            ret = mem_[i * col_ + col];
        }
    }

    return ret;
}

int NNMatrix::getIndexOfColMax(int col) const {
    assert(col < col_);
    float maxVal = mem_[col];
    int ret = 0;
    for (int i = 1; i < row_; i++) {
        if (maxVal < mem_[i * col_ + col]) {
            maxVal = mem_[i * col_ + col];
            ret = i;
        }
    }

    return ret;
}
