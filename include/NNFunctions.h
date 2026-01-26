#pragma once

#include <string>
#include "NNMatrix.h"

typedef std::function<float(float)> MatrixFunc;

class NNFunctions {
private:
    static const std::string TAG;

public:
    static MatrixFunc SigmoidDrevative;
    static MatrixFunc SigmoidFunc;
    static NNMatrix softmax(const NNMatrix &matrix);
};