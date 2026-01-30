#pragma once

#include "NNMatrix.h"

#include <string>

class NNFunctions {
  private:
    static const std::string TAG;

  public:
    static MatrixFunc SigmoidDrevative;
    static MatrixFunc SigmoidFunc;
    static MatrixFunc ReLUFunc;
    static MatrixFunc ReLUDrevative;
    static NNMatrix softmax(const NNMatrix& matrix);
};