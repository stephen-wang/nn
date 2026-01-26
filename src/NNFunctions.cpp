#include "NNFunctions.h"
#include "NNUtils.h"

const std::string NNFunctions::TAG = "NNFunctions";
MatrixFunc NNFunctions::SigmoidFunc = [](float x) {
    if (x < -700)
        return 0.0f;
    else if (x > 700)
        return 1.0f;
    else
        return 1.0f / (1.0f + std::exp(-x));
};

MatrixFunc NNFunctions::SigmoidDrevative = [](float y) { return y * (1 - y); };

NNMatrix NNFunctions::softmax(const NNMatrix &input) {
    NNMatrix ret(input.getRowSize(), 1);
    if (input.getColSize() != 1) {
        LOG << "Invalid input, col size " << input.getColSize() << std::endl;
        return ret;
    }

    float colMax = input.getColMax(0);
    float sum = 0.0f;
    for (int i = 0; i < input.getRowSize(); i++) {
        float val = std::expf(input.get(i, 0) - colMax);
        sum += val;
        ret.set(i, 0, val);
    }

    sum = std::max(sum, 1e-5f);
    ret = ret.applyFunction([sum](float x) { return x/ sum; });
    return ret;
}