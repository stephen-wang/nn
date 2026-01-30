#include "NNFunctions.h"
#include "NNUtils.h"

const std::string NNFunctions::TAG = "NNFunctions";
MatrixFunc NNFunctions::SigmoidFunc = [](float x)
{
    if (x < -700)
        return 0.0f;
    else if (x > 700)
        return 1.0f;
    else
        return 1.0f / (1.0f + std::exp(-x));
};

MatrixFunc NNFunctions::SigmoidDrevative = [](float y)
{ return y * (1 - y); };

MatrixFunc NNFunctions::ReLUFunc = [](float x)
{ return x > 0.0f ? x : 0.0f; };
MatrixFunc NNFunctions::ReLUDrevative = [](float y)
{ return y > 0.0f ? 1.0f : 0.0f; };

NNMatrix NNFunctions::softmax(const NNMatrix &input)
{
    const int rows = input.getRowSize();
    const int cols = input.getColSize();
    NNMatrix ret(rows, 1);
    if (rows <= 0 || cols != 1)
    {
        LOG << "Invalid input, row size " << rows << ", col size " << cols << std::endl;
        return ret;
    }

    float colMax = input.getColMax(0);
    float sum = 0.0f;
    for (int i = 0; i < rows; i++)
    {
        float val = std::expf(input.get(i, 0) - colMax);
        sum += val;
        ret.set(i, 0, val);
    }

    sum = std::max(sum, 1e-5f);
    ret /= sum;
    return ret;
}