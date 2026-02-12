#pragma once

#include "NNFunctions.h"
#include "NNLayer.h"
#include "NNMatrix.h"
#include "NNUtils.h"

class FCNNLayer : public NNLayer { // fully connecter neural network layer
  public:
    FCNNLayer(int inputSize = 1, int outputSize = 1);
    NNMatrix forward(const NNMatrix& input, MatrixFunc derivateFunc = NNFunctions::SigmoidDrevative,
                     bool debug = false);
    NNMatrixPtrV forward(const NNMatrixPtrV& input);
    NNMatrix calculatePrevLayerDA(const NNMatrix& dz);
    NNMatrix setDz(NNMatrix&& other);
    void update(const NNMatrix& dw, const NNMatrix& db, float alpha, float momentum);
    int getInputSize() const { return weight.getColSize(); }
    int getOutputSize() const { return weight.getRowSize(); }
    void dump();

  private:
    const std::string TAG = "FCNNLayer";
    NNMatrix weight;
    NNMatrix vWeight;
    NNMatrix bias;
    NNMatrix vBias;
    NNMatrix dz_;
    int batchSize = 1;
};
