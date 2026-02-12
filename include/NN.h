#pragma once

#include "NNMatrix.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

// Base class for neural networks.
// Currently provides shared loss helpers (cross-entropy for softmax outputs).
class NN {
  protected:
    static int argmax(const NNMatrix& x) {
        assert(x.getColSize() == 1);
        return x.getIndexOfColMax(0);
    }

    static float crossEntropyLoss(const NNMatrix& actual, const NNMatrix& expect) {
        // Keep behavior aligned with DNN::calculateCrossEntropyLoss.
        if (actual.getRowSize() != expect.getRowSize() || actual.getColSize() != 1 ||
            expect.getColSize() != 1) {
            return 0.0f;
        }

        const float eps = 1e-15f;
        float loss = 0.0f;
        for (int i = 0; i < actual.getRowSize(); i++) {
            const float expectElem = expect.get(i, 0);
            const float actualElem = actual.get(i, 0);
            const float actualElemClipped = std::max(eps, std::min(1.0f - eps, actualElem));
            loss -= expectElem * std::log(actualElemClipped);
        }
        return loss;
    }

    static float batchCrossEntropyLoss(const std::vector<NNMatrixPtr>& actual,
                                       const std::vector<NNMatrixPtr>& expect) {
        if (expect.empty() || actual.size() != expect.size()) {
            return 0.0f;
        }

        float totalLoss = 0.0f;
        for (size_t i = 0; i < expect.size(); ++i) {
            if (!actual[i] || !expect[i]) {
                continue;
            }
            totalLoss += crossEntropyLoss(*actual[i], *expect[i]);
        }
        return totalLoss / static_cast<float>(expect.size());
    }

    static float batchAccuracy(const std::vector<NNMatrixPtr>& actual,
                               const std::vector<NNMatrixPtr>& expect) {
        if (expect.empty() || actual.size() != expect.size()) {
            return 0.0f;
        }

        int correct = 0;
        for (size_t i = 0; i < expect.size(); ++i) {
            if (!actual[i] || !expect[i]) {
                continue;
            }
            if (argmax(*actual[i]) == argmax(*expect[i])) {
                correct += 1;
            }
        }

        return static_cast<float>(correct) / static_cast<float>(expect.size());
    }

  public:
    virtual ~NN() = default;
};
