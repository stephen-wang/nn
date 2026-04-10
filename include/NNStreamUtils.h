#pragma once

#include "NNLayer.h"

#include <cstdint>
#include <istream>
#include <ostream>

class NNStreamUtils {
  public:
    template <typename T> static bool writeScalar(std::ostream& os, const T& value) {
        os.write(reinterpret_cast<const char*>(&value), sizeof(T));
        return os.good();
    }

    template <typename T> static bool readScalar(std::istream& is, T& value) {
        is.read(reinterpret_cast<char*>(&value), sizeof(T));
        return is.good();
    }

    static bool writeLayerType(std::ostream& os, NNLayerType type);
    static bool readLayerType(std::istream& is, NNLayerType& type);

    static bool writeVector(std::ostream& os, const std::vector<float>& values);
    static bool readVector(std::istream& is, std::vector<float>& values);
};