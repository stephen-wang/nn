#include "NNStreamUtils.h"

bool NNStreamUtils::writeLayerType(std::ostream& os, NNLayerType type) {
    const std::uint8_t v = static_cast<std::uint8_t>(type);
    return writeScalar(os, v);
}

bool NNStreamUtils::readLayerType(std::istream& is, NNLayerType& type) {
    std::uint8_t v = 0;
    if (!readScalar(is, v)) {
        return false;
    }
    type = static_cast<NNLayerType>(v);
    return true;
}

bool NNStreamUtils::writeVector(std::ostream& os, const std::vector<float>& values) {
    const std::int32_t count = static_cast<std::int32_t>(values.size());
    os.write(reinterpret_cast<const char*>(&count), sizeof(count));
    if (!os.good() || count < 0) {
        return false;
    }
    if (count == 0) {
        return true;
    }
    os.write(reinterpret_cast<const char*>(values.data()),
             static_cast<std::streamsize>(values.size() * sizeof(float)));
    return os.good();
}

bool NNStreamUtils::readVector(std::istream& is, std::vector<float>& values) {
    std::int32_t count = 0;
    is.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!is.good() || count < 0) {
        return false;
    }
    values.assign(static_cast<std::size_t>(count), 0.0f);
    if (count == 0) {
        return true;
    }
    is.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    return is.good();
}