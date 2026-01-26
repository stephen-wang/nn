#include "NNUtils.h"
#include <random>
#include <fstream>

const std::string NNUtils::TAG = "NNUtils";
uint32_t NNUtils::swap_endian(uint32_t val) {
    val = ((val << 8) & 0xff00ff00) | ((val >> 8) & 0xff00ff);
    val = (val >> 16) | (val << 16);
    return val;
}

float NNUtils::xavierInit(int inputSize, int outputSize) {
  float limit = std::sqrt(6.0f / float((inputSize + outputSize)));
  return NNUtils::random(-limit, limit);
} 

void NNUtils::normalizeMnistData(std::vector<NNMatrixPtr>& data) {
    for (auto &inputPtr : data) {
        auto &input = *inputPtr;
        input /= 255.0f;
    }
}

void NNUtils::normalizeMnistLabel(std::vector<NNMatrixPtr>& labels) {
    for (auto &labelPtr : labels) {
        auto &label = *labelPtr;
        label.toOneHot();
    }
}
  

std::vector<NNMatrixPtr> NNUtils::read_mnist_data(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open " + filePath); 
    }
  
    int magic;
    file.read((char *)&magic, sizeof(magic));
    magic = swap_endian(magic);
    if (magic != MNIST_IMAGE_MAGIC) {
      throw std::runtime_error("Invalid mnist image file!");
    }
  
    int numImages = 0, row = 0, col = 0;
    file.read((char *)&numImages, sizeof(numImages));
    numImages = swap_endian(numImages);
    
    file.read((char *)&row, sizeof(row));
    row = swap_endian(row);
  
    file.read((char *)&col, sizeof(col));
    col= swap_endian(col);
  
    std::cout << "Totally, " << numImages << " images, width " << col << ", height " << row << std::endl;
    std::vector<NNMatrixPtr> result(numImages);
    for (int i = 0; i < numImages; i++) {
      auto imgSize = row * col;
      auto imgData = std::make_shared<NNMatrix>(imgSize, 1);
      std::vector<unsigned char> buffer(imgSize, 0);
      file.read(reinterpret_cast<char *>(buffer.data()), imgSize);
      for (int j = 0; j < imgSize; j++) {
        imgData->set(j, 0, static_cast<float>(buffer[j]));
      }
  
      result[i] = imgData;
    }

    return result;
}

std::vector<NNMatrixPtr> NNUtils::read_mnist_labels(const std::string& filePath) {
    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("Unable to open " + filePath); 
    }
  
    int magic;
    file.read((char *)&magic, sizeof(magic));
    magic = swap_endian(magic);
    if (magic != MNIST_LABEL_MAGIC) {
      throw std::runtime_error("Invalid mnist image file!");
    }
  
    int numLabels = 0;
    file.read((char *)&numLabels, sizeof(numLabels));
    numLabels = swap_endian(numLabels);
    std::cout << "Totally, " << numLabels << " labels" << std::endl;
  
    std::vector<NNMatrixPtr> result(numLabels);
    for (int i = 0; i < numLabels; i++) {
      unsigned char ch = 0;  
      file.read((char *)&ch, 1);
      assert(ch >= 0 && ch <= 9);
      
      auto label = std::make_shared<NNMatrix>(10, 1);
      label->set(ch, 0, 1.0f);
  
      result[i] = label;
    }
  
    file.close();
    return result;
  }

float NNUtils::random(float a, float b) {
    static std::random_device rd; // Non-deterministic random seed
    static std::mt19937 gen(rd()); // Mersenne Twister engine
    std::uniform_real_distribution<float> dist(a, b); // Range [a, b]
    return dist(gen);
}

void NNUtils::shuffle(std::vector<NNMatrixPtr>& input, std::vector<NNMatrixPtr>& label) {
    assert(input.size() == label.size());

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 gen(seed); // Mersenne Twister engine
    
    std::vector<std::pair<NNMatrixPtr, NNMatrixPtr>> combinedData;
    for (int i = 0; i < input.size(); i++) {
      combinedData.push_back(std::make_pair(input[i], label[i]));
    }
    std::shuffle(combinedData.begin(), combinedData.end(), gen);
    
    input.clear();
    label.clear();
    for (auto &data : combinedData) {
        input.push_back(data.first);
        label.push_back(data.second);
    }
}

std::vector<NNMatrixPtr> NNUtils::getBatch(std::vector<NNMatrixPtr>& input, int batchNo, int batchSize) {
    std::vector<NNMatrixPtr> ret;

    int start = batchNo * batchSize;
    int end = std::min(start + batchSize, (int)input.size());
    for (int i = start; i < end; i++) {
        ret.push_back(input[i]);
    }
    
    return ret;
}
