#pragma once

// Default DNN training / topology parameters (MNIST).
inline constexpr int DEFAULT_DNN_INPUT_SIZE = 784; // 28x28 pixels
inline constexpr int DEFAULT_DNN_HIDDEN1_SIZE = 128;
inline constexpr int DEFAULT_DNN_HIDDEN2_SIZE = 64;
inline constexpr int DEFAULT_DNN_OUTPUT_SIZE = 10;

inline constexpr int DEFAULT_DNN_EPOCHS = 9;
inline constexpr int DEFAULT_DNN_BATCH_SIZE = 16;
inline constexpr float DEFAULT_DNN_LEARNING_RATE = 0.005f;
inline constexpr float DEFAULT_DNN_MOMENTUM = 0.9f;
