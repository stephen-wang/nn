#pragma once

const int DEFAULT_CNN_INPUT_SIZE = 28;
const int DEFAULT_CNN_CONV_FILTER_SIZE = 3;
const int DEFAULT_CNN_CONV_PADDING_SIZE = 1;
const int DEFAULT_CNN_CONV_STRIDE_SIZE = 1;
const int DEFAULT_CNN_CONV_OUTPUT_SIZE = 28;

const int DEFAULT_CNN_POOLING_INPUT_SIZE = 28;
const int DEFAULT_CNN_POOLING_FILTER_SIZE = 2;
const int DEFAULT_CNN_POOLING_STRIDE_SIZE = 2;
const int DEFAULT_CNN_POOLING_OUTPUT_SIZE = 13;

const int DEFAULT_CNN_FLATTEN_INPUT_SIZE = 13;
const int DEFAULT_CNN_OUTPUT_SIZE = 10;

// CIFAR-100 (32x32 RGB, 100 classes)
// Note: this project represents a single sample as a contiguous block of matrices
// (R,G,B) in the flat trainInput_ vector.
const int CIFAR100_CNN_IN_CHANNELS = 3;
const int CIFAR100_CNN_CONV1_OUT_CHANNELS = 16;
const int CIFAR100_CNN_CONV2_OUT_CHANNELS = 32;
const int CIFAR100_CNN_CONV_FILTER_SIZE = 3;
const int CIFAR100_CNN_CONV_PADDING_SIZE = 1;
const int CIFAR100_CNN_CONV_STRIDE_SIZE = 1;
const int CIFAR100_CNN_POOLING_FILTER_SIZE = 2;
const int CIFAR100_CNN_POOLING_STRIDE_SIZE = 2;

// After two 3x3 convs (stride=1,pad=1): 32x32 -> 32x32. After 2x2 pool stride 2: 16x16.
const int CIFAR100_CNN_AFTER_POOL_SIDE = 16;
const int CIFAR100_CNN_FC1_OUT_SIZE = 256;
const int CIFAR100_CNN_FC1_IN_SIZE =
    CIFAR100_CNN_CONV2_OUT_CHANNELS * CIFAR100_CNN_AFTER_POOL_SIDE * CIFAR100_CNN_AFTER_POOL_SIDE;
const int CIFAR100_CNN_OUTPUT_SIZE = 100;

const int CIFAR100_CNN_EPOCHS = 9;
const int CIFAR100_CNN_BATCH_SIZE = 64;
const float CIFAR100_CNN_LEARNING_RATE = 0.02f;
const float CIFAR100_CNN_MOMENTUM = 0.9f;

// Practical defaults: CNN training is CPU-heavy; keep the default run small to iterate faster.
// Increase these (or set to <=0) for full-dataset training.
const int CIFAR100_CNN_MAX_TRAIN_SAMPLES = 50000;
const int CIFAR100_CNN_MAX_TEST_SAMPLES = 10000;
