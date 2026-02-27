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

// Conv1: 32x32x3 -> 32x32x32
// Conv2: 32x32x32 -> 32x32x64
// Pool1: 32x32x64 -> 16x16x64
// Conv3: 16x16x64 -> 16x16x128
// Pool2: 16x16x128 -> 8x8x128
const int CIFAR100_CNN_CONV1_OUT_CHANNELS = 32;
const int CIFAR100_CNN_CONV2_OUT_CHANNELS = 64;
const int CIFAR100_CNN_CONV3_OUT_CHANNELS = 128;
const int CIFAR100_CNN_CONV_FILTER_SIZE = 3;
const int CIFAR100_CNN_CONV_PADDING_SIZE = 1;
const int CIFAR100_CNN_CONV_STRIDE_SIZE = 1;
const int CIFAR100_CNN_POOLING_FILTER_SIZE = 2;
const int CIFAR100_CNN_POOLING_STRIDE_SIZE = 2;

// Spatial sizes: 32x32 -> (pool) 16x16 -> (pool) 8x8.
const int CIFAR100_CNN_AFTER_POOL1_SIDE = 16;
const int CIFAR100_CNN_AFTER_POOL2_SIDE = 8;

// FC1: 8 x 8 x 128 = 8192 x 1 -> 512 x 1
const int CIFAR100_CNN_FC1_OUT_SIZE = 512;
const int CIFAR100_CNN_FC1_IN_SIZE =
    CIFAR100_CNN_CONV3_OUT_CHANNELS * CIFAR100_CNN_AFTER_POOL2_SIDE * CIFAR100_CNN_AFTER_POOL2_SIDE;
// FC2: 512 x 1 -> 100 x 1
const int CIFAR100_CNN_OUTPUT_SIZE = 100;

const int CIFAR100_CNN_EPOCHS = 15;
const int CIFAR100_CNN_BATCH_SIZE = 64;
const float CIFAR100_CNN_LEARNING_RATE = 0.02f;
const float CIFAR100_CNN_MOMENTUM = 0.9f;

// Training regularization / stabilization.
const float CIFAR100_CNN_WEIGHT_DECAY = 5e-4f;
const bool CIFAR100_CNN_USE_BATCHNORM = true;
const bool CIFAR100_CNN_USE_DATA_AUGMENTATION = true;
const int CIFAR100_CNN_AUGMENT_PAD = 4;

// Practical defaults: CNN training is CPU-heavy; keep the default run small to iterate faster.
// Increase these (or set to <=0) for full-dataset training.
const int CIFAR100_CNN_MAX_TRAIN_SAMPLES = 50000;
const int CIFAR100_CNN_MAX_TEST_SAMPLES = 10000;
