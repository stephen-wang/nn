# Neural Network (C++)

A small C++17 implementation of a fully connected neural network trained on MNIST, with sigmoid hidden layers and a softmax output. Includes basic utilities for loading MNIST data, normalization, and a simple training loop.

## Project layout

- `src/` — core implementation (`NeuralNetwork`, `NNLayer`, `NNMatrix`, utils)
- `include/` — public headers
- `mnist/` — MNIST idx data files (train/test images & labels)
- `test/` — unit tests (GoogleTest)
- `Makefile` — build rules

## Build

The default build target compiles the training executable:

```zsh
make
```

This produces `./main`.

## GUI (nn_gui)

The GUI target visualizes training progress and the network topology.

### Dependencies

- GLFW (Homebrew path assumed in the Makefile)
- OpenGL (macOS frameworks)
- ImGui (vendored in `third_party/imgui`)

### Build

```zsh
make nn_gui
```

You can also print the build details from the Makefile:

```zsh
make nn_gui_info
```

### Run

```zsh
./nn_gui
```

You can limit/continue training to an absolute epoch with checkpoint resume:

```zsh
./nn_gui --maxEpoch 20 --model dnn
./nn_gui --maxEpoch 30 --model cnn
```

The GUI build uses the same entry point as the CLI build. You can force CLI mode with:

```zsh
./nn_gui --cli
```

## Run

```zsh
./main
```

CLI also supports absolute max-epoch resume training:

```zsh
./nn --model dnn --maxEpoch 20
./nn --model cnn --maxEpoch 30
```

Behavior of `--maxEpoch`:

- Training checkpoint is automatically loaded before training starts.
- Training continues from the last saved epoch up to `maxEpoch`.
- Updated checkpoint is automatically saved after training ends.

The program expects MNIST files in `mnist/`:

- `mnist/train-images-idx3-ubyte`
- `mnist/train-labels-idx1-ubyte`
- `mnist/t10k-images-idx3-ubyte`
- `mnist/t10k-labels-idx1-ubyte`

## CNN model persistence

`CNN` supports binary serialization of model parameters and optimizer state:

- Save: `cnn.save("model.bin")`
- Load: `cnn.load("model.bin")`

Notes:

- `load` restores parameters into an already-constructed `CNN`.
- The target `CNN` must use the same architecture (same layers/shapes/order) as the saved file.

## DNN model persistence

`DNN` also supports binary serialization of model parameters and optimizer state:

- Save: `dnn.save("model.bin")`
- Load: `dnn.load("model.bin")`

Notes:

- `load` restores parameters into an already-constructed `DNN`.
- The target `DNN` must use the same architecture (same layers/shapes/order) as the saved file.

## Tests (GoogleTest)

The `Makefile` includes a test target that links against GoogleTest installed via Homebrew.

```zsh
make nn_test
./nn_test
```

### Troubleshooting tests

- The `Makefile` expects GoogleTest in `/opt/homebrew/Cellar/googletest/1.17.0/`.
- If your version differs, update `GTEST_VERSION` or override paths in the `Makefile`.

## Lint / Format

Recommended baseline:

- `clang-format` for consistent formatting
- `clang-tidy` for static analysis (readability/bugprone/performance)

This repo includes configs:

- `.clang-format`
- `.clang-tidy`

### Run locally

```zsh
make lint
```

Targets are also available individually:

```zsh
make format-check
make tidy
```

### CI

GitHub Actions workflow: `.github/workflows/lint.yml`

## Notes

- Inputs are normalized to `[0, 1]` in `NNUtils::normalizeMnistData`.
- Labels are one-hot encoded in `NNUtils::read_mnist_labels`.
- Hidden layers use sigmoid activation; the output layer uses softmax.
