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

## Run

```zsh
./main
```

The program expects MNIST files in `mnist/`:

- `mnist/train-images-idx3-ubyte`
- `mnist/train-labels-idx1-ubyte`
- `mnist/t10k-images-idx3-ubyte`
- `mnist/t10k-labels-idx1-ubyte`

## Tests (GoogleTest)

The `Makefile` includes a test target that links against GoogleTest installed via Homebrew.

```zsh
make nn_test
./nn_test
```

## Coverage

Coverage uses gcovr. Install it first (e.g. `pipx install gcovr` or `pip install gcovr`).

```zsh
make coverage
```

For an HTML report:

```zsh
make coverage_html
open coverage/index.html
```

### Troubleshooting tests

- The `Makefile` expects GoogleTest in `/opt/homebrew/Cellar/googletest/1.17.0/`.
- If your version differs, update `GTEST_VERSION` or override paths in the `Makefile`.

## Notes

- Inputs are normalized to `[0, 1]` in `NNUtils::normalizeMnistData`.
- Labels are one-hot encoded in `NNUtils::read_mnist_labels`.
- Hidden layers use sigmoid activation; the output layer uses softmax.
