#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CLANG_FORMAT_BIN="${CLANG_FORMAT:-clang-format}"
CLANG_TIDY_BIN="${CLANG_TIDY:-clang-tidy}"
BEAR_BIN="${BEAR:-bear}"

# Only lint our code (exclude vendored code and data).
FILES=$(git ls-files '*.cpp' '*.h' \
  ':!:third_party/**' \
  ':!:build/**' \
  ':!:coverage/**' \
  ':!:mnist/**' || true)

if [[ -z "$FILES" ]]; then
  echo "No source files found to lint."
  exit 0
fi

echo "== clang-format (check) =="
"$CLANG_FORMAT_BIN" --version
# --dry-run + -Werror fails if formatting would change.
"$CLANG_FORMAT_BIN" --dry-run -Werror $FILES

echo "== clang-tidy =="
"$CLANG_TIDY_BIN" --version

TIDY_EXTRA_ARGS=()
TIDY_EXTRA_ARGS+=("--extra-arg=-w")
if [[ "$(uname -s)" == "Darwin" ]] && command -v xcrun >/dev/null 2>&1; then
  SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
  if [[ -n "${SDKROOT}" ]]; then
    # Homebrew LLVM tools may not automatically locate macOS SDK headers.
    TIDY_EXTRA_ARGS+=("--extra-arg=-isysroot" "--extra-arg=${SDKROOT}")
  fi
fi

# clang-tidy works best with a compilation database. Generate one using bear.
if [[ ! -f compile_commands.json ]]; then
  if ! command -v "$BEAR_BIN" >/dev/null 2>&1; then
    echo "ERROR: compile_commands.json missing and 'bear' not found."
    echo "Install bear, or generate compile_commands.json another way."
    exit 2
  fi

  # Build the main target (no googletest/glfw dependencies) to capture compile commands.
  "$BEAR_BIN" -- make -B main
fi

SRC_FILES=$(git ls-files 'src/*.cpp' ':!:src/gui_main.cpp' || true)
if [[ -z "$SRC_FILES" ]]; then
  echo "No src/*.cpp files found for clang-tidy."
  exit 0
fi

# Run clang-tidy over translation units; headers are analyzed via #includes.
"$CLANG_TIDY_BIN" --quiet -p . "${TIDY_EXTRA_ARGS[@]}" $SRC_FILES 2>&1 \
  | sed -E "/\\[[0-9]+\\/[0-9]+\\] Processing file/d; /warnings generated\\./d"

echo "Lint OK"
