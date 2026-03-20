CXX = g++
GLFW_LIB_INC = /opt/homebrew/Cellar/glfw/3.4/include
GLFW_LIB = /opt/homebrew/Cellar/glfw/3.4/lib
GTEST_VERSION = 1.17.0
GTEST_LIB_PATH = /opt/homebrew/Cellar/googletest/$(GTEST_VERSION)/lib
GETST_LIB_INC = /opt/homebrew/Cellar/googletest/$(GTEST_VERSION)/include
GTEST_LIBS = -lgtest -lgtest_main
TEST_DIR = test
SRC_DIR = src
INC_DIR = include
OMP ?= 0

CXXFLAGS = -std=c++17 -Wall -O3 -g -DNDEBUG $(ARCH_FLAGS) -I$(INC_DIR) -Ithird_party

# Release flags: focus on CPU throughput (training is compute-heavy)
# - Native CPU tuning is compiler/arch dependent:
#   - Apple clang on arm64: prefers -mcpu=native
#   - GCC/clang on x86_64: -march=native (and -mtune=native for GCC)
UNAME_M := $(shell uname -m)
CXX_VERSION := $(shell $(CXX) --version 2>/dev/null)
ARCH_FLAGS :=
ifneq (,$(findstring clang,$(CXX_VERSION)))
ifeq ($(UNAME_M),arm64)
ARCH_FLAGS := -mcpu=native
else
ARCH_FLAGS := -march=native
endif
else
ARCH_FLAGS := -march=native -mtune=native
endif

# -flto: enable link-time optimization across translation units
# -fomit-frame-pointer: can help optimizer in tight loops
RELEASE_CXXFLAGS = -std=c++17 -Wall -O3 -DNDEBUG $(ARCH_FLAGS) -flto -fomit-frame-pointer -I$(INC_DIR) -Ithird_party
LDFLAGS =

# Optional: OpenMP (multi-core speedups for convolution-heavy training)
# macOS: `brew install libomp`, then build with `make OMP=1`.
ifeq ($(OMP),1)
LIBOMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
OMP_DEFS := -DNN_ENABLE_OMP
ifneq (,$(findstring clang,$(CXX_VERSION)))
OMP_COMPILE_FLAGS := -Xpreprocessor -fopenmp
OMP_LINK_FLAGS := -lomp
else
OMP_COMPILE_FLAGS := -fopenmp
OMP_LINK_FLAGS := -fopenmp
endif
ifneq ($(strip $(LIBOMP_PREFIX)),)
ifneq ($(wildcard $(LIBOMP_PREFIX)/include/omp.h),)
OMP_COMPILE_FLAGS += -I$(LIBOMP_PREFIX)/include
OMP_LINK_FLAGS += -L$(LIBOMP_PREFIX)/lib
else
$(error OpenMP requested (OMP=1) but '$(LIBOMP_PREFIX)/include/omp.h' was not found. Run: brew install libomp)
endif
else
$(error OpenMP requested (OMP=1) but Homebrew libomp was not found. Run: brew install libomp)
endif

CXXFLAGS += $(OMP_DEFS) $(OMP_COMPILE_FLAGS)
RELEASE_CXXFLAGS += $(OMP_DEFS) $(OMP_COMPILE_FLAGS)
LDFLAGS += $(OMP_LINK_FLAGS)
endif
TESTFLAGS =  -I$(GETST_LIB_INC) -L$(GTEST_LIB_PATH) $(GTEST_LIBS) -pthread
TARGET = nn
TEST_TARGET = nn_test
COVERAGE_TARGET = nn_test_cov
GUI_TARGET = nn_gui
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_SRCS = $(filter-out $(SRC_DIR)/DNNGuiUtils.cpp $(SRC_DIR)/CNNGuiUtils.cpp $(SRC_DIR)/NNGuiUtils.cpp,$(SRC_FILES))
GUI_SRCS = $(SRC_FILES)
IMGUI_DIR = third_party/imgui
IMGUI_BACKENDS = $(IMGUI_DIR)/backends
IMGUI_SRCS = \
	$(IMGUI_DIR)/imgui.cpp \
	$(IMGUI_DIR)/imgui_draw.cpp \
	$(IMGUI_DIR)/imgui_tables.cpp \
	$(IMGUI_DIR)/imgui_widgets.cpp \
	$(IMGUI_BACKENDS)/imgui_impl_glfw.cpp \
	$(IMGUI_BACKENDS)/imgui_impl_opengl3.cpp
GUI_LIBS = -L$(GLFW_LIB) -lglfw -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
GUI_DEPS = glfw imgui opengl
GUI_CXXFLAGS = $(CXXFLAGS) -DNN_ENABLE_GUI -DNN_DEFAULT_GUI
GUI_BUILD_CMD = $(CXX) $(GUI_CXXFLAGS) -I$(GLFW_LIB_INC) -I$(IMGUI_DIR) -I$(IMGUI_BACKENDS) $(GUI_LIBS) -o $(GUI_TARGET) $(GUI_SRCS) $(IMGUI_SRCS)
COVERAGE_FLAGS = -O0 --coverage
COVERAGE_TESTFLAGS = $(TESTFLAGS)
COV_OBJ_DIR = build/coverage

# all: default to the fastest build (LTO + native tuning) since training is throughput-bound.
# Use `make nn` if you explicitly want the non-LTO build.
all: nn_release clean

$(TARGET): $(MAIN_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

nn_release: $(MAIN_SRCS)
	$(CXX) $(RELEASE_CXXFLAGS) -o $(TARGET) $^ $(LDFLAGS)

$(GUI_TARGET): $(GUI_SRCS) $(IMGUI_SRCS) 
	$(CXX) $(GUI_CXXFLAGS) -I$(GLFW_LIB_INC) -I$(IMGUI_DIR) -I$(IMGUI_BACKENDS) $(GUI_LIBS) -o $@ $^ $(LDFLAGS)

nn_gui_info:
	@echo "Target: $(GUI_TARGET)"
	@echo "Dependencies: $(GUI_DEPS)"
	@echo "GLFW include: $(GLFW_LIB_INC)"
	@echo "GLFW lib: $(GLFW_LIB)"
	@echo "ImGui dir: $(IMGUI_DIR)"
	@echo "Build command: $(GUI_BUILD_CMD)"

NON_MAIN_SRCS = $(filter-out $(SRC_DIR)/main.cpp $(SRC_DIR)/DNNGuiUtils.cpp $(SRC_DIR)/CNNGuiUtils.cpp $(SRC_DIR)/NNGuiUtils.cpp,$(SRC_FILES))
COV_SRCS = $(NON_MAIN_SRCS)
COV_TEST_SRCS = $(wildcard $(TEST_DIR)/*.cpp)
COV_OBJS = $(patsubst $(SRC_DIR)/%.cpp,$(COV_OBJ_DIR)/%.o,$(COV_SRCS))
COV_TEST_OBJS = $(patsubst $(TEST_DIR)/%.cpp,$(COV_OBJ_DIR)/test_%.o,$(COV_TEST_SRCS))
$(TEST_TARGET): $(TEST_DIR)/*.cpp $(NON_MAIN_SRCS)
	@echo "NON Main srcs: $(NON_MAIN_SRCS)"
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $@ $^

$(COV_OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(COV_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(COVERAGE_FLAGS) -c $< -o $@

$(COV_OBJ_DIR)/test_%.o: $(TEST_DIR)/%.cpp
	@mkdir -p $(COV_OBJ_DIR)
	$(CXX) $(CXXFLAGS) $(COVERAGE_FLAGS) -I$(GETST_LIB_INC) -pthread -c $< -o $@

$(COVERAGE_TARGET): $(COV_OBJS) $(COV_TEST_OBJS)
	$(CXX) $(CXXFLAGS) $(COVERAGE_FLAGS) -o $@ $^ $(TESTFLAGS)

coverage: clean_coverage $(COVERAGE_TARGET)
	./$(COVERAGE_TARGET)
	gcovr -r . --object-directory $(COV_OBJ_DIR) --exclude ".*test/.*" --exclude ".*mnist/.*" --print-summary

coverage_html: clean_coverage $(COVERAGE_TARGET)
	./$(COVERAGE_TARGET)
	@mkdir -p coverage
	gcovr -r . --object-directory $(COV_OBJ_DIR) --exclude ".*test/.*" --exclude ".*mnist/.*" --html-details -o coverage/index.html


clean:
	rm -rf *.o *dSYM
clean_all:
	rm -rf *.o $(TEST_TARGET) $(TARGET) *dSYM
clean_coverage:
	rm -rf *.gcda *.gcno coverage $(COV_OBJ_DIR)

# -------- Lint / Format --------
CLANG_FORMAT ?= clang-format
CLANG_TIDY ?= clang-tidy
BEAR ?= bear
CLANG_TIDY_EXTRA_ARGS ?= --extra-arg=-w
CLANG_TIDY_PLATFORM_ARGS ?=

# Homebrew LLVM clang-tidy may not automatically find macOS SDK headers.
# If we're on macOS and xcrun is available, pass the SDK sysroot.
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
SDKROOT := $(shell xcrun --sdk macosx --show-sdk-path 2>/dev/null)
ifneq ($(strip $(SDKROOT)),)
CLANG_TIDY_PLATFORM_ARGS := --extra-arg=-isysroot --extra-arg=$(SDKROOT)
endif
endif

LINT_FILES = $(shell git ls-files '*.cpp' '*.h' ':!:third_party/**' ':!:build/**' ':!:coverage/**' ':!:mnist/**')
	# clang-tidy should only run on files present in the compilation database generated by `bear -- make main`.
	# non-GUI builds exclude GUI sources, so exclude them here as well.
TIDY_SRCS = $(filter-out $(SRC_DIR)/DNNGuiUtils.cpp $(SRC_DIR)/CNNGuiUtils.cpp,$(MAIN_SRCS))

define REQUIRE_TOOL
	@command -v $(1) >/dev/null 2>&1 || { \
		echo "ERROR: $(1) not found."; \
		echo "Install (macOS): brew install llvm bear"; \
		echo "  then set PATH or run e.g. CLANG_FORMAT=/opt/homebrew/opt/llvm/bin/clang-format"; \
		echo "Install (Ubuntu): sudo apt-get install clang-format clang-tidy bear"; \
		exit 2; \
	}
endef

format:
	$(call REQUIRE_TOOL,$(CLANG_FORMAT))
	$(CLANG_FORMAT) -i $(LINT_FILES)

format-check:
	$(call REQUIRE_TOOL,$(CLANG_FORMAT))
	$(CLANG_FORMAT) --dry-run -Werror $(LINT_FILES)

tidy:
	$(call REQUIRE_TOOL,$(BEAR))
	$(call REQUIRE_TOOL,$(CLANG_TIDY))
	$(BEAR) -- make -B main
	@bash -o pipefail -c '$(CLANG_TIDY) --quiet -p . $(CLANG_TIDY_EXTRA_ARGS) $(CLANG_TIDY_PLATFORM_ARGS) $(TIDY_SRCS) 2>&1 | sed -E "/\\[[0-9]+\\/[0-9]+\\] Processing file/d; /warnings generated\\./d"'

lint: format-check tidy

.PHONY: nn_gui_info format format-check tidy lint
