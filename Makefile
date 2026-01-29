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
CXXFLAGS = -std=c++17 -Wall -g -I$(INC_DIR) -Ithird_party
TESTFLAGS =  -I$(GETST_LIB_INC) -L$(GTEST_LIB_PATH) $(GTEST_LIBS) -pthread
TARGET = main
TEST_TARGET = nn_test
COVERAGE_TARGET = nn_test_cov
GUI_TARGET = nn_gui
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
MAIN_SRCS = $(filter-out $(SRC_DIR)/gui_main.cpp,$(SRC_FILES))
GUI_SRCS = $(filter-out $(SRC_DIR)/main.cpp,$(SRC_FILES))
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
GUI_BUILD_CMD = $(CXX) $(CXXFLAGS) -I$(GLFW_LIB_INC) -I$(IMGUI_DIR) -I$(IMGUI_BACKENDS) $(GUI_LIBS) -o $(GUI_TARGET) $(GUI_SRCS) $(IMGUI_SRCS)
COVERAGE_FLAGS = -O0 --coverage
COVERAGE_TESTFLAGS = $(TESTFLAGS)
COV_OBJ_DIR = build/coverage

#all: (TARGET $(TEST_TARGET) clean
all: $(TARGET) clean

$(TARGET): $(MAIN_SRCS)
	$(CXX) $(CXXFLAGS) -o $@ $^

$(GUI_TARGET): $(GUI_SRCS) $(IMGUI_SRCS) 
	$(CXX) $(CXXFLAGS) -I$(GLFW_LIB_INC) -I$(IMGUI_DIR) -I$(IMGUI_BACKENDS) $(GUI_LIBS) -o $@ $^ 

nn_gui_info:
	@echo "Target: $(GUI_TARGET)"
	@echo "Dependencies: $(GUI_DEPS)"
	@echo "GLFW include: $(GLFW_LIB_INC)"
	@echo "GLFW lib: $(GLFW_LIB)"
	@echo "ImGui dir: $(IMGUI_DIR)"
	@echo "Build command: $(GUI_BUILD_CMD)"

NON_MAIN_SRCS = $(filter-out $(SRC_DIR)/main.cpp $(SRC_DIR)/gui_main.cpp,$(SRC_FILES))
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

.PHONY: nn_gui_info
