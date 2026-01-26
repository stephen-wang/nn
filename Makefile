CXX = g++
GTEST_VERSION = 1.17.0
GTEST_LIB_PATH = /opt/homebrew/Cellar/googletest/$(GTEST_VERSION)/lib
GETST_LIB_INC = /opt/homebrew/Cellar/googletest/$(GTEST_VERSION)/include
GTEST_LIBS = -lgtest -lgtest_main
TEST_DIR = test
SRC_DIR = src
INC_DIR = include
CXXFLAGS = -std=c++17 -Wall -g -I$(INC_DIR)
TESTFLAGS =  -I$(GETST_LIB_INC) -L$(GTEST_LIB_PATH) $(GTEST_LIBS) -pthread
TARGET = main
TEST_TARGET = nn_test
COVERAGE_TARGET = nn_test_cov
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)
COVERAGE_FLAGS = -O0 --coverage
COVERAGE_TESTFLAGS = $(TESTFLAGS)
COV_OBJ_DIR = build/coverage

#all: (TARGET $(TEST_TARGET) clean
all: $(TARGET) clean

$(TARGET): $(SRC_DIR)/*.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

NON_MAIN_SRCS = $(filter-out $(SRC_DIR)/main.cpp,$(SRC_FILES))
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
