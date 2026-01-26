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
SRC_FILES = $(wildcard $(SRC_DIR)/*.cpp)

#all: $(TARGET) $(TEST_TARGET) clean
all: $(TARGET) clean

$(TARGET): $(SRC_DIR)/*.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

#NON_MAIN_SRCS = $(filter-out $(SRC_DIR)/main.cpp,$(SRC_FILES))
$(patsubst %.c,%.o,$(wildcard *.c))
NON_MAIN_SRCS = $(patsubst $(SRC_DIR)/main.cpp,, $(SRC_FILES))
$(TEST_TARGET): $(TEST_DIR)/*.cpp $(NON_MAIN_SRCS)
	@echo "NON Main srcs: $(NON_MAIN_SRCS)"
	$(CXX) $(CXXFLAGS) $(TESTFLAGS) -o $@ $^


clean:
	rm -rf *.o *dSYM
clean_all:
	rm -rf *.o $(TEST_TARGET) $(TARGET) *dSYM
