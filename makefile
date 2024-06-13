# Makefile for CUDA C++ Matrix Library

# Compiler
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -std=c++17 -O0
NVCCFLAGS := -std=c++17 -O0

# Target executable
TARGET := a.out

# Source files
SRCS := 
CU_SRCS := example.cu

# Object files
OBJS := $(SRCS:.cpp=.o)
CU_OBJS := $(CU_SRCS:.cu=.o)

# Rules
all: $(TARGET)

$(TARGET): $(OBJS) $(CU_OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(CU_OBJS) $(TARGET)

.PHONY: all clean
