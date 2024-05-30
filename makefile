# Compiler and compiler flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++20 -O2 -I ./eigen/Eigen
NVCC_FLAGS := -O3 --std=c++20
LD_FLAGS := -lcudart

# Target executable
TARGET := mat-add

# Source files
CPP_SRCS := test.cpp
CUDA_SRCS := 

# Object files
CPP_OBJS := $(CPP_SRCS:.cpp=.o)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
OBJS := $(CPP_OBJS) $(CUDA_OBJS)

# Default target
all: $(TARGET)

# Link the target executable
$(TARGET): $(OBJS)
	$(NVCC) -o $@ $^ $(LD_FLAGS)

# Compile C++ source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA source files into object files
# %.o: %.cu
# 	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

# Clean up build files
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
