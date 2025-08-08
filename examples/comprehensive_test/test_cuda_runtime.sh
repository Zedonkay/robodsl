#!/bin/bash

# CUDA Runtime Test Script for RoboDSL Generated CUDA Files
# Tests the actual CUDA kernels and wrappers

set -e

echo "🔧 Starting CUDA Runtime Test"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if build exists
if [ ! -d "build" ]; then
    print_error "Build directory not found. Run ./test_build.sh first."
    exit 1
fi

cd build

# Create a simple CUDA test program
print_status "Creating CUDA runtime test program..."

cat > test_cuda_runtime.cpp << 'EOF'
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Include our generated CUDA wrappers
#include "../include/cuda/vector_add_wrapper.hpp"
#include "../include/cuda/matrix_multiply_wrapper.hpp"
#include "../include/cuda/image_filter_wrapper.hpp"

int main() {
    std::cout << "Testing CUDA Runtime..." << std::endl;
    
    // Test vector_add wrapper
    try {
        std::cout << "Testing vector_add wrapper..." << std::endl;
        auto vector_wrapper = robodsl::createvector_addWrapper();
        
        if (!vector_wrapper->initialize(0)) {
            std::cerr << "Failed to initialize vector_add wrapper" << std::endl;
            return 1;
        }
        
        // Test data
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        std::vector<float> output_data;
        
        if (vector_wrapper->processData(input_data, output_data)) {
            std::cout << "vector_add test PASSED" << std::endl;
            std::cout << "Input: ";
            for (float f : input_data) std::cout << f << " ";
            std::cout << std::endl;
            std::cout << "Output: ";
            for (float f : output_data) std::cout << f << " ";
            std::cout << std::endl;
        } else {
            std::cerr << "vector_add test FAILED: " << vector_wrapper->getLastError() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "vector_add test exception: " << e.what() << std::endl;
        return 1;
    }
    
    // Test matrix_multiply wrapper
    try {
        std::cout << "\nTesting matrix_multiply wrapper..." << std::endl;
        auto matrix_wrapper = robodsl::creatematrix_multiplyWrapper();
        
        if (!matrix_wrapper->initialize(0)) {
            std::cerr << "Failed to initialize matrix_multiply wrapper" << std::endl;
            return 1;
        }
        
        // Test data (small matrix)
        std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f}; // 2x2 matrix
        std::vector<float> output_data;
        
        if (matrix_wrapper->processData(input_data, output_data)) {
            std::cout << "matrix_multiply test PASSED" << std::endl;
        } else {
            std::cerr << "matrix_multiply test FAILED: " << matrix_wrapper->getLastError() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "matrix_multiply test exception: " << e.what() << std::endl;
        return 1;
    }
    
    // Test image_filter wrapper
    try {
        std::cout << "\nTesting image_filter wrapper..." << std::endl;
        auto image_wrapper = robodsl::createimage_filterWrapper();
        
        if (!image_wrapper->initialize(0)) {
            std::cerr << "Failed to initialize image_filter wrapper" << std::endl;
            return 1;
        }
        
        // Test data (small image)
        std::vector<float> input_data(16, 1.0f); // 4x4 image
        std::vector<float> output_data;
        
        if (image_wrapper->processData(input_data, output_data)) {
            std::cout << "image_filter test PASSED" << std::endl;
        } else {
            std::cerr << "image_filter test FAILED: " << image_wrapper->getLastError() << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "image_filter test exception: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "\n🎉 All CUDA runtime tests PASSED!" << std::endl;
    return 0;
}
EOF

# Compile the test program
print_status "Compiling CUDA runtime test..."
nvcc -std=c++17 -I../include -I/usr/local/cuda/include \
     -L/usr/local/cuda/lib64 -lcudart \
     test_cuda_runtime.cpp -o test_cuda_runtime

if [ $? -eq 0 ]; then
    print_success "CUDA runtime test compiled successfully"
else
    print_error "CUDA runtime test compilation failed"
    exit 1
fi

# Run the test
print_status "Running CUDA runtime test..."
./test_cuda_runtime

if [ $? -eq 0 ]; then
    print_success "CUDA runtime test PASSED"
else
    print_error "CUDA runtime test FAILED"
    exit 1
fi

# Test CUDA memory allocation
print_status "Testing CUDA memory allocation..."
cat > test_cuda_memory.cpp << 'EOF'
#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout << "Testing CUDA memory allocation..." << std::endl;
    
    // Test device memory allocation
    float* d_data;
    size_t size = 1024 * sizeof(float);
    
    cudaError_t result = cudaMalloc(&d_data, size);
    if (result != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(result) << std::endl;
        return 1;
    }
    
    // Test host to device copy
    std::vector<float> h_data(1024, 1.0f);
    result = cudaMemcpy(d_data, h_data.data(), size, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(result) << std::endl;
        cudaFree(d_data);
        return 1;
    }
    
    // Test device to host copy
    std::vector<float> h_result(1024);
    result = cudaMemcpy(h_result.data(), d_data, size, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        std::cerr << "cudaMemcpy D2H failed: " << cudaGetErrorString(result) << std::endl;
        cudaFree(d_data);
        return 1;
    }
    
    // Cleanup
    cudaFree(d_data);
    
    std::cout << "CUDA memory allocation test PASSED" << std::endl;
    return 0;
}
EOF

nvcc test_cuda_memory.cpp -o test_cuda_memory
./test_cuda_memory

if [ $? -eq 0 ]; then
    print_success "CUDA memory allocation test PASSED"
else
    print_error "CUDA memory allocation test FAILED"
    exit 1
fi

# Test CUDA kernel compilation
print_status "Testing CUDA kernel compilation..."
cat > test_kernel_compilation.cu << 'EOF'
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// Simple test kernel
__global__ void test_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    std::cout << "Testing CUDA kernel compilation..." << std::endl;
    
    // This just tests that we can compile and link CUDA kernels
    std::cout << "CUDA kernel compilation test PASSED" << std::endl;
    return 0;
}
EOF

nvcc test_kernel_compilation.cu -o test_kernel_compilation
./test_kernel_compilation

if [ $? -eq 0 ]; then
    print_success "CUDA kernel compilation test PASSED"
else
    print_error "CUDA kernel compilation test FAILED"
    exit 1
fi

cd ..

echo ""
print_success "All CUDA runtime tests completed successfully!"
echo ""
print_status "Next steps:"
echo "  1. Run: ./test_ros2_integration.sh (if ROS2 is available)"
echo "  2. Run: ./test_performance.sh"
echo ""

exit 0
