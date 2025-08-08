#!/bin/bash

# Comprehensive Test Script for RoboDSL Generated CUDA Files
# Run this on your CUDA-enabled Linux machine

set -e  # Exit on any error

echo "🚀 Starting Comprehensive CUDA Build Test"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
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

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run this script from the comprehensive_test directory."
    exit 1
fi

# Check CUDA availability
print_status "Checking CUDA installation..."
if ! command -v nvcc &> /dev/null; then
    print_error "nvcc not found. Please install CUDA toolkit."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
print_success "CUDA version: $CUDA_VERSION"

# Check GPU availability
print_status "Checking GPU availability..."
if ! command -v nvidia-smi &> /dev/null; then
    print_warning "nvidia-smi not found. GPU testing will be limited."
else
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    print_success "Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits
fi

# Check ROS2 installation
print_status "Checking ROS2 installation..."
if ! command -v ros2 &> /dev/null; then
    print_warning "ROS2 not found. Some tests will be skipped."
    ROS2_AVAILABLE=false
else
    ROS2_VERSION=$(ros2 --version | head -n1)
    print_success "ROS2 version: $ROS2_VERSION"
    ROS2_AVAILABLE=true
fi

# Check required packages
print_status "Checking required packages..."
MISSING_PACKAGES=""

# Check for ament_cmake
if ! pkg-config --exists ament_cmake; then
    MISSING_PACKAGES="$MISSING_PACKAGES ament_cmake"
fi

# Check for rclcpp
if ! pkg-config --exists rclcpp; then
    MISSING_PACKAGES="$MISSING_PACKAGES rclcpp"
fi

# Check for std_msgs
if ! pkg-config --exists std_msgs; then
    MISSING_PACKAGES="$MISSING_PACKAGES std_msgs"
fi

# Check for sensor_msgs
if ! pkg-config --exists sensor_msgs; then
    MISSING_PACKAGES="$MISSING_PACKAGES sensor_msgs"
fi

if [ -n "$MISSING_PACKAGES" ]; then
    print_warning "Missing packages: $MISSING_PACKAGES"
    print_status "Install with: sudo apt install $MISSING_PACKAGES"
else
    print_success "All required packages found"
fi

# Create build directory
print_status "Creating build directory..."
rm -rf build
mkdir -p build
cd build

# Test RoboDSL generation
print_status "Testing RoboDSL generation..."
if command -v robodsl &> /dev/null; then
    # Test generating from the .robodsl file
    robodsl generate comprehensive_test.robodsl --output-dir ./generated_test
    
    if [ $? -eq 0 ]; then
        print_success "RoboDSL generation successful"
    else
        print_error "RoboDSL generation failed"
        exit 1
    fi
else
    print_warning "robodsl command not found, skipping generation test"
    print_status "Install RoboDSL with: pip install -e ."
fi

# Configure with CMake (ROS2 colcon style)
print_status "Configuring with CMake (ROS2 style)..."
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=all

if [ $? -eq 0 ]; then
    print_success "CMake configuration successful"
else
    print_error "CMake configuration failed"
    exit 1
fi

# Build the project
print_status "Building project..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    print_success "CMake build successful"
else
    print_error "CMake build failed"
    exit 1
fi

# Test ROS2 colcon build if available
print_status "Testing ROS2 colcon build..."
if command -v colcon &> /dev/null; then
    cd ..
    
    # Create a temporary ROS2 workspace for testing
    mkdir -p test_workspace/src
    cp -r . test_workspace/src/comprehensive_test/
    cd test_workspace
    
    # Build with colcon
    colcon build --packages-select comprehensive_test
    
    if [ $? -eq 0 ]; then
        print_success "ROS2 colcon build successful"
    else
        print_warning "ROS2 colcon build failed (this is optional)"
    fi
    
    cd ..
else
    print_warning "colcon not found, skipping ROS2 build test"
    print_status "Install colcon with: pip install colcon-common-extensions"
fi

# Check generated files
print_status "Checking generated files..."
cd ..

# Check CUDA files
CUDA_FILES=$(find src/cuda -name "*.cu" 2>/dev/null | wc -l)
CUDA_HEADERS=$(find include/cuda -name "*.cuh" 2>/dev/null | wc -l)
CUDA_WRAPPERS=$(find include/cuda -name "*_wrapper.hpp" 2>/dev/null | wc -l)

print_success "Found $CUDA_FILES CUDA source files"
print_success "Found $CUDA_HEADERS CUDA header files"
print_success "Found $CUDA_WRAPPERS CUDA wrapper files"

# Check C++ node files
CPP_NODES=$(find src/nodes -name "*.cpp" 2>/dev/null | wc -l)
HPP_NODES=$(find include/nodes -name "*.hpp" 2>/dev/null | wc -l)

print_success "Found $CPP_NODES C++ node source files"
print_success "Found $HPP_NODES C++ node header files"

# Check ONNX files
ONNX_FILES=$(find src/onnx -name "*.cpp" 2>/dev/null | wc -l)
ONNX_HEADERS=$(find include/onnx -name "*.hpp" 2>/dev/null | wc -l)

print_success "Found $ONNX_FILES ONNX source files"
print_success "Found $ONNX_HEADERS ONNX header files"

# Check pipeline files
PIPELINE_FILES=$(find src/pipelines -name "*.cpp" 2>/dev/null | wc -l)
PIPELINE_HEADERS=$(find include/pipelines -name "*.hpp" 2>/dev/null | wc -l)

print_success "Found $PIPELINE_FILES pipeline source files"
print_success "Found $PIPELINE_HEADERS pipeline header files"

# Check utilities
UTILS_FILES=$(find src/utils -name "*.cpp" 2>/dev/null | wc -l)
UTILS_HEADERS=$(find include/utils -name "*.hpp" 2>/dev/null | wc -l)

print_success "Found $UTILS_FILES utility source files"
print_success "Found $UTILS_HEADERS utility header files"

echo ""
print_success "Build test completed successfully!"
echo ""
print_status "Next steps:"
echo "  1. Run: ./test_cuda_runtime.sh"
echo "  2. Run: ./test_ros2_integration.sh (if ROS2 is available)"
echo "  3. Run: ./test_performance.sh"
echo ""

exit 0
