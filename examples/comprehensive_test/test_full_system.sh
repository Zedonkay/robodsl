#!/bin/bash

# Full System Test Script for RoboDSL Generated Files
# Runs all tests in sequence and provides a comprehensive report

set -e

echo "🚀 Starting Full System Test Suite"
echo "=================================="

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

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TESTS_SKIPPED=0

# Function to run a test and track results
run_test() {
    local test_name="$1"
    local test_script="$2"
    local description="$3"
    
    echo ""
    echo "=========================================="
    print_status "Running: $test_name"
    print_status "Description: $description"
    echo "=========================================="
    
    if [ -f "$test_script" ]; then
        if bash "$test_script"; then
            print_success "$test_name PASSED"
            ((TESTS_PASSED++))
        else
            print_error "$test_name FAILED"
            ((TESTS_FAILED++))
        fi
    else
        print_warning "$test_name SKIPPED (script not found)"
        ((TESTS_SKIPPED++))
    fi
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run this script from the comprehensive_test directory."
    exit 1
fi

# Make test scripts executable
chmod +x test_build.sh test_cuda_runtime.sh test_ros2_integration.sh test_performance.sh 2>/dev/null || true

# Run all tests
print_status "Starting comprehensive test suite..."

# 1. RoboDSL CLI Test
run_test "RoboDSL CLI Test" "test_robodsl_cli.sh" "Tests RoboDSL command-line interface and generation"

# 2. Build Test
run_test "Build Test" "test_build.sh" "Tests CMake configuration, compilation, and file generation"

# 2. CUDA Runtime Test
run_test "CUDA Runtime Test" "test_cuda_runtime.sh" "Tests CUDA kernel compilation and basic functionality"

# 3. ROS2 Integration Test
run_test "ROS2 Integration Test" "test_ros2_integration.sh" "Tests ROS2 node compilation and integration"

# 4. Performance Test
run_test "Performance Test" "test_performance.sh" "Benchmarks CUDA kernels and memory bandwidth"

# Generate test report
echo ""
echo "=========================================="
echo "📊 TEST SUMMARY REPORT"
echo "=========================================="
echo "Tests Passed:  $TESTS_PASSED"
echo "Tests Failed:  $TESTS_FAILED"
echo "Tests Skipped: $TESTS_SKIPPED"
echo "Total Tests:   $((TESTS_PASSED + TESTS_FAILED + TESTS_SKIPPED))"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    print_success "🎉 ALL TESTS PASSED! Your RoboDSL generated files are working correctly."
    echo ""
    print_status "Your system is ready for production use!"
    echo ""
    print_status "Next steps:"
    echo "  1. Deploy to your target system"
    echo "  2. Run with actual data: ros2 launch launch/main_launch.py"
    echo "  3. Monitor performance in production"
    echo ""
else
    print_error "❌ SOME TESTS FAILED. Please review the errors above."
    echo ""
    print_status "Troubleshooting tips:"
    echo "  1. Check CUDA installation: nvidia-smi && nvcc --version"
    echo "  2. Check ROS2 installation: ros2 --version"
    echo "  3. Install missing packages: sudo apt install <package-name>"
    echo "  4. Check system requirements in README.md"
    echo ""
fi

# System information
echo "=========================================="
echo "💻 SYSTEM INFORMATION"
echo "=========================================="

# OS information
echo "OS: $(lsb_release -d | cut -f2 2>/dev/null || uname -a)"

# CUDA information
if command -v nvcc &> /dev/null; then
    echo "CUDA: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)"
else
    echo "CUDA: Not installed"
fi

# GPU information
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)"
    echo "GPU Memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1) MB"
else
    echo "GPU: Not detected"
fi

# ROS2 information
if command -v ros2 &> /dev/null; then
    echo "ROS2: $(ros2 --version | head -n1)"
else
    echo "ROS2: Not installed"
fi

# Compiler information
if command -v g++ &> /dev/null; then
    echo "GCC: $(g++ --version | head -n1)"
fi

if command -v nvcc &> /dev/null; then
    echo "NVCC: $(nvcc --version | head -n1)"
fi

echo ""

# File count summary
echo "=========================================="
echo "📁 GENERATED FILES SUMMARY"
echo "=========================================="

CUDA_FILES=$(find src/cuda -name "*.cu" 2>/dev/null | wc -l)
CUDA_HEADERS=$(find include/cuda -name "*.cuh" 2>/dev/null | wc -l)
CUDA_WRAPPERS=$(find include/cuda -name "*_wrapper.hpp" 2>/dev/null | wc -l)
CPP_NODES=$(find src/nodes -name "*.cpp" 2>/dev/null | wc -l)
HPP_NODES=$(find include/nodes -name "*.hpp" 2>/dev/null | wc -l)
ONNX_FILES=$(find src/onnx -name "*.cpp" 2>/dev/null | wc -l)
ONNX_HEADERS=$(find include/onnx -name "*.hpp" 2>/dev/null | wc -l)
PIPELINE_FILES=$(find src/pipelines -name "*.cpp" 2>/dev/null | wc -l)
PIPELINE_HEADERS=$(find include/pipelines -name "*.hpp" 2>/dev/null | wc -l)
UTILS_FILES=$(find src/utils -name "*.cpp" 2>/dev/null | wc -l)
UTILS_HEADERS=$(find include/utils -name "*.hpp" 2>/dev/null | wc -l)

echo "CUDA Source Files:     $CUDA_FILES"
echo "CUDA Header Files:     $CUDA_HEADERS"
echo "CUDA Wrapper Files:    $CUDA_WRAPPERS"
echo "C++ Node Source Files: $CPP_NODES"
echo "C++ Node Header Files: $HPP_NODES"
echo "ONNX Source Files:     $ONNX_FILES"
echo "ONNX Header Files:     $ONNX_HEADERS"
echo "Pipeline Source Files: $PIPELINE_FILES"
echo "Pipeline Header Files: $PIPELINE_HEADERS"
echo "Utility Source Files:  $UTILS_FILES"
echo "Utility Header Files:  $UTILS_HEADERS"

TOTAL_FILES=$((CUDA_FILES + CUDA_HEADERS + CUDA_WRAPPERS + CPP_NODES + HPP_NODES + ONNX_FILES + ONNX_HEADERS + PIPELINE_FILES + PIPELINE_HEADERS + UTILS_FILES + UTILS_HEADERS))
echo "Total Generated Files: $TOTAL_FILES"

echo ""

# Exit with appropriate code
if [ $TESTS_FAILED -eq 0 ]; then
    exit 0
else
    exit 1
fi
