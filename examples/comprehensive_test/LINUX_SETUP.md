# Linux Setup Guide for RoboDSL Testing

This guide will help you set up and test your RoboDSL generated files on a CUDA-enabled Linux machine.

## 🚀 Quick Start

1. **Copy the `comprehensive_test/` folder to your Linux machine**
2. **Install RoboDSL:**
   ```bash
   # From the RoboDSL source directory
   pip install -e .
   ```
3. **Run the full test suite:**
   ```bash
   cd comprehensive_test/
   ./test_full_system.sh
   ```

## 📋 Prerequisites

### Required Software
- **CUDA Toolkit** (11.0 or later)
- **GCC/G++** (7.0 or later)
- **CMake** (3.16 or later)
- **ROS2** (Humble or Foxy) - Optional but recommended

### Installation Commands

#### Ubuntu/Debian:
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
sudo apt update
sudo apt install cuda-toolkit-11-8

# Install build tools
sudo apt install build-essential cmake pkg-config

# Install ROS2 (optional)
# Follow instructions at: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
```

#### CentOS/RHEL:
```bash
# Install CUDA Toolkit
sudo yum install cuda-toolkit

# Install build tools
sudo yum groupinstall "Development Tools"
sudo yum install cmake3
```

## 🧪 Testing Strategy

### 1. **RoboDSL CLI Test** (`test_robodsl_cli.sh`)
- ✅ Tests RoboDSL command-line interface
- ✅ Tests `robodsl init` command
- ✅ Tests `robodsl generate` command
- ✅ Tests `robodsl create-node` command
- ✅ Tests different templates

### 2. **Build Test** (`test_build.sh`)
- ✅ Tests RoboDSL generation from .robodsl file
- ✅ Tests CMake configuration
- ✅ Tests ROS2 colcon build
- ✅ Compiles all generated files
- ✅ Verifies CUDA compilation
- ✅ Checks file organization

### 2. **CUDA Runtime Test** (`test_cuda_runtime.sh`)
- ✅ Tests CUDA kernel compilation
- ✅ Tests CUDA wrapper functionality
- ✅ Tests memory allocation
- ✅ Tests basic kernel execution

### 3. **ROS2 Integration Test** (`test_ros2_integration.sh`)
- ✅ Tests ROS2 node compilation
- ✅ Tests message type compilation
- ✅ Tests CUDA-ROS2 integration
- ⚠️ Skips if ROS2 not installed

### 4. **Performance Test** (`test_performance.sh`)
- ✅ Benchmarks CUDA kernels
- ✅ Tests memory bandwidth
- ✅ Stress tests with multiple threads
- ✅ Provides performance metrics

### 5. **Full System Test** (`test_full_system.sh`)
- ✅ Runs all tests in sequence
- ✅ Generates comprehensive report
- ✅ Provides system information
- ✅ Shows file count summary

## 🔧 Individual Test Commands

```bash
# Test RoboDSL CLI only
./test_robodsl_cli.sh

# Test build system only
./test_build.sh

# Test CUDA runtime only
./test_cuda_runtime.sh

# Test ROS2 integration only
./test_ros2_integration.sh

# Test performance only
./test_performance.sh

# Run all tests
./test_full_system.sh
```

## 📊 Expected Results

### Successful Test Output:
```
🚀 Starting Full System Test Suite
==================================

[INFO] Starting comprehensive test suite...

==========================================
[INFO] Running: Build Test
[INFO] Description: Tests CMake configuration, compilation, and file generation
==========================================
[SUCCESS] Build test completed successfully!

==========================================
[INFO] Running: CUDA Runtime Test
[INFO] Description: Tests CUDA kernel compilation and basic functionality
==========================================
[SUCCESS] All CUDA runtime tests completed successfully!

==========================================
[INFO] Running: ROS2 Integration Test
[INFO] Description: Tests ROS2 node compilation and integration
==========================================
[SUCCESS] All ROS2 integration tests completed successfully!

==========================================
[INFO] Running: Performance Test
[INFO] Description: Benchmarks CUDA kernels and memory bandwidth
==========================================
[SUCCESS] All performance tests completed successfully!

==========================================
📊 TEST SUMMARY REPORT
==========================================
Tests Passed:  4
Tests Failed:  0
Tests Skipped: 0
Total Tests:   4

🎉 ALL TESTS PASSED! Your RoboDSL generated files are working correctly.
```

## 🐛 Troubleshooting

### Common Issues:

#### 1. **CUDA not found**
```bash
# Check CUDA installation
nvcc --version
nvidia-smi

# If not found, install CUDA toolkit
sudo apt install cuda-toolkit-11-8
```

#### 2. **ROS2 not found**
```bash
# Check ROS2 installation
ros2 --version

# If not found, install ROS2
# Follow: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html
```

#### 3. **Build failures**
```bash
# Check compiler versions
g++ --version
nvcc --version

# Install missing packages
sudo apt install build-essential cmake pkg-config
```

#### 4. **Permission denied**
```bash
# Make scripts executable
chmod +x test_*.sh
```

#### 5. **Memory issues**
```bash
# Check GPU memory
nvidia-smi

# Reduce test data sizes in test scripts if needed
```

## 📈 Performance Expectations

### Typical Performance Results:
- **Vector Add**: 50-200 GB/s throughput
- **Matrix Multiply**: 100-500 GB/s throughput  
- **Image Filter**: 30-150 GB/s throughput
- **Memory Bandwidth**: 80-90% of theoretical max

### Factors Affecting Performance:
- GPU model and compute capability
- Memory bandwidth
- Data size and access patterns
- System load and temperature

## 🚀 Production Deployment

After successful testing:

1. **Deploy to target system**
2. **Run with real data:**
   ```bash
   ros2 launch launch/main_launch.py
   ```
3. **Monitor performance in production**
4. **Set up logging and monitoring**

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test output for specific error messages
3. Verify system requirements are met
4. Check GPU driver and CUDA compatibility

## 📝 Notes

- Tests are designed to be non-destructive
- Performance tests may take several minutes
- Some tests require GPU memory (ensure sufficient VRAM)
- ROS2 tests are optional but recommended for full functionality
