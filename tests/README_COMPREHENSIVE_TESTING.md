# Comprehensive Linux Dependency Testing

This directory contains extensive test coverage for all Linux dependencies and advanced features of the RoboDSL tool. The testing system is designed to validate functionality across multiple platforms and dependency configurations.

## 🧪 Test Overview

### Core Test Files

1. **`test_comprehensive_linux_dependencies.py`** - Main comprehensive test suite
   - System dependency detection
   - Multi-GPU support
   - Memory management
   - Performance optimization
   - Error handling and recovery
   - Compatibility testing
   - Security testing
   - Deployment testing

2. **`test_cuda_advanced_features.py`** - Advanced CUDA features
   - Multi-GPU support
   - CUDA streams and events
   - Dynamic parallelism
   - Cooperative groups
   - Advanced memory management
   - Optimization techniques
   - Performance monitoring
   - Error handling

3. **`test_tensorrt_advanced_features.py`** - Advanced TensorRT features
   - Mixed precision modes
   - Dynamic shapes and batch sizes
   - INT8 quantization
   - TensorRT plugins
   - Performance tuning
   - Memory optimization
   - Multi-stream inference
   - Profiling and debugging

4. **`test_ros2_advanced_features.py`** - Advanced ROS2 features
   - QoS configurations
   - Lifecycle nodes
   - Services and actions
   - Parameters and parameter servers
   - Advanced communication patterns
   - Multi-node systems
   - Real-time features
   - Security features

5. **`test_onnx_advanced_features.py`** - Advanced ONNX features
   - Dynamic shapes
   - Custom operators
   - Advanced optimization passes
   - Multi-device inference
   - Performance tuning
   - Memory management
   - Streaming inference
   - Model optimization

6. **`test_cmake_advanced_features.py`** - Advanced CMake features
   - Multi-configuration builds
   - Cross-compilation support
   - Advanced package management
   - Build system optimization
   - Dependency management
   - Custom build targets
   - Platform-specific configurations

## 🚀 Running Tests

### Quick Start

```bash
# Run all comprehensive tests
./run_linux_tests.sh

# Or run the Python test runner directly
python tests/run_comprehensive_linux_tests.py --verbose --report
```

### Advanced Usage

```bash
# Run tests in parallel
python tests/run_comprehensive_linux_tests.py --parallel --verbose

# Run specific test categories
python tests/run_comprehensive_linux_tests.py --filter "cuda"

# Run with custom timeout
python tests/run_comprehensive_linux_tests.py --timeout 1200

# Generate coverage report
python tests/run_comprehensive_linux_tests.py --coverage --report
```

### Individual Test Files

```bash
# Run specific test modules
pytest tests/test_cuda_advanced_features.py -v
pytest tests/test_tensorrt_advanced_features.py -v
pytest tests/test_ros2_advanced_features.py -v
pytest tests/test_onnx_advanced_features.py -v
pytest tests/test_cmake_advanced_features.py -v
```

## 📋 System Requirements

### Required Dependencies

- **Python 3.8+**
- **CUDA 11.0+** (for GPU tests)
- **TensorRT 8.0+** (for optimization tests)
- **ROS2 Humble+** (for ROS2 tests)
- **ONNX Runtime 1.12+** (for inference tests)
- **CMake 3.16+** (for build tests)
- **GCC 9+ or Clang 12+** (for compilation tests)

### Optional Dependencies

- **Multiple GPUs** (for multi-GPU tests)
- **Large GPU memory** (for memory stress tests)
- **High-performance CPU** (for optimization tests)
- **Real-time kernel** (for real-time tests)

## 🔍 Test Categories

### 1. System Detection Tests

Tests that verify the system can properly detect and configure:
- Available GPUs and their capabilities
- CUDA version and compatibility
- TensorRT installation and version
- ROS2 distribution and packages
- ONNX Runtime providers
- Compiler versions and capabilities

### 2. Performance Tests

Tests that validate performance characteristics:
- Memory usage and management
- GPU utilization and efficiency
- Inference latency and throughput
- Build time optimization
- Parallel processing capabilities

### 3. Compatibility Tests

Tests that ensure compatibility across:
- Different CUDA versions (10.0-12.2)
- Different TensorRT versions (7.0-9.0)
- Different ROS2 distributions (Foxy-Rolling)
- Different Python versions (3.8-3.12)
- Different compiler versions (GCC 9-12, Clang 12-15)

### 4. Edge Case Tests

Tests that handle edge cases and error conditions:
- Invalid configurations
- Memory exhaustion scenarios
- GPU failures and recovery
- Network timeouts and retries
- Malicious input handling

### 5. Security Tests

Tests that validate security features:
- Input validation and sanitization
- Memory safety and buffer overflow protection
- Authentication and authorization
- Encryption and secure communication
- Privacy-preserving features

## 📊 Test Reports

### Generated Reports

The test runner generates comprehensive reports including:

1. **System Information**
   - Platform and architecture
   - Python version
   - Available dependencies
   - Hardware specifications

2. **Test Results**
   - Pass/fail status for each test
   - Execution time and performance metrics
   - Error messages and stack traces
   - Coverage information

3. **Dependency Analysis**
   - Available vs required dependencies
   - Version compatibility matrix
   - Missing or incompatible components

4. **Performance Metrics**
   - Memory usage patterns
   - GPU utilization statistics
   - Build and execution times
   - Optimization effectiveness

### Report Files

- `test_report.json` - Comprehensive test results
- `coverage_report.html` - Code coverage report (if enabled)
- `performance_metrics.json` - Performance analysis
- `dependency_matrix.json` - Dependency compatibility matrix

## 🛠️ Test Configuration

### Environment Variables

```bash
# CUDA configuration
export CUDA_VISIBLE_DEVICES=0,1
export CUDA_LAUNCH_BLOCKING=1

# TensorRT configuration
export TENSORRT_ROOT=/usr/local/tensorrt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$TENSORRT_ROOT/lib

# ROS2 configuration
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ONNX Runtime configuration
export ORT_DISABLE_GPU_POOLING=1
export ORT_DISABLE_MEMORY_PATTERN=1
```

### Test Configuration Files

- `tests/conftest.py` - Pytest configuration and fixtures
- `tests/test_config.json` - Test-specific configurations
- `tests/dependency_config.json` - Dependency-specific settings

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Tests Failing**
   ```bash
   # Check CUDA installation
   nvidia-smi
   nvcc --version
   
   # Verify GPU availability
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **TensorRT Tests Failing**
   ```bash
   # Check TensorRT installation
   ls /usr/local/tensorrt/lib/
   python -c "import tensorrt as trt; print(trt.__version__)"
   ```

3. **ROS2 Tests Failing**
   ```bash
   # Check ROS2 installation
   ros2 --version
   ros2 pkg list | grep rclcpp
   ```

4. **ONNX Tests Failing**
   ```bash
   # Check ONNX Runtime installation
   python -c "import onnxruntime as ort; print(ort.__version__)"
   ```

### Debug Mode

```bash
# Run tests with debug output
python tests/run_comprehensive_linux_tests.py --verbose --debug

# Run specific test with maximum verbosity
pytest tests/test_cuda_advanced_features.py -vvv -s
```

## 📈 Performance Benchmarks

### Baseline Performance

The tests include performance benchmarks for:
- **Memory allocation**: 1GB/s sustained
- **GPU computation**: 90%+ utilization
- **Inference latency**: <16ms for 224x224 images
- **Build time**: <5 minutes for full project
- **Test execution**: <10 minutes for all tests

### Optimization Targets

- **Memory efficiency**: <2GB peak usage
- **GPU efficiency**: >95% utilization
- **CPU efficiency**: <80% utilization
- **Network efficiency**: <100ms latency
- **Build efficiency**: <3 minutes incremental

## 🤝 Contributing

### Adding New Tests

1. **Create test file**: `tests/test_new_feature.py`
2. **Add test class**: Inherit from appropriate base class
3. **Implement test methods**: Use descriptive names
4. **Add to test runner**: Update `run_comprehensive_linux_tests.py`
5. **Update documentation**: Add to this README

### Test Guidelines

- **Descriptive names**: Use clear, descriptive test names
- **Comprehensive coverage**: Test all code paths and edge cases
- **Performance testing**: Include performance benchmarks
- **Error handling**: Test error conditions and recovery
- **Documentation**: Document complex test scenarios

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **System tests**: Test end-to-end functionality
- **Performance tests**: Test performance characteristics
- **Security tests**: Test security features and vulnerabilities

## 📚 Additional Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ROS2 Documentation](https://docs.ros.org/en/humble/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/)
- [CMake Documentation](https://cmake.org/documentation/)

## 🐛 Bug Reports

When reporting bugs, please include:
1. **System information**: OS, Python version, dependency versions
2. **Test output**: Full test output with error messages
3. **Reproduction steps**: Steps to reproduce the issue
4. **Expected behavior**: What should happen
5. **Actual behavior**: What actually happens

## 📄 License

This testing framework is part of the RoboDSL project and follows the same license terms. 