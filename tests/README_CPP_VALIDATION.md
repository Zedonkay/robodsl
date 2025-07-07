# C++ Code Validation Test Suite

This directory contains comprehensive test cases to validate that generated C++ code is correct, efficient, and follows modern C++ best practices.

## Overview

The C++ validation test suite consists of several specialized test files that validate different aspects of generated C++ code:

### Test Files

#### Core C++ Validation
1. **`test_cpp_code_validation.py`** - Main validation tests for syntax, memory management, and basic correctness
2. **`test_cpp_efficiency_validation.py`** - Efficiency-focused validation including performance optimizations
3. **`test_cpp_correctness_validation.py`** - Correctness validation including type safety and error handling
4. **`test_cuda_code_validation.py`** - CUDA-specific validation for kernel correctness and efficiency
5. **`test_comprehensive_cpp_validation.py`** - Comprehensive test runner that combines all validations
6. **`test_cpp_validation_config.py`** - Test configuration and test case definitions

#### Advanced Features Validation
7. **`test_pipeline_validation.py`** - Pipeline generation validation tests
8. **`test_onnx_tensorrt_validation.py`** - ONNX and TensorRT integration validation tests
9. **`test_advanced_features_config.py`** - Test configurations for advanced features
10. **`test_advanced_features_runner.py`** - Advanced features test runner
11. **`run_advanced_features_validation.py`** - Comprehensive advanced features validation runner

## Test Categories

### 1. Basic Node Validation
- Simple ROS2 node generation
- Publisher/subscriber functionality
- Service and action server integration
- Lifecycle node support

### 2. CUDA Kernel Validation
- CUDA kernel syntax correctness
- Memory management (cudaMalloc/cudaFree)
- Error handling and synchronization
- Performance optimization patterns

### 3. Advanced C++ Features
- Template generation (structs, classes, functions)
- Operator overloading
- Constexpr variables and functions
- Concepts and type constraints

### 4. Efficiency Validation
- Memory allocation patterns
- Smart pointer usage
- Move semantics
- Algorithm efficiency
- Compiler optimization compatibility

### 5. Correctness Validation
- Type safety
- Exception safety
- Thread safety
- Resource management
- Error handling

### 6. Edge Cases
- Zero values and empty containers
- Large values and overflow conditions
- Boundary conditions
- Error scenarios

### 7. Pipeline Validation
- Multi-stage pipeline generation
- Stage integration and data flow
- ROS2 topic management
- CUDA and ONNX integration in pipelines

### 8. ONNX Integration Validation
- ONNX Runtime session management
- Tensor handling and memory management
- Model loading and inference
- Error handling and exception safety

### 9. TensorRT Optimization Validation
- TensorRT provider configuration
- FP16 and INT8 optimization
- Engine caching and workspace management
- Performance optimization patterns

## Running the Tests

### Prerequisites

1. **C++ Compiler**: GCC with C++17 support
2. **CUDA Compiler**: NVCC for CUDA kernel validation
3. **Python Dependencies**: pytest, robodsl

### Basic Test Execution

```bash
# Run all C++ validation tests
pytest tests/test_cpp_code_validation.py -v

# Run efficiency tests only
pytest tests/test_cpp_efficiency_validation.py -v

# Run CUDA tests only
pytest tests/test_cuda_code_validation.py -v

# Run comprehensive validation
pytest tests/test_comprehensive_cpp_validation.py -v

# Run advanced features validation
pytest tests/test_advanced_features_runner.py -v

# Run pipeline validation
pytest tests/test_pipeline_validation.py -v

# Run ONNX/TensorRT validation
pytest tests/test_onnx_tensorrt_validation.py -v
```

### Running Specific Test Categories

```bash
# Run basic node tests
pytest tests/test_cpp_code_validation.py::TestCppCodeValidation::test_basic_node_generation_syntax -v

# Run CUDA kernel tests
pytest tests/test_cuda_code_validation.py::TestCudaCodeValidation::test_basic_cuda_kernel_syntax -v

# Run efficiency tests
pytest tests/test_cpp_efficiency_validation.py::TestCppEfficiencyValidation::test_memory_allocation_efficiency -v
```

### Running with Custom Configuration

```bash
# Run with specific compiler flags
CXXFLAGS="-std=c++17 -O3" pytest tests/test_cpp_code_validation.py -v

# Run with CUDA support
CUDA_PATH=/usr/local/cuda pytest tests/test_cuda_code_validation.py -v

# Run advanced features with custom configuration
python tests/run_advanced_features_validation.py --category pipeline --verbose

# Run all advanced features validation
python tests/run_advanced_features_validation.py --category all --report validation_report.json
```

## Validation Criteria

### Syntax Validation
- **Requirement**: All generated C++ code must compile without errors
- **Compiler Flags**: `-std=c++17 -Wall -Wextra -Werror -O2`
- **Tolerance**: 0 syntax errors allowed

### Memory Management Validation
- **Requirement**: Proper RAII and smart pointer usage
- **Checks**: new/delete balance, malloc/free balance
- **Tolerance**: 1 memory suggestion allowed

### Performance Validation
- **Requirement**: Efficient algorithms and data structures
- **Checks**: Container usage, loop optimization, copy avoidance
- **Tolerance**: 2 performance suggestions allowed

### Best Practices Validation
- **Requirement**: Modern C++ practices
- **Checks**: const correctness, range-based loops, nullptr usage
- **Tolerance**: 2 practice suggestions allowed

### CUDA Validation
- **Requirement**: Proper CUDA patterns and error handling
- **Checks**: Memory management, synchronization, error checking
- **Tolerance**: 1 CUDA suggestion allowed

### Pipeline Validation
- **Requirement**: Proper pipeline structure and stage integration
- **Checks**: ROS2 integration, data flow, stage management
- **Tolerance**: 2 pipeline suggestions allowed

### ONNX Validation
- **Requirement**: Proper ONNX Runtime integration
- **Checks**: Session management, tensor handling, error handling
- **Tolerance**: 2 ONNX suggestions allowed

### TensorRT Validation
- **Requirement**: Proper TensorRT optimization configuration
- **Checks**: Provider setup, optimization settings, cache management
- **Tolerance**: 2 TensorRT suggestions allowed

## Test Case Structure

Each test case follows this structure:

```python
{
    "name": "test_case_name",
    "source": """
        // RoboDSL source code
        node test_node {
            parameter int value = 42
            publisher /test: std_msgs/String
        }
    """,
    "expected_issues": {
        "syntax": 0,      # No syntax errors
        "memory": 1,      # Allow 1 memory suggestion
        "performance": 1, # Allow 1 performance suggestion
        "practices": 1    # Allow 1 practice suggestion
    }
}
```

## Validation Report

The comprehensive test runner generates detailed reports including:

- **Success Rate**: Percentage of tests that passed
- **Issue Categories**: Breakdown of issues by type
- **Detailed Results**: Per-file validation results
- **Performance Metrics**: Compilation time and file sizes

### Sample Report

```json
{
    "summary": {
        "total_tests": 15,
        "passed_tests": 14,
        "failed_tests": 1,
        "success_rate": 93.3
    },
    "issues": {
        "syntax_issues": 0,
        "memory_issues": 2,
        "performance_issues": 3,
        "practice_issues": 1,
        "cuda_issues": 1
    }
}
```

## Adding New Test Cases

### 1. Define Test Case

Add your test case to the appropriate category in `test_cpp_validation_config.py`:

```python
NEW_TEST_CASE = {
    "name": "my_new_test",
    "source": """
        // Your RoboDSL source code here
    """,
    "expected_issues": {
        "syntax": 0,
        "memory": 1,
        "performance": 1,
        "practices": 1
    }
}
```

### 2. Add to Test Category

Add your test case to the appropriate test category:

```python
MY_TEST_CATEGORY = [
    # ... existing test cases ...
    NEW_TEST_CASE
]
```

### 3. Create Test Function

Add a test function to the appropriate test class:

```python
def test_my_new_test(self, test_output_dir):
    """Test my new functionality."""
    source = NEW_TEST_CASE["source"]
    expected_issues = NEW_TEST_CASE["expected_issues"]
    
    # Run validation
    results = self.validator.run_comprehensive_validation(source, "my_new_test")
    report = self.validator.generate_validation_report(results)
    
    # Assert results
    assert report['summary']['success_rate'] >= 80
    assert report['issues']['syntax_issues'] <= expected_issues['syntax']
```

## Continuous Integration

These tests are designed to run in CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Run C++ Validation Tests
  run: |
    pytest tests/test_cpp_code_validation.py -v
    pytest tests/test_cuda_code_validation.py -v
    pytest tests/test_comprehensive_cpp_validation.py -v
```

## Troubleshooting

### Common Issues

1. **Compiler Not Found**: Ensure GCC and NVCC are installed and in PATH
2. **CUDA Not Available**: Skip CUDA tests if CUDA is not available
3. **Timeout Errors**: Increase timeout for complex test cases
4. **Memory Issues**: Check for proper cleanup in test fixtures

### Debug Mode

Run tests with debug output:

```bash
pytest tests/test_cpp_code_validation.py -v -s --tb=long
```

### Verbose Output

Get detailed validation information:

```bash
pytest tests/test_comprehensive_cpp_validation.py -v -s
```

## Performance Benchmarks

The test suite includes performance benchmarks:

- **Compilation Time**: Should be under 30 seconds per file
- **File Size**: Generated files should be under 100KB
- **Memory Usage**: No memory leaks detected
- **Optimization**: Code should compile with -O3

## Contributing

When adding new validation tests:

1. Follow the existing test structure
2. Include comprehensive test cases
3. Document expected behavior
4. Add appropriate error handling
5. Update this README if needed

## References

- [Modern C++ Best Practices](https://isocpp.github.io/CppCoreGuidelines/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [ROS2 C++ Style Guide](https://docs.ros.org/en/humble/Contributing/Developer-Guide.html#cpp-style-guide) 