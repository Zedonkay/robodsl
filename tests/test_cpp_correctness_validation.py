"""C++ Code Correctness Validation Tests.

This test suite specifically validates the correctness aspects of generated C++ code:
- Type safety
- Exception safety
- Thread safety
- Resource management correctness
- API correctness
- Error handling
"""

import os
import sys
import pytest
import subprocess
import tempfile
import re
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.generators import MainGenerator


class CppCorrectnessValidator:
    """Validates correctness aspects of generated C++ code."""
    
    def __init__(self):
        self.strict_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-Werror', '-Wpedantic',
            '-fno-exceptions', '-fno-rtti', '-DNDEBUG'
        ]
    
    def check_type_safety(self, cpp_code: str) -> List[str]:
        """Check for type safety issues."""
        issues = []
        
        # Check for implicit conversions
        if 'int' in cpp_code and 'float' in cpp_code:
            issues.append("Check for potential implicit type conversions")
        
        # Check for proper const usage
        if 'const' not in cpp_code and 'std::string' in cpp_code:
            issues.append("Consider using const for immutable data")
        
        # Check for proper reference usage
        if '&' not in cpp_code and 'std::vector' in cpp_code:
            issues.append("Consider using references to avoid copies")
        
        return issues
    
    def check_exception_safety(self, cpp_code: str) -> List[str]:
        """Check for exception safety issues."""
        issues = []
        
        # Check for RAII usage
        if 'new' in cpp_code and 'std::unique_ptr' not in cpp_code:
            issues.append("Use RAII for exception safety")
        
        # Check for proper cleanup in destructors
        if '__del__' in cpp_code and 'delete' not in cpp_code:
            issues.append("Ensure proper cleanup in destructors")
        
        return issues
    
    def check_thread_safety(self, cpp_code: str) -> List[str]:
        """Check for thread safety issues."""
        issues = []
        
        # Check for mutable shared state
        if 'static' in cpp_code and 'const' not in cpp_code:
            issues.append("Consider thread safety for static variables")
        
        # Check for proper synchronization
        if 'std::thread' in cpp_code and 'std::mutex' not in cpp_code:
            issues.append("Consider synchronization for multi-threaded code")
        
        return issues
    
    def check_resource_management_correctness(self, cpp_code: str) -> List[str]:
        """Check for correct resource management."""
        issues = []
        
        # Check for proper RAII
        if 'cudaMalloc' in cpp_code and 'cudaFree' not in cpp_code:
            issues.append("Ensure CUDA memory is properly freed")
        
        # Check for smart pointer usage
        if 'new' in cpp_code and 'std::unique_ptr' not in cpp_code and 'std::shared_ptr' not in cpp_code:
            issues.append("Consider using smart pointers for automatic resource management")
        
        return issues
    
    def check_api_correctness(self, cpp_code: str) -> List[str]:
        """Check for API correctness."""
        issues = []
        
        # Check for proper ROS2 API usage
        if 'rclcpp' in cpp_code and 'get_logger' not in cpp_code:
            issues.append("Consider using proper ROS2 logging")
        
        # Check for proper error handling
        if 'cudaError_t' in cpp_code and 'cudaSuccess' not in cpp_code:
            issues.append("Check CUDA error codes")
        
        return issues
    
    def check_error_handling(self, cpp_code: str) -> List[str]:
        """Check for proper error handling."""
        issues = []
        
        # Check for null pointer checks
        if '*' in cpp_code and 'nullptr' not in cpp_code and 'null' not in cpp_code:
            issues.append("Consider null pointer checks")
        
        # Check for bounds checking
        if '[' in cpp_code and 'size()' not in cpp_code:
            issues.append("Consider bounds checking for array access")
        
        return issues
    
    def validate_strict_compilation(self, cpp_code: str) -> bool:
        """Validate that code compiles with strict flags."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.strict_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(temp_file)


class TestCppCorrectnessValidation:
    """Test suite for C++ correctness validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CppCorrectnessValidator()
        self.generator = MainGenerator()
    
    def test_type_safety(self, test_output_dir):
        """Test that generated code is type-safe."""
        source = """
        node type_safe_node {
            parameter int count = 10
            parameter float threshold = 0.5
            
            def type_safe_function(input: const std::vector<float>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(input.size());
                
                for (const auto& value : input) {
                    if (value > threshold) {
                        result.push_back(value);
                    }
                }
                return result;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            type_issues = self.validator.check_type_safety(content)
            assert len(type_issues) <= 2, \
                f"Type safety issues in {cpp_file}: {type_issues}"
    
    def test_exception_safety(self, test_output_dir):
        """Test that generated code is exception-safe."""
        source = """
        node exception_safe_node {
            def exception_safe_function() -> std::unique_ptr<std::vector<float>> {
                auto result = std::make_unique<std::vector<float>>();
                result->reserve(1000);
                return result;  // RAII ensures cleanup
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            exception_issues = self.validator.check_exception_safety(content)
            assert len(exception_issues) <= 1, \
                f"Exception safety issues in {cpp_file}: {exception_issues}"
    
    def test_thread_safety(self, test_output_dir):
        """Test that generated code is thread-safe."""
        source = """
        node thread_safe_node {
            parameter const int max_threads = 4
            
            def thread_safe_function() -> int {
                static const int thread_id = 0;  // const for thread safety
                return thread_id;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            thread_issues = self.validator.check_thread_safety(content)
            assert len(thread_issues) <= 1, \
                f"Thread safety issues in {cpp_file}: {thread_issues}"
    
    def test_resource_management_correctness(self, test_output_dir):
        """Test that generated code correctly manages resources."""
        source = """
        node resource_correct_node {
            def correct_resource_management() -> std::unique_ptr<std::vector<float>> {
                auto result = std::make_unique<std::vector<float>>();
                result->reserve(1000);
                return result;  // RAII ensures cleanup
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            resource_issues = self.validator.check_resource_management_correctness(content)
            assert len(resource_issues) <= 1, \
                f"Resource management correctness issues in {cpp_file}: {resource_issues}"
    
    def test_api_correctness(self, test_output_dir):
        """Test that generated code uses APIs correctly."""
        source = """
        node api_correct_node {
            def api_correct_function() -> bool {
                RCLCPP_INFO(this->get_logger(), "Using proper ROS2 logging");
                return true;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            api_issues = self.validator.check_api_correctness(content)
            assert len(api_issues) <= 1, \
                f"API correctness issues in {cpp_file}: {api_issues}"
    
    def test_error_handling(self, test_output_dir):
        """Test that generated code handles errors properly."""
        source = """
        node error_handling_node {
            def error_handling_function(input: const std::vector<float>&) -> float {
                if (input.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Empty input vector");
                    return 0.0f;
                }
                
                if (input.size() > 1000) {
                    RCLCPP_ERROR(this->get_logger(), "Input too large");
                    return 0.0f;
                }
                
                return input[0];
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            error_issues = self.validator.check_error_handling(content)
            assert len(error_issues) <= 2, \
                f"Error handling issues in {cpp_file}: {error_issues}"
    
    def test_strict_compilation(self, test_output_dir):
        """Test that generated code compiles with strict flags."""
        source = """
        node strict_node {
            parameter const int max_size = 1000
            
            def strict_function(input: const std::vector<float>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(input.size());
                
                for (const auto& value : input) {
                    result.push_back(value);
                }
                return result;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_strict_compilation(content), \
                f"Generated code in {cpp_file} does not compile with strict flags"
    
    def test_cuda_correctness(self, test_output_dir):
        """Test that generated CUDA code is correct."""
        source = """
        cuda_kernel correct_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // Should include proper error checking
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            
            # Check for proper CUDA patterns
            assert 'cudaError_t' in content or 'cudaGetLastError' in content, \
                "CUDA code should include error checking"
            
            # Check for proper memory management
            if 'cudaMalloc' in content:
                assert 'cudaFree' in content, "CUDA memory should be freed"
    
    def test_comprehensive_correctness(self, test_output_dir):
        """Test comprehensive correctness validation."""
        source = """
        include <iostream>
        include <vector>
        include <memory>
        
        template<typename T> struct SafeContainer {
            std::unique_ptr<T[]> data;
            size_t size;
            
            SafeContainer(size_t n) : size(n) {
                data = std::make_unique<T[]>(n);
            }
        }
        
        node comprehensive_correct_node {
            parameter const int max_size = 1000
            parameter const float safety_threshold = 0.5
            
            publisher /safe/data: std_msgs/Float32MultiArray
            subscriber /safe/command: std_msgs/Float32MultiArray
            
            def comprehensive_correct_function(input: const std::vector<float>&) -> std::vector<float> {
                if (input.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Empty input");
                    return {};
                }
                
                if (input.size() > max_size) {
                    RCLCPP_ERROR(this->get_logger(), "Input too large");
                    return {};
                }
                
                std::vector<float> result;
                result.reserve(input.size());
                
                for (const auto& value : input) {
                    if (value > safety_threshold) {
                        result.push_back(value);
                    }
                }
                
                RCLCPP_INFO(this->get_logger(), "Processed %zu values", result.size());
                return result;
            }
            
            def __init__() {
                // Constructor with proper initialization
            }
            
            def __del__() {
                // Destructor with proper cleanup
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Validate all generated files
        for file_path in generated_files:
            if file_path.suffix in ['.cpp', '.hpp']:
                content = file_path.read_text()
                
                # Check strict compilation
                assert self.validator.validate_strict_compilation(content), \
                    f"Strict compilation failed for {file_path}"
                
                # Check type safety
                type_issues = self.validator.check_type_safety(content)
                assert len(type_issues) <= 3, \
                    f"Type safety issues in {file_path}: {type_issues}"
                
                # Check exception safety
                exception_issues = self.validator.check_exception_safety(content)
                assert len(exception_issues) <= 2, \
                    f"Exception safety issues in {file_path}: {exception_issues}"
                
                # Check error handling
                error_issues = self.validator.check_error_handling(content)
                assert len(error_issues) <= 2, \
                    f"Error handling issues in {file_path}: {error_issues}"


if __name__ == "__main__":
    pytest.main([__file__]) 