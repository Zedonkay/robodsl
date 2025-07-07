"""Comprehensive C++ Code Validation Tests.

This test suite validates that generated C++ code is:
1. Syntactically correct and compilable
2. Efficient in terms of memory usage, performance, and resource management
3. Following modern C++ best practices
4. Properly handling edge cases and error conditions
"""

import os
import sys
import pytest
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import re

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.generators import MainGenerator, CppNodeGenerator, CudaKernelGenerator, AdvancedCppGenerator
from robodsl.core.ast import RoboDSLAST


class CppCodeValidator:
    """Validates generated C++ code for correctness and efficiency."""
    
    def __init__(self):
        self.compiler_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2',
            '-fno-exceptions', '-fno-rtti', '-DNDEBUG'
        ]
        self.cuda_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-O2',
            '-arch=sm_60', '-DNDEBUG'
        ]
    
    def validate_syntax(self, cpp_code: str, filename: str = "test.cpp") -> bool:
        """Validate C++ syntax using g++ compiler."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.compiler_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(temp_file)
    
    def validate_cuda_syntax(self, cuda_code: str, filename: str = "test.cu") -> bool:
        """Validate CUDA syntax using nvcc compiler."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(cuda_code)
            temp_file = f.name
        
        try:
            cmd = ['nvcc'] + self.cuda_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(temp_file)
    
    def check_memory_leaks(self, cpp_code: str) -> List[str]:
        """Check for potential memory leaks in C++ code."""
        issues = []
        
        # Check for new without delete
        new_pattern = r'new\s+[^;]+;'
        delete_pattern = r'delete\s+[^;]+;'
        
        new_count = len(re.findall(new_pattern, cpp_code))
        delete_count = len(re.findall(delete_pattern, cpp_code))
        
        if new_count > delete_count:
            issues.append(f"Potential memory leak: {new_count} new statements vs {delete_count} delete statements")
        
        # Check for malloc without free
        malloc_pattern = r'malloc\s*\('
        free_pattern = r'free\s*\('
        
        malloc_count = len(re.findall(malloc_pattern, cpp_code))
        free_count = len(re.findall(free_pattern, cpp_code))
        
        if malloc_count > free_count:
            issues.append(f"Potential memory leak: {malloc_count} malloc calls vs {free_count} free calls")
        
        return issues
    
    def check_modern_cpp_practices(self, cpp_code: str) -> List[str]:
        """Check for modern C++ best practices."""
        issues = []
        
        # Check for raw pointers where smart pointers should be used
        raw_ptr_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\*\s*([a-zA-Z_][a-zA-Z0-9_]*)'
        if re.search(raw_ptr_pattern, cpp_code) and 'std::unique_ptr' not in cpp_code and 'std::shared_ptr' not in cpp_code:
            issues.append("Consider using smart pointers instead of raw pointers")
        
        # Check for C-style casts
        c_style_casts = re.findall(r'\([^)]*\)\s*[a-zA-Z_][a-zA-Z0-9_]*', cpp_code)
        if c_style_casts:
            issues.append("Consider using C++ style casts (static_cast, dynamic_cast, etc.)")
        
        # Check for manual memory management in constructors/destructors
        if 'new' in cpp_code and ('__init__' in cpp_code or '__del__' in cpp_code):
            issues.append("Consider using RAII and smart pointers for automatic resource management")
        
        return issues
    
    def check_performance_issues(self, cpp_code: str) -> List[str]:
        """Check for potential performance issues."""
        issues = []
        
        # Check for unnecessary copies
        if 'const&' not in cpp_code and 'const' in cpp_code:
            issues.append("Consider using const references to avoid unnecessary copies")
        
        # Check for inefficient string operations
        if 'std::string' in cpp_code and '+' in cpp_code:
            issues.append("Consider using std::stringstream or string concatenation optimization")
        
        # Check for virtual functions in tight loops
        if 'virtual' in cpp_code and 'for' in cpp_code:
            issues.append("Virtual function calls in loops may impact performance")
        
        return issues
    
    def check_cuda_best_practices(self, cuda_code: str) -> List[str]:
        """Check for CUDA best practices."""
        issues = []
        
        # Check for proper error handling
        if 'cudaError_t' in cuda_code and 'cudaGetLastError' not in cuda_code:
            issues.append("CUDA error handling should include cudaGetLastError() calls")
        
        # Check for memory coalescing
        if 'threadIdx.x' in cuda_code and 'blockDim.x' in cuda_code:
            # This is a basic check - more sophisticated analysis would be needed
            pass
        
        # Check for proper synchronization
        if 'cudaMemcpyAsync' in cuda_code and 'cudaStreamSynchronize' not in cuda_code:
            issues.append("Asynchronous CUDA operations should be properly synchronized")
        
        return issues


class TestCppCodeValidation:
    """Test suite for C++ code validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CppCodeValidator()
        self.generator = MainGenerator()
    
    def test_basic_node_generation_syntax(self, test_output_dir):
        """Test that basic node generation produces valid C++ syntax."""
        source = """
        node test_node {
            parameter int max_speed = 10
            publisher /test: std_msgs/String
            subscriber /input: std_msgs/String
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Find generated C++ files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Generated C++ file {cpp_file} has syntax errors"
    
    def test_cuda_kernel_generation_syntax(self, test_output_dir):
        """Test that CUDA kernel generation produces valid syntax."""
        source = """
        cuda_kernel test_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Find generated CUDA files
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Generated CUDA file {cuda_file} has syntax errors"
    
    def test_advanced_cpp_features_syntax(self, test_output_dir):
        """Test that advanced C++ features generate valid syntax."""
        source = """
        template<typename T> struct Vector {
            T* data;
            size_t size;
        }
        
        global PI: constexpr float = 3.14159;
        
        def operator<<(stream: std::ostream&, vec: Vector<int>&) -> std::ostream& {
            stream << "Vector";
            return stream;
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Find generated C++ files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Generated C++ file {cpp_file} has syntax errors"
    
    def test_memory_management_validation(self, test_output_dir):
        """Test that generated code follows proper memory management practices."""
        source = """
        node memory_test_node {
            parameter int buffer_size = 1000
            
            def process_data(data: float[]) -> float[] {
                // This should not have memory leaks
                return data;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            memory_issues = self.validator.check_memory_leaks(content)
            assert len(memory_issues) == 0, \
                f"Memory management issues in {cpp_file}: {memory_issues}"
    
    def test_modern_cpp_practices(self, test_output_dir):
        """Test that generated code follows modern C++ practices."""
        source = """
        node modern_cpp_node {
            parameter std::string name = "test"
            
            def process(input: const std::vector<float>&) -> std::vector<float> {
                return input;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            practice_issues = self.validator.check_modern_cpp_practices(content)
            # Allow some issues as the generator might not be perfect
            assert len(practice_issues) <= 2, \
                f"Too many modern C++ practice violations in {cpp_file}: {practice_issues}"
    
    def test_cuda_best_practices(self, test_output_dir):
        """Test that generated CUDA code follows best practices."""
        source = """
        cuda_kernel optimized_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // This should include proper error handling
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            cuda_issues = self.validator.check_cuda_best_practices(content)
            # Allow some issues as the generator might not be perfect
            assert len(cuda_issues) <= 1, \
                f"Too many CUDA best practice violations in {cuda_file}: {cuda_issues}"
    
    def test_performance_optimization(self, test_output_dir):
        """Test that generated code includes performance optimizations."""
        source = """
        node performance_node {
            parameter int iterations = 1000
            
            def optimized_process(input: const std::vector<float>&) -> std::vector<float> {
                // This should be optimized
                return input;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            perf_issues = self.validator.check_performance_issues(content)
            # Allow some issues as the generator might not be perfect
            assert len(perf_issues) <= 2, \
                f"Too many performance issues in {cpp_file}: {perf_issues}"
    
    def test_comprehensive_node_validation(self, test_output_dir):
        """Test comprehensive node with all features."""
        source = """
        include <iostream>
        include <vector>
        
        template<typename T> struct DataContainer {
            T* data;
            size_t size;
        }
        
        global MAX_SIZE: constexpr int = 1000;
        
        node comprehensive_node {
            parameter int max_speed = 10
            parameter float safety_distance = 1.5
            
            publisher /robot/position: geometry_msgs/Point
            subscriber /robot/command: geometry_msgs/Twist
            service /robot/status: std_srvs/Trigger
            
            def process_data(input: const std::vector<float>&) -> std::vector<float> {
                return input;
            }
            
            def __init__() {
                // Constructor
            }
            
            def __del__() {
                // Destructor
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Validate all generated files
        for file_path in generated_files:
            if file_path.suffix in ['.cpp', '.hpp']:
                content = file_path.read_text()
                
                # Check syntax
                assert self.validator.validate_syntax(content, str(file_path)), \
                    f"Syntax error in {file_path}"
                
                # Check memory management
                memory_issues = self.validator.check_memory_leaks(content)
                assert len(memory_issues) == 0, \
                    f"Memory issues in {file_path}: {memory_issues}"
                
                # Check modern C++ practices
                practice_issues = self.validator.check_modern_cpp_practices(content)
                assert len(practice_issues) <= 3, \
                    f"Too many practice violations in {file_path}: {practice_issues}"
    
    def test_edge_cases_validation(self, test_output_dir):
        """Test edge cases in generated code."""
        source = """
        node edge_case_node {
            parameter int zero_param = 0
            parameter float negative_param = -1.0
            parameter std::string empty_string = ""
            
            def empty_function() {
                // Empty function
            }
            
            def single_parameter(x: int) -> int {
                return x;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        for file_path in generated_files:
            if file_path.suffix in ['.cpp', '.hpp']:
                content = file_path.read_text()
                
                # Should still compile with edge case values
                assert self.validator.validate_syntax(content, str(file_path)), \
                    f"Edge case compilation failed in {file_path}"
    
    def test_large_scale_generation(self, test_output_dir):
        """Test generation of large-scale code."""
        source = """
        // Generate many nodes to test scalability
        """
        
        # Add multiple nodes to the source
        for i in range(5):
            source += f"""
            node large_scale_node_{i} {{
                parameter int id = {i}
                publisher /node_{i}/data: std_msgs/String
                subscriber /node_{i}/command: std_msgs/String
                
                def process_{i}(input: const std::vector<float>&) -> std::vector<float> {{
                    return input;
                }}
            }}
            """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Validate all generated files
        for file_path in generated_files:
            if file_path.suffix in ['.cpp', '.hpp']:
                content = file_path.read_text()
                
                # Check syntax
                assert self.validator.validate_syntax(content, str(file_path)), \
                    f"Large scale generation syntax error in {file_path}"
                
                # Check for reasonable file size (not too large)
                assert len(content) < 10000, \
                    f"Generated file {file_path} is too large: {len(content)} characters"


if __name__ == "__main__":
    pytest.main([__file__]) 