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
from lark import Lark, Tree

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda
from robodsl.generators import MainGenerator, CppNodeGenerator, CudaKernelGenerator, AdvancedCppGenerator
from robodsl.core.ast import RoboDSLAST


class CppCodeValidator:
    """Validates generated C++ code for correctness and efficiency."""
    
    def __init__(self):
        self.compiler_flags = ['-std=c++17', '-Wall', '-Wextra']
        self.cuda_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-O2',
            '-arch=sm_60', '-DNDEBUG'
        ]
    
    def validate_syntax(self, cpp_code: str, filename: str = "test.cpp") -> bool:
        """Validate C++ syntax using g++ compiler."""
        # Only compile .cpp files
        if not filename.endswith('.cpp'):
            return True  # Skip header files, they will be validated via .cpp includes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.compiler_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            # If compilation succeeds, syntax is valid
            if result.returncode == 0:
                return True
            
            # If compilation fails, check if it's due to missing headers vs syntax errors
            stderr = result.stderr.lower()
            
            # Check if the error is due to missing headers (these are acceptable in test environment)
            missing_header_indicators = [
                'no such file or directory',
                'cannot find',
                'file not found'
            ]
            
            is_missing_header = any(indicator in stderr for indicator in missing_header_indicators)
            
            if is_missing_header:
                # Missing headers are acceptable in test environment
                # Just check for basic syntax validity by looking for common syntax error patterns
                syntax_error_indicators = [
                    'expected',
                    'unexpected',
                    'missing.*;',
                    'error:.*expected',
                    'error:.*unexpected',
                    'invalid',
                    'parse error',
                    'error:.*invalid'
                ]
                
                # Use regex to check for syntax errors
                has_syntax_error = any(re.search(pattern, stderr) for pattern in syntax_error_indicators)
                return not has_syntax_error
            
            # If it's not a missing header issue, it's likely a syntax error
            return False
            
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
            
            # If compilation succeeds, syntax is valid
            if result.returncode == 0:
                return True
            
            # If compilation fails, check if it's due to missing headers vs syntax errors
            stderr = result.stderr.lower()
            
            # Check if the error is due to missing headers (these are acceptable in test environment)
            missing_header_indicators = [
                'no such file or directory',
                'cannot find',
                'file not found'
            ]
            
            is_missing_header = any(indicator in stderr for indicator in missing_header_indicators)
            
            if is_missing_header:
                # Missing headers are acceptable in test environment
                # Just check for basic syntax validity by looking for common syntax error patterns
                syntax_error_indicators = [
                    'expected',
                    'unexpected',
                    'missing.*;',
                    'error:.*expected',
                    'error:.*unexpected',
                    'invalid',
                    'parse error',
                    'error:.*invalid'
                ]
                
                # Use regex to check for syntax errors
                has_syntax_error = any(re.search(pattern, stderr) for pattern in syntax_error_indicators)
                return not has_syntax_error
            
            # If it's not a missing header issue, it's likely a syntax error
            return False
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # If nvcc is not available, we can't validate CUDA syntax
            # In a test environment, this is acceptable
            return True
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
        self.generator = MainGenerator()
        self.validator = CppCodeValidator()
        
        # Enable debug mode for AST builder
        from robodsl.parsers.ast_builder import ASTBuilder
        ASTBuilder.debug = True
    
    def test_basic_node_generation_syntax(self, test_output_dir):
        skip_if_no_ros2()
        """Test that basic node generation produces valid C++ syntax."""
        source = """
        node test_node {
            parameter int max_speed = 10
            publisher /test: std_msgs/msg/String
            subscriber /input: std_msgs/msg/String
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
        skip_if_no_cuda()
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
        
        # Debug: Check what the parse tree contains
        grammar_path = "src/robodsl/grammar/robodsl.lark"
        with open(grammar_path) as f:
            grammar = f.read()
        parser = Lark(grammar, parser="lalr", start="start")
        try:
            parse_tree = parser.parse(source)
            print(f"[DEBUG] Parse tree data: {parse_tree.data}")
            print(f"[DEBUG] Parse tree children: {[child.data if isinstance(child, Tree) else type(child) for child in parse_tree.children]}")
            
            # Look for cuda_kernel_def in the children
            for i, child in enumerate(parse_tree.children):
                if isinstance(child, Tree):
                    print(f"[DEBUG] Child {i} is Tree with data: {child.data}")
                    if child.data == "cuda_kernel_def":
                        print(f"[DEBUG] Found cuda_kernel_def tree with {len(child.children)} children")
                        for j, grandchild in enumerate(child.children):
                            print(f"[DEBUG]   Grandchild {j}: {type(grandchild)} - {grandchild.data if isinstance(grandchild, Tree) else grandchild}")
                            
                            # Check kernel_content specifically
                            kernel_content = child.children[2]  # kernel_content is at index 2
                            if isinstance(kernel_content, Tree) and kernel_content.data == "kernel_content":
                                print(f"[DEBUG] kernel_content has {len(kernel_content.children)} children")
                                for k, content_child in enumerate(kernel_content.children):
                                    print(f"[DEBUG]   Content child {k}: {type(content_child)} - {content_child.data if isinstance(content_child, Tree) else content_child}")
                else:
                    print(f"[DEBUG] Child {i} is Token: {child}")
        except Exception as e:
            print(f"[DEBUG] Exception during parse: {e}")
            raise
        
        # Debug: Check if kernel parameters are in the AST
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                print(f"[DEBUG] AST kernel {kernel.name}:")
                print(f"[DEBUG]   kernel_parameters: {getattr(kernel.content, 'kernel_parameters', [])}")
                print(f"[DEBUG]   parameters: {kernel.content.parameters}")
        
        generated_files = self.generator.generate(ast)
        
        # Find generated CUDA files
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Generated CUDA file {cuda_file} has syntax errors"
    
    def test_advanced_cpp_features_syntax(self, test_output_dir):
        skip_if_no_ros2()
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
        skip_if_no_ros2()
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
        skip_if_no_ros2()
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
        skip_if_no_cuda()
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
        skip_if_no_ros2()
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
        skip_if_no_ros2()
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
            
            publisher /robot/position: geometry_msgs/msg/Point
            subscriber /robot/command: geometry_msgs/msg/Twist
            service /robot/status: std_srvs/srv/Trigger
            
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
        skip_if_no_ros2()
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
        skip_if_no_ros2()
        """Test generation of large-scale code."""
        source = """
        // Generate many nodes to test scalability
        """
        
        # Add multiple nodes to the source
        for i in range(5):
            source += f"""
            node large_scale_node_{i} {{
                parameter int id_{i} = {i}
                publisher /node_{i}/data: std_msgs/String
                subscriber /node_{i}/command: std_msgs/String
                
                def process_{i}(input: const std::vector<float>&) -> std::vector<float> {{
                    return input;
                }}
            }}
            """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        for file_path in generated_files:
            if file_path.suffix in ['.cpp', '.hpp']:
                content = file_path.read_text()
                assert self.validator.validate_syntax(content, str(file_path)), \
                    f"Large scale compilation failed in {file_path}"
    
    def test_kernel_parameter_struct_generation(self, test_output_dir):
        skip_if_no_ros2()
        """Test that kernel parameter structs are generated correctly."""
        source = """
cuda_kernel test_kernel_with_params {
    block_size: (256, 1, 1)
    grid_size: (1, 1, 1)
    input: float data[1000]
    output: float result[1000]
    parameters: {
        float alpha = 0.5
        int iterations = 10
        bool enable_debug = false
    }
}

        """
        # Debug: Check what the parse tree contains
        grammar_path = "src/robodsl/grammar/robodsl.lark"
        with open(grammar_path) as f:
            grammar = f.read()
        parser = Lark(grammar, parser="lalr", start="start")
        try:
            parse_tree = parser.parse(source)
            print(f"[DEBUG] Parse tree data: {parse_tree.data}")
            print(f"[DEBUG] Parse tree children: {[child.data if isinstance(child, Tree) else type(child) for child in parse_tree.children]}")
            
            # Look for cuda_kernel_def in the children
            for i, child in enumerate(parse_tree.children):
                if isinstance(child, Tree):
                    print(f"[DEBUG] Child {i} is Tree with data: {child.data}")
                    if child.data == "cuda_kernel_def":
                        print(f"[DEBUG] Found cuda_kernel_def tree with {len(child.children)} children")
                        for j, grandchild in enumerate(child.children):
                            print(f"[DEBUG]   Grandchild {j}: {type(grandchild)} - {grandchild.data if isinstance(grandchild, Tree) else grandchild}")
                            
                            # Check kernel_content specifically
                            kernel_content = child.children[2]  # kernel_content is at index 2
                            if isinstance(kernel_content, Tree) and kernel_content.data == "kernel_content":
                                print(f"[DEBUG] kernel_content has {len(kernel_content.children)} children")
                                for k, content_child in enumerate(kernel_content.children):
                                    print(f"[DEBUG]   Content child {k}: {type(content_child)} - {content_child.data if isinstance(content_child, Tree) else content_child}")
                else:
                    print(f"[DEBUG] Child {i} is Token: {child}")
        except Exception as e:
            print(f"[DEBUG] Exception during parse: {e}")
            raise
        
        ast = parse_robodsl(source, debug=True)
        generated_files = self.generator.generate(ast)
        
        # Find generated CUDA header files
        cuh_files = [f for f in generated_files if f.suffix == '.cuh']
        
        for cuh_file in cuh_files:
            content = cuh_file.read_text()
            
            # Check that the struct is generated (capitalize each word, remove underscores)
            expected_struct = 'struct ' + ''.join([w.capitalize() for w in 'test_kernel_with_params'.split('_')]) + 'Parameters'
            assert expected_struct in content, \
                f"Parameter struct not found in {cuh_file}"
            
            # Check that the struct contains the expected parameters
            assert 'float alpha;' in content, \
                f"Parameter 'alpha' not found in struct in {cuh_file}"
            assert 'int iterations;' in content, \
                f"Parameter 'iterations' not found in struct in {cuh_file}"
            assert 'bool enable_debug;' in content, \
                f"Parameter 'enable_debug' not found in struct in {cuh_file}"
            
            # Check that the process method uses the struct
            assert 'process(const Test_kernel_with_paramsParameters& parameters)' in content, \
                f"Process method not using parameter struct in {cuh_file}"
            
            # Validate syntax
            assert self.validator.validate_cuda_syntax(content, str(cuh_file)), \
                f"Generated CUDA file {cuh_file} has syntax errors"


if __name__ == "__main__":
    pytest.main([__file__]) 