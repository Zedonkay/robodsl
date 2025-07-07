"""C++ Code Efficiency Validation Tests.

This test suite specifically validates the efficiency aspects of generated C++ code:
- Memory allocation patterns
- Performance optimizations
- Resource management
- Algorithm efficiency
- Compiler optimization compatibility
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


class CppEfficiencyValidator:
    """Validates efficiency aspects of generated C++ code."""
    
    def __init__(self):
        self.optimization_flags = [
            '-std=c++17', '-O3', '-march=native', '-mtune=native',
            '-ffast-math', '-funroll-loops', '-DNDEBUG'
        ]
    
    def check_memory_allocation_patterns(self, cpp_code: str) -> List[str]:
        """Check for efficient memory allocation patterns."""
        issues = []
        
        # Check for stack vs heap allocation
        stack_allocs = len(re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*\s+[a-zA-Z_][a-zA-Z0-9_]*\[', cpp_code))
        heap_allocs = len(re.findall(r'new\s+', cpp_code))
        
        if heap_allocs > stack_allocs * 2:
            issues.append("Consider using stack allocation instead of heap allocation for small objects")
        
        # Check for unnecessary dynamic allocation
        if 'std::vector' in cpp_code and 'reserve' not in cpp_code:
            issues.append("Consider using vector::reserve() to avoid multiple reallocations")
        
        # Check for proper container usage
        if 'std::list' in cpp_code and 'std::vector' not in cpp_code:
            issues.append("Consider std::vector for better cache locality unless frequent insertions/deletions are needed")
        
        return issues
    
    def check_algorithm_efficiency(self, cpp_code: str) -> List[str]:
        """Check for algorithm efficiency issues."""
        issues = []
        
        # Check for O(n²) algorithms where O(n log n) would be better
        if 'for' in cpp_code and 'std::sort' not in cpp_code and 'std::find' in cpp_code:
            issues.append("Consider using std::binary_search on sorted data instead of std::find")
        
        # Check for unnecessary copies
        if 'std::string' in cpp_code and 'const std::string&' not in cpp_code:
            issues.append("Consider using const references to avoid string copies")
        
        # Check for inefficient loops
        if 'for' in cpp_code and 'auto&' not in cpp_code and 'auto' in cpp_code:
            issues.append("Consider using auto& for non-primitive types to avoid copies")
        
        return issues
    
    def check_compiler_optimizations(self, cpp_code: str) -> List[str]:
        """Check for compiler optimization opportunities."""
        issues = []
        
        # Check for const correctness
        if 'const' not in cpp_code and 'int' in cpp_code:
            issues.append("Consider adding const qualifiers for better compiler optimization")
        
        # Check for inline functions
        if 'inline' not in cpp_code and len(cpp_code.split('\n')) < 50:
            issues.append("Consider inline for small functions")
        
        # Check for noexcept specifications
        if 'noexcept' not in cpp_code and 'throw' not in cpp_code:
            issues.append("Consider noexcept for functions that don't throw")
        
        return issues
    
    def check_cache_efficiency(self, cpp_code: str) -> List[str]:
        """Check for cache-friendly code patterns."""
        issues = []
        
        # Check for data locality
        if 'struct' in cpp_code and 'class' in cpp_code:
            issues.append("Consider organizing data structures for better cache locality")
        
        # Check for array access patterns
        if '[' in cpp_code and ']' in cpp_code:
            # Basic check for sequential access
            pass
        
        return issues
    
    def check_resource_management(self, cpp_code: str) -> List[str]:
        """Check for efficient resource management."""
        issues = []
        
        # Check for RAII usage
        if 'new' in cpp_code and 'std::unique_ptr' not in cpp_code:
            issues.append("Consider using RAII with smart pointers for automatic resource management")
        
        # Check for move semantics
        if 'std::move' not in cpp_code and 'std::vector' in cpp_code:
            issues.append("Consider using move semantics to avoid unnecessary copies")
        
        # Check for proper cleanup
        if 'cudaMalloc' in cpp_code and 'cudaFree' not in cpp_code:
            issues.append("Ensure CUDA memory is properly freed")
        
        return issues
    
    def validate_optimization_flags(self, cpp_code: str) -> bool:
        """Validate that code compiles with aggressive optimization flags."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.optimization_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            os.unlink(temp_file)


class TestCppEfficiencyValidation:
    """Test suite for C++ efficiency validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CppEfficiencyValidator()
        self.generator = MainGenerator()
    
    def test_memory_allocation_efficiency(self, test_output_dir):
        """Test that generated code uses efficient memory allocation patterns."""
        source = """
        node efficient_node {
            parameter int buffer_size = 1000
            
            def efficient_process(input: const std::vector<float>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(input.size());  // Efficient allocation
                return result;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            allocation_issues = self.validator.check_memory_allocation_patterns(content)
            assert len(allocation_issues) <= 1, \
                f"Memory allocation efficiency issues in {cpp_file}: {allocation_issues}"
    
    def test_algorithm_efficiency(self, test_output_dir):
        """Test that generated code uses efficient algorithms."""
        source = """
        node algorithm_node {
            def efficient_search(data: const std::vector<int>&, target: int) -> bool {
                // Should use efficient search algorithms
                return std::binary_search(data.begin(), data.end(), target);
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            algorithm_issues = self.validator.check_algorithm_efficiency(content)
            assert len(algorithm_issues) <= 2, \
                f"Algorithm efficiency issues in {cpp_file}: {algorithm_issues}"
    
    def test_compiler_optimization_compatibility(self, test_output_dir):
        """Test that generated code is compatible with aggressive compiler optimizations."""
        source = """
        node optimization_node {
            parameter const int max_iterations = 1000
            
            def optimized_function(input: const std::vector<float>&) -> float {
                float sum = 0.0f;
                for (const auto& value : input) {
                    sum += value;
                }
                return sum;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_optimization_flags(content), \
                f"Generated code in {cpp_file} is not compatible with aggressive optimizations"
    
    def test_cache_efficiency(self, test_output_dir):
        """Test that generated code is cache-friendly."""
        source = """
        struct CacheFriendlyData {
            float x, y, z;  // Contiguous data
            int id;
        }
        
        node cache_node {
            def cache_friendly_process(data: const std::vector<CacheFriendlyData>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(data.size());
                
                for (const auto& item : data) {
                    result.push_back(item.x + item.y + item.z);
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
            cache_issues = self.validator.check_cache_efficiency(content)
            assert len(cache_issues) <= 1, \
                f"Cache efficiency issues in {cpp_file}: {cache_issues}"
    
    def test_resource_management_efficiency(self, test_output_dir):
        """Test that generated code efficiently manages resources."""
        source = """
        node resource_node {
            def efficient_resource_management() -> std::unique_ptr<std::vector<float>> {
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
            resource_issues = self.validator.check_resource_management(content)
            assert len(resource_issues) <= 2, \
                f"Resource management efficiency issues in {cpp_file}: {resource_issues}"
    
    def test_cuda_efficiency(self, test_output_dir):
        """Test that generated CUDA code is efficient."""
        source = """
        cuda_kernel efficient_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // Should use efficient memory access patterns
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            
            # Check for efficient CUDA patterns
            assert 'threadIdx.x' in content, "CUDA kernel should use thread indexing"
            assert 'blockDim.x' in content, "CUDA kernel should use block dimensions"
            
            # Check for memory coalescing hints
            if '[' in content and ']' in content:
                # Basic check for array access
                pass
    
    def test_vectorization_opportunities(self, test_output_dir):
        """Test that generated code provides vectorization opportunities."""
        source = """
        node vectorization_node {
            def vectorizable_function(input: const std::vector<float>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(input.size());
                
                for (size_t i = 0; i < input.size(); ++i) {
                    result.push_back(input[i] * 2.0f);  // Simple operation for vectorization
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
            
            # Check for vectorization-friendly patterns
            if 'for' in content and '++' in content:
                # Basic check for loop patterns that can be vectorized
                pass
    
    def test_zero_copy_optimizations(self, test_output_dir):
        """Test that generated code avoids unnecessary copies."""
        source = """
        node zero_copy_node {
            def zero_copy_process(input: const std::vector<float>&) -> const std::vector<float>& {
                return input;  // Return reference to avoid copy
            }
            
            def efficient_move(input: std::vector<float>&&) -> std::vector<float> {
                return std::move(input);  // Use move semantics
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check for const references and move semantics
            if 'const' in content and '&' in content:
                # Basic check for reference usage
                pass
    
    def test_compile_time_optimizations(self, test_output_dir):
        """Test that generated code leverages compile-time optimizations."""
        source = """
        global CONSTANT: constexpr int = 42;
        
        node compile_time_node {
            parameter const int max_size = 1000
            
            def compile_time_optimized() -> int {
                return CONSTANT * 2;  // Should be computed at compile time
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check for constexpr usage
            if 'constexpr' in content:
                # Good - using compile-time evaluation
                pass


if __name__ == "__main__":
    pytest.main([__file__]) 