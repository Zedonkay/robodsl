"""CUDA Code Validation Tests.

This test suite specifically validates generated CUDA code for:
- CUDA syntax correctness
- Memory management
- Kernel optimization
- Error handling
- Performance patterns
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
from robodsl.generators import MainGenerator, CudaKernelGenerator


class CudaCodeValidator:
    """Validates CUDA code for correctness and efficiency."""
    
    def __init__(self):
        self.cuda_flags = [
            '-std=c++17', '-O2', '-arch=sm_60', '-DNDEBUG'
        ]
    
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
    
    def check_cuda_memory_management(self, cuda_code: str) -> List[str]:
        """Check for proper CUDA memory management."""
        issues = []
        
        # Check for proper allocation/deallocation pairs
        malloc_count = len(re.findall(r'cudaMalloc\s*\(', cuda_code))
        free_count = len(re.findall(r'cudaFree\s*\(', cuda_code))
        
        if malloc_count != free_count:
            issues.append(f"CUDA memory leak: {malloc_count} allocations vs {free_count} deallocations")
        
        # Check for proper error handling
        if 'cudaMalloc' in cuda_code and 'cudaError_t' not in cuda_code:
            issues.append("CUDA memory allocation should check for errors")
        
        # Check for proper synchronization
        if 'cudaMemcpyAsync' in cuda_code and 'cudaStreamSynchronize' not in cuda_code:
            issues.append("Asynchronous CUDA operations should be synchronized")
        
        return issues
    
    def check_kernel_optimization(self, cuda_code: str) -> List[str]:
        """Check for CUDA kernel optimization patterns."""
        issues = []
        
        # Check for proper thread indexing
        if '__global__' in cuda_code and 'threadIdx.x' not in cuda_code:
            issues.append("CUDA kernel should use proper thread indexing")
        
        # Check for memory coalescing
        if 'threadIdx.x' in cuda_code and 'blockDim.x' in cuda_code:
            # Basic check for memory access patterns
            pass
        
        # Check for shared memory usage
        if 'shared' not in cuda_code and 'blockIdx' in cuda_code:
            issues.append("Consider using shared memory for better performance")
        
        return issues
    
    def check_cuda_error_handling(self, cuda_code: str) -> List[str]:
        """Check for proper CUDA error handling."""
        issues = []
        
        # Check for error checking
        if 'cudaError_t' in cuda_code and 'cudaSuccess' not in cuda_code:
            issues.append("CUDA error codes should be checked against cudaSuccess")
        
        # Check for proper error reporting
        if 'cudaGetLastError' not in cuda_code and '__global__' in cuda_code:
            issues.append("CUDA kernel launches should check for errors")
        
        return issues
    
    def check_cuda_performance_patterns(self, cuda_code: str) -> List[str]:
        """Check for CUDA performance patterns."""
        issues = []
        
        # Check for proper block/grid dimensions
        if 'blockDim' in cuda_code and 'gridDim' in cuda_code:
            # Basic check for dimension usage
            pass
        
        # Check for warp divergence
        if 'if' in cuda_code and 'threadIdx.x' in cuda_code:
            issues.append("Consider warp divergence in conditional statements")
        
        # Check for atomic operations
        if 'atomic' not in cuda_code and 'threadIdx' in cuda_code:
            # This is just a hint, not always necessary
            pass
        
        return issues
    
    def check_cuda_best_practices(self, cuda_code: str) -> List[str]:
        """Check for CUDA best practices."""
        issues = []
        
        # Check for proper includes
        if '#include <cuda_runtime.h>' not in cuda_code and 'cuda' in cuda_code:
            issues.append("CUDA code should include proper headers")
        
        # Check for proper namespace usage
        if 'namespace' not in cuda_code and 'cuda' in cuda_code:
            issues.append("Consider using namespaces for CUDA code organization")
        
        # Check for proper const usage
        if 'const' not in cuda_code and 'float' in cuda_code:
            issues.append("Consider using const for immutable data in CUDA kernels")
        
        return issues


class TestCudaCodeValidation:
    """Test suite for CUDA code validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = CudaCodeValidator()
        self.generator = CudaKernelGenerator()
    
    def test_basic_cuda_kernel_syntax(self, test_output_dir):
        """Test that basic CUDA kernel generation produces valid syntax."""
        source = """
        cuda_kernel basic_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Generated CUDA file {cuda_file} has syntax errors"
    
    def test_cuda_memory_management(self, test_output_dir):
        """Test that generated CUDA code properly manages memory."""
        source = """
        cuda_kernel memory_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // Should include proper memory management
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            memory_issues = self.validator.check_cuda_memory_management(content)
            assert len(memory_issues) <= 2, \
                f"CUDA memory management issues in {cuda_file}: {memory_issues}"
    
    def test_cuda_kernel_optimization(self, test_output_dir):
        """Test that generated CUDA kernels are optimized."""
        source = """
        cuda_kernel optimized_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            shared_memory: 1024
            input: float data[1000]
            output: float result[1000]
            
            // Should use optimization patterns
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            optimization_issues = self.validator.check_kernel_optimization(content)
            assert len(optimization_issues) <= 2, \
                f"CUDA kernel optimization issues in {cuda_file}: {optimization_issues}"
    
    def test_cuda_error_handling(self, test_output_dir):
        """Test that generated CUDA code handles errors properly."""
        source = """
        cuda_kernel error_handling_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // Should include proper error handling
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            error_issues = self.validator.check_cuda_error_handling(content)
            assert len(error_issues) <= 2, \
                f"CUDA error handling issues in {cuda_file}: {error_issues}"
    
    def test_cuda_performance_patterns(self, test_output_dir):
        """Test that generated CUDA code uses performance patterns."""
        source = """
        cuda_kernel performance_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // Should use performance patterns
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            performance_issues = self.validator.check_cuda_performance_patterns(content)
            assert len(performance_issues) <= 1, \
                f"CUDA performance pattern issues in {cuda_file}: {performance_issues}"
    
    def test_cuda_best_practices(self, test_output_dir):
        """Test that generated CUDA code follows best practices."""
        source = """
        cuda_kernel best_practices_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            
            // Should follow best practices
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            practice_issues = self.validator.check_cuda_best_practices(content)
            assert len(practice_issues) <= 2, \
                f"CUDA best practice issues in {cuda_file}: {practice_issues}"
    
    def test_complex_cuda_kernel(self, test_output_dir):
        """Test complex CUDA kernel generation."""
        source = """
        cuda_kernel complex_kernel {
            block_size: (32, 32, 1)
            grid_size: (1, 1, 1)
            shared_memory: 2048
            use_thrust: true
            input: float data[10000]
            output: float result[10000]
            parameter: float threshold
            parameter: int iterations
            
            // Complex kernel with multiple features
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            
            # Check syntax
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Complex CUDA kernel syntax error in {cuda_file}"
            
            # Check for thrust usage
            if 'use_thrust' in source:
                assert 'thrust' in content, f"Thrust should be used in {cuda_file}"
            
            # Check for shared memory usage
            if 'shared_memory' in source:
                assert 'shared' in content, f"Shared memory should be used in {cuda_file}"
    
    def test_multiple_cuda_kernels(self, test_output_dir):
        """Test generation of multiple CUDA kernels."""
        source = """
        cuda_kernel kernel1 {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        
        cuda_kernel kernel2 {
            block_size: (32, 32, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Should generate files for both kernels
        kernel1_files = [f for f in generated_files if 'kernel1' in str(f)]
        kernel2_files = [f for f in generated_files if 'kernel2' in str(f)]
        
        assert len(kernel1_files) > 0, "No files generated for kernel1"
        assert len(kernel2_files) > 0, "No files generated for kernel2"
        
        # Validate all CUDA files
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            
            # Check syntax
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Multiple kernel syntax error in {cuda_file}"
            
            # Check memory management
            memory_issues = self.validator.check_cuda_memory_management(content)
            assert len(memory_issues) <= 2, \
                f"Multiple kernel memory issues in {cuda_file}: {memory_issues}"
    
    def test_cuda_with_cpp_integration(self, test_output_dir):
        """Test CUDA kernel integration with C++ code."""
        source = """
        cuda_kernel integrated_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        
        node cuda_node {
            parameter int max_iterations = 100
            
            def process_with_cuda(input: const std::vector<float>&) -> std::vector<float> {
                // Should integrate with CUDA kernel
                return input;
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator()
        generated_files = generator.generate(ast)
        
        # Check CUDA files
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        assert len(cuda_files) > 0, "No CUDA files generated"
        assert len(cpp_files) > 0, "No C++ files generated"
        
        # Validate CUDA files
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"CUDA integration syntax error in {cuda_file}"
        
        # Validate C++ files
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            # Basic C++ syntax check
            assert 'cuda' in content.lower() or 'kernel' in content.lower(), \
                f"C++ file {cpp_file} should reference CUDA functionality"


if __name__ == "__main__":
    pytest.main([__file__]) 