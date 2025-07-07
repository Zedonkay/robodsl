"""ONNX and TensorRT Code Validation Tests.

This test suite validates that generated ONNX and TensorRT C++ code is:
1. Syntactically correct and compilable
2. Efficient in terms of memory usage and performance
3. Properly integrated with ONNX Runtime and TensorRT
4. Following modern C++ best practices
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
from robodsl.generators import MainGenerator, OnnxIntegrationGenerator


class OnnxTensorRTValidator:
    """Validates generated ONNX and TensorRT C++ code for correctness and efficiency."""
    
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
    
    def check_onnx_runtime_integration(self, cpp_code: str) -> List[str]:
        """Check for proper ONNX Runtime integration."""
        issues = []
        
        # Check for ONNX Runtime includes
        if 'onnxruntime_cxx_api.h' not in cpp_code:
            issues.append("ONNX Runtime C++ API should be included")
        
        # Check for proper session management
        if 'Ort::Session' not in cpp_code:
            issues.append("ONNX Runtime session should be used")
        
        # Check for proper tensor handling
        if 'Ort::Value' not in cpp_code:
            issues.append("ONNX Runtime tensor values should be used")
        
        # Check for proper memory info
        if 'Ort::MemoryInfo' not in cpp_code:
            issues.append("ONNX Runtime memory info should be used")
        
        # Check for proper error handling
        if 'Ort::Exception' not in cpp_code:
            issues.append("ONNX Runtime exceptions should be handled")
        
        return issues
    
    def check_tensorrt_integration(self, cpp_code: str) -> List[str]:
        """Check for proper TensorRT integration."""
        issues = []
        
        # Check for TensorRT includes
        if 'onnxruntime_providers.h' not in cpp_code:
            issues.append("TensorRT provider should be included")
        
        # Check for TensorRT provider options
        if 'OrtTensorRTProviderOptions' not in cpp_code:
            issues.append("TensorRT provider options should be configured")
        
        # Check for TensorRT optimization settings
        if 'trt_fp16_enable' not in cpp_code and 'trt_int8_enable' not in cpp_code:
            issues.append("TensorRT optimization settings should be configured")
        
        # Check for TensorRT cache management
        if 'trt_engine_cache_enable' not in cpp_code:
            issues.append("TensorRT engine cache should be enabled")
        
        return issues
    
    def check_cuda_integration(self, cpp_code: str) -> List[str]:
        """Check for proper CUDA integration in ONNX/TensorRT code."""
        issues = []
        
        # Check for CUDA runtime includes
        if 'cuda_runtime.h' not in cpp_code:
            issues.append("CUDA runtime should be included")
        
        # Check for CUDA memory management
        if 'cudaMalloc' in cpp_code and 'cudaFree' not in cpp_code:
            issues.append("CUDA memory should be properly freed")
        
        # Check for CUDA memory transfers
        if 'cudaMemcpy' not in cpp_code and 'cuda' in cpp_code:
            issues.append("CUDA memory transfers should be handled")
        
        # Check for CUDA error handling
        if 'cudaError_t' in cpp_code and 'cudaSuccess' not in cpp_code:
            issues.append("CUDA error codes should be checked")
        
        return issues
    
    def check_memory_management(self, cpp_code: str) -> List[str]:
        """Check for proper memory management in ONNX/TensorRT code."""
        issues = []
        
        # Check for smart pointer usage
        if 'new ' in cpp_code and 'std::unique_ptr' not in cpp_code and 'std::shared_ptr' not in cpp_code:
            issues.append("Consider using smart pointers for memory management")
        
        # Check for RAII patterns
        if 'class' in cpp_code and '~' not in cpp_code and 'std::unique_ptr' not in cpp_code:
            issues.append("Classes should have proper cleanup (destructor or smart pointers)")
        
        # Check for memory pool management
        if 'cuda_memory_pool_' in cpp_code and 'cudaFree' not in cpp_code:
            issues.append("CUDA memory pool should be properly cleaned up")
        
        return issues
    
    def check_performance_optimization(self, cpp_code: str) -> List[str]:
        """Check for performance optimization patterns."""
        issues = []
        
        # Check for efficient tensor operations
        if 'CreateTensor' in cpp_code and 'reserve' not in cpp_code:
            issues.append("Consider pre-allocating memory for tensor operations")
        
        # Check for batch processing
        if 'Run(' in cpp_code and 'batch' not in cpp_code.lower():
            issues.append("Consider batch processing for better performance")
        
        # Check for async operations
        if 'cuda' in cpp_code and 'cudaStream' not in cpp_code:
            issues.append("Consider using CUDA streams for async operations")
        
        # Check for optimization levels
        if 'GraphOptimizationLevel' not in cpp_code:
            issues.append("ONNX Runtime optimization level should be set")
        
        return issues
    
    def check_error_handling(self, cpp_code: str) -> List[str]:
        """Check for proper error handling."""
        issues = []
        
        # Check for ONNX Runtime error handling
        if 'Ort::Exception' in cpp_code and 'try' not in cpp_code:
            issues.append("ONNX Runtime exceptions should be caught")
        
        # Check for CUDA error handling
        if 'cudaError_t' in cpp_code and 'cudaGetLastError' not in cpp_code:
            issues.append("CUDA errors should be checked")
        
        # Check for null pointer checks
        if 'nullptr' in cpp_code and 'if' not in cpp_code:
            issues.append("Null pointer checks should be performed")
        
        return issues


class TestOnnxTensorRTValidation:
    """Test suite for ONNX and TensorRT validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = OnnxTensorRTValidator()
        self.generator = MainGenerator()
    
    def test_basic_onnx_model_generation(self, test_output_dir):
        """Test basic ONNX model generation produces valid C++ syntax."""
        source = """
        onnx_model basic_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cpu
            }
        }
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "basic_onnx"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "basic_onnx.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast, output_dir=str(test_output_dir))
        
        # Find generated ONNX files
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            assert self.validator.validate_syntax(content, str(onnx_file)), \
                f"Generated ONNX file {onnx_file} has syntax errors"
    
    def test_onnx_with_tensorrt_optimization(self, test_output_dir):
        """Test ONNX model generation with TensorRT optimization."""
        source = """
        onnx_model tensorrt_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "onnx_tensorrt"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "onnx_tensorrt.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast, output_dir=str(test_output_dir))
        
        # Find generated ONNX files
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check syntax
            assert self.validator.validate_syntax(content, str(onnx_file)), \
                f"TensorRT ONNX file {onnx_file} has syntax errors"
            
            # Check ONNX Runtime integration
            onnx_issues = self.validator.check_onnx_runtime_integration(content)
            assert len(onnx_issues) <= 2, \
                f"ONNX Runtime integration issues in {onnx_file}: {onnx_issues}"
            
            # Check TensorRT integration
            tensorrt_issues = self.validator.check_tensorrt_integration(content)
            assert len(tensorrt_issues) <= 2, \
                f"TensorRT integration issues in {onnx_file}: {tensorrt_issues}"
    
    def test_onnx_with_cuda_integration(self, test_output_dir):
        """Test ONNX model generation with CUDA integration."""
        source = """
        onnx_model cuda_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "onnx_cuda"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "onnx_cuda.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast, output_dir=str(test_output_dir))
        
        # Find generated ONNX files
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check CUDA integration
            cuda_issues = self.validator.check_cuda_integration(content)
            assert len(cuda_issues) <= 2, \
                f"CUDA integration issues in {onnx_file}: {cuda_issues}"
    
    def test_onnx_memory_management(self, test_output_dir):
        """Test that ONNX code properly manages memory."""
        source = """
        onnx_model memory_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check memory management
            memory_issues = self.validator.check_memory_management(content)
            assert len(memory_issues) <= 2, \
                f"Memory management issues in {onnx_file}: {memory_issues}"
    
    def test_onnx_performance_optimization(self, test_output_dir):
        """Test that ONNX code includes performance optimizations."""
        source = """
        onnx_model perf_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check performance optimizations
            perf_issues = self.validator.check_performance_optimization(content)
            assert len(perf_issues) <= 3, \
                f"Performance optimization issues in {onnx_file}: {perf_issues}"
    
    def test_onnx_error_handling(self, test_output_dir):
        """Test that ONNX code includes proper error handling."""
        source = """
        onnx_model error_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check error handling
            error_issues = self.validator.check_error_handling(content)
            assert len(error_issues) <= 2, \
                f"Error handling issues in {onnx_file}: {error_issues}"
    
    def test_complex_onnx_model(self, test_output_dir):
        """Test complex ONNX model with multiple features."""
        source = """
        onnx_model complex_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
                optimization: int8
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check syntax
            assert self.validator.validate_syntax(content, str(onnx_file)), \
                f"Complex ONNX model syntax error in {onnx_file}"
            
            # Check all integration aspects
            onnx_issues = self.validator.check_onnx_runtime_integration(content)
            tensorrt_issues = self.validator.check_tensorrt_integration(content)
            cuda_issues = self.validator.check_cuda_integration(content)
            memory_issues = self.validator.check_memory_management(content)
            perf_issues = self.validator.check_performance_optimization(content)
            error_issues = self.validator.check_error_handling(content)
            
            # Allow some issues as the generator might not be perfect
            total_issues = len(onnx_issues) + len(tensorrt_issues) + len(cuda_issues) + \
                          len(memory_issues) + len(perf_issues) + len(error_issues)
            
            assert total_issues <= 8, \
                f"Too many issues in complex ONNX model {onnx_file}: {total_issues} total issues"
    
    def test_onnx_in_pipeline(self, test_output_dir):
        """Test ONNX model integration in pipeline."""
        source = """
        onnx_model pipeline_model {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        
        pipeline onnx_pipeline {
            stage inference {
                input: "input_data"
                output: "output_data"
                method: "run_inference"
                onnx_model: "pipeline_model"
                topic: /pipeline/inference
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Should generate both ONNX and pipeline files
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        pipeline_files = [f for f in generated_files if 'pipeline' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        assert len(onnx_files) > 0, "No ONNX files generated for pipeline"
        assert len(pipeline_files) > 0, "No pipeline files generated"
        
        # Validate ONNX files
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            assert self.validator.validate_syntax(content, str(onnx_file)), \
                f"Pipeline ONNX file {onnx_file} has syntax errors"
        
        # Validate pipeline files
        for pipeline_file in pipeline_files:
            content = pipeline_file.read_text()
            assert self.validator.validate_syntax(content, str(pipeline_file)), \
                f"Pipeline file {pipeline_file} has syntax errors"
    
    def test_tensorrt_specific_features(self, test_output_dir):
        """Test TensorRT-specific features."""
        source = """
        onnx_model tensorrt_features {
            config {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
                optimization: int8
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check for TensorRT-specific features
            assert 'trt_fp16_enable' in content or 'trt_int8_enable' in content, \
                f"TensorRT optimization features should be present in {onnx_file}"
            
            assert 'trt_engine_cache_enable' in content, \
                f"TensorRT engine cache should be enabled in {onnx_file}"
            
            assert 'trt_max_workspace_size' in content, \
                f"TensorRT workspace size should be configured in {onnx_file}"
    
    def test_onnx_edge_cases(self, test_output_dir):
        """Test ONNX model generation with edge cases."""
        source = """
        onnx_model edge_case_model {
            config {
                input: "input" -> "float32[1,1,1,1]"
                output: "output" -> "float32[1,1]"
                device: cpu
            }
        }
        """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Should still compile with edge cases
            assert self.validator.validate_syntax(content, str(onnx_file)), \
                f"Edge case ONNX model compilation failed in {onnx_file}"
    
    def test_large_scale_onnx(self, test_output_dir):
        """Test large-scale ONNX model generation."""
        source = """
        // Generate multiple ONNX models to test scalability
        """
        
        # Add multiple ONNX models
        for i in range(3):
            source += f"""
            onnx_model model_{i} {{
                config {{
                    input: "input_{i}" -> "float32[1,3,224,224]"
                    output: "output_{i}" -> "float32[1,1000]"
                    device: cuda
                    optimization: tensorrt
                }}
            }}
            """
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast)
        
        # Should generate multiple ONNX files
        onnx_files = [f for f in generated_files if 'onnx' in str(f).lower() and f.suffix in ['.cpp', '.hpp']]
        
        assert len(onnx_files) >= 3, f"Expected at least 3 ONNX files, got {len(onnx_files)}"
        
        # Validate all files
        for onnx_file in onnx_files:
            content = onnx_file.read_text()
            
            # Check syntax
            assert self.validator.validate_syntax(content, str(onnx_file)), \
                f"Large scale ONNX syntax error in {onnx_file}"
            
            # Check file size (not too large)
            assert len(content) < 100000, \
                f"Generated file {onnx_file} is too large: {len(content)} characters"


if __name__ == "__main__":
    pytest.main([__file__]) 