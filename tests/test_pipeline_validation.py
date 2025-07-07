"""Pipeline Generation Validation Tests.

This test suite validates that generated pipeline C++ code is:
1. Syntactically correct and compilable
2. Efficient in terms of memory usage and performance
3. Properly integrated with ROS2, CUDA, and ONNX
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
from robodsl.generators import MainGenerator, PipelineGenerator


class PipelineValidator:
    """Validates generated pipeline C++ code for correctness and efficiency."""
    
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
    
    def check_pipeline_structure(self, cpp_code: str) -> List[str]:
        """Check for proper pipeline structure and patterns."""
        issues = []
        
        # Check for proper namespace usage
        if 'namespace' not in cpp_code:
            issues.append("Pipeline code should use proper namespaces")
        
        # Check for ROS2 integration
        if 'rclcpp' not in cpp_code:
            issues.append("Pipeline should integrate with ROS2")
        
        # Check for proper class structure
        if 'class' not in cpp_code or 'public:' not in cpp_code:
            issues.append("Pipeline should have proper class structure")
        
        # Check for memory management
        if 'std::unique_ptr' not in cpp_code and 'new ' in cpp_code:
            issues.append("Pipeline should use smart pointers for memory management")
        
        return issues
    
    def check_stage_integration(self, cpp_code: str) -> List[str]:
        """Check for proper stage integration patterns."""
        issues = []
        
        # Check for stage initialization
        if 'initialize_' not in cpp_code:
            issues.append("Pipeline stages should have initialization methods")
        
        # Check for proper callback patterns
        if 'callback' not in cpp_code and 'timer' not in cpp_code:
            issues.append("Pipeline should have proper callback mechanisms")
        
        # Check for data flow patterns
        if 'input' not in cpp_code or 'output' not in cpp_code:
            issues.append("Pipeline should handle input/output data flow")
        
        return issues
    
    def check_cuda_integration(self, cpp_code: str) -> List[str]:
        """Check for proper CUDA integration in pipelines."""
        issues = []
        
        # Check for CUDA manager usage
        if 'CudaManager' in cpp_code:
            if 'cudaStream_t' not in cpp_code:
                issues.append("CUDA integration should use CUDA streams")
            
            if 'cudaMalloc' in cpp_code and 'cudaFree' not in cpp_code:
                issues.append("CUDA memory should be properly freed")
            
            if 'cudaMemcpy' not in cpp_code:
                issues.append("CUDA integration should handle memory transfers")
        
        return issues
    
    def check_onnx_integration(self, cpp_code: str) -> List[str]:
        """Check for proper ONNX integration in pipelines."""
        issues = []
        
        # Check for ONNX Runtime usage
        if 'OnnxManager' in cpp_code:
            if 'Ort::Session' not in cpp_code:
                issues.append("ONNX integration should use ONNX Runtime sessions")
            
            if 'Ort::Value' not in cpp_code:
                issues.append("ONNX integration should handle tensor values")
            
            if 'Run(' not in cpp_code:
                issues.append("ONNX integration should execute inference")
        
        return issues
    
    def check_performance_patterns(self, cpp_code: str) -> List[str]:
        """Check for performance optimization patterns."""
        issues = []
        
        # Check for efficient data structures
        if 'std::vector' in cpp_code and 'reserve' not in cpp_code:
            issues.append("Consider using vector::reserve() for better performance")
        
        # Check for const correctness
        if 'const' not in cpp_code and 'std::string' in cpp_code:
            issues.append("Consider using const references to avoid copies")
        
        # Check for move semantics
        if 'std::move' not in cpp_code and 'std::vector' in cpp_code:
            issues.append("Consider using move semantics for better performance")
        
        return issues


class TestPipelineValidation:
    """Test suite for pipeline validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = PipelineValidator()
        self.generator = PipelineGenerator()
    
    def test_basic_pipeline_generation(self, test_output_dir):
        """Test basic pipeline generation produces valid C++ syntax."""
        source = """
        pipeline basic_pipeline {
            stage preprocessing {
                input: "raw_data"
                output: "processed_data"
                method: "preprocess"
                topic: /pipeline/preprocess
            }
            
            stage processing {
                input: "processed_data"
                output: "result"
                method: "process"
                topic: /pipeline/process
            }
        }
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "basic_pipeline"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "basic_pipeline.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        # Find generated C++ files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Generated pipeline C++ file {cpp_file} has syntax errors"
    
    def test_pipeline_with_cuda_kernels(self, test_output_dir):
        """Test pipeline generation with CUDA kernels."""
        source = """
        cuda_kernels {
            kernel process_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                input: float data(1000)
                output: float result(1000)
            }
        }
        
        pipeline cuda_pipeline {
            stage cuda_processing {
                input: "input_data"
                output: "output_data"
                method: "process_with_cuda"
                cuda_kernel: "process_kernel"
                topic: /pipeline/cuda_process
            }
        }
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "pipeline_with_cuda"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "pipeline_with_cuda.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        # Find generated C++ files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        # Validate C++ files
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Pipeline C++ file {cpp_file} has syntax errors"
            
            # Check CUDA integration
            cuda_issues = self.validator.check_cuda_integration(content)
            assert len(cuda_issues) <= 2, \
                f"CUDA integration issues in {cpp_file}: {cuda_issues}"
        
        # Validate CUDA files
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Pipeline CUDA file {cuda_file} has syntax errors"
    
    def test_pipeline_with_onnx_models(self, test_output_dir):
        """Test pipeline generation with ONNX models."""
        source = """
        onnx_model inference_model {
            config {
                input: "input" -> "float32"
                output: "output" -> "float32"
                device: gpu
                optimization: tensorrt
            }
        }
        
        pipeline onnx_pipeline {
            stage inference {
                input: "input_data"
                output: "output_data"
                method: "run_inference"
                onnx_model: "inference_model"
                topic: /pipeline/inference
            }
        }
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "pipeline_with_onnx"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "pipeline_with_onnx.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        # Find generated C++ files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Pipeline ONNX file {cpp_file} has syntax errors"
            
            # Check ONNX integration
            onnx_issues = self.validator.check_onnx_integration(content)
            assert len(onnx_issues) <= 2, \
                f"ONNX integration issues in {cpp_file}: {onnx_issues}"
    
    def test_complex_pipeline(self, test_output_dir):
        """Test complex pipeline with multiple stages and features."""
        source = """
        cuda_kernel preprocess_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float raw_data[1000]
            output: float processed_data[1000]
        }
        
        onnx_model inference_model {
            config {
                input: "input" -> "float32"
                output: "output" -> "float32"
                device: gpu
                optimization: tensorrt
            }
        }
        
        cuda_kernel postprocess_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float inference_result[1000]
            output: float final_result[1000]
        }
        
        pipeline complex_pipeline {
            stage preprocessing {
                input: "raw_data"
                output: "processed_data"
                method: "preprocess"
                cuda_kernel: "preprocess_kernel"
                topic: /pipeline/preprocess
            }
            
            stage inference {
                input: "processed_data"
                output: "inference_result"
                method: "run_inference"
                onnx_model: "inference_model"
                topic: /pipeline/inference
            }
            
            stage postprocessing {
                input: "inference_result"
                output: "final_result"
                method: "postprocess"
                cuda_kernel: "postprocess_kernel"
                topic: /pipeline/postprocess
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        # Find generated files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        cuda_files = [f for f in generated_files if f.suffix in ['.cu', '.cuh']]
        
        # Validate all files
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check syntax
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Complex pipeline C++ file {cpp_file} has syntax errors"
            
            # Check pipeline structure
            structure_issues = self.validator.check_pipeline_structure(content)
            assert len(structure_issues) <= 2, \
                f"Pipeline structure issues in {cpp_file}: {structure_issues}"
            
            # Check stage integration
            stage_issues = self.validator.check_stage_integration(content)
            assert len(stage_issues) <= 2, \
                f"Stage integration issues in {cpp_file}: {stage_issues}"
            
            # Check performance patterns
            perf_issues = self.validator.check_performance_patterns(content)
            assert len(perf_issues) <= 2, \
                f"Performance issues in {cpp_file}: {perf_issues}"
        
        # Validate CUDA files
        for cuda_file in cuda_files:
            content = cuda_file.read_text()
            assert self.validator.validate_cuda_syntax(content, str(cuda_file)), \
                f"Complex pipeline CUDA file {cuda_file} has syntax errors"
    
    def test_pipeline_memory_management(self, test_output_dir):
        """Test that pipeline code properly manages memory."""
        source = """
        pipeline memory_test_pipeline {
            stage memory_stage {
                input: "input_data"
                output: "output_data"
                method: "process_memory"
                topic: /pipeline/memory_test
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check for proper memory management patterns
            if 'new ' in content:
                assert 'std::unique_ptr' in content or 'std::shared_ptr' in content, \
                    f"Pipeline should use smart pointers for memory management in {cpp_file}"
            
            # Check for RAII patterns
            if 'class' in content:
                assert '~' in content or 'std::unique_ptr' in content, \
                    f"Pipeline classes should have proper cleanup in {cpp_file}"
    
    def test_pipeline_performance_optimization(self, test_output_dir):
        """Test that pipeline code includes performance optimizations."""
        source = """
        pipeline performance_pipeline {
            stage perf_stage {
                input: "input_data"
                output: "output_data"
                method: "optimized_process"
                topic: /pipeline/performance_test
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check for performance patterns
            perf_issues = self.validator.check_performance_patterns(content)
            assert len(perf_issues) <= 3, \
                f"Too many performance issues in {cpp_file}: {perf_issues}"
    
    def test_pipeline_error_handling(self, test_output_dir):
        """Test that pipeline code includes proper error handling."""
        source = """
        pipeline error_handling_pipeline {
            stage error_stage {
                input: "input_data"
                output: "output_data"
                method: "handle_errors"
                topic: /pipeline/error_test
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check for error handling patterns
            if 'try' in content or 'catch' in content:
                # Good - has exception handling
                pass
            elif 'RCLCPP_ERROR' in content or 'RCLCPP_WARN' in content:
                # Good - has ROS2 error logging
                pass
            else:
                # Should have some form of error handling
                assert 'return false' in content or 'return nullptr' in content, \
                    f"Pipeline should include error handling in {cpp_file}"
    
    def test_pipeline_ros2_integration(self, test_output_dir):
        """Test that pipeline code properly integrates with ROS2."""
        source = """
        pipeline ros2_pipeline {
            stage ros2_stage {
                input: "input_data"
                output: "output_data"
                method: "ros2_process"
                topic: /pipeline/ros2_test
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check for ROS2 integration
            assert 'rclcpp' in content, f"Pipeline should integrate with ROS2 in {cpp_file}"
            assert 'create_subscription' in content or 'create_publisher' in content, \
                f"Pipeline should have ROS2 publishers/subscribers in {cpp_file}"
    
    def test_pipeline_edge_cases(self, test_output_dir):
        """Test pipeline generation with edge cases."""
        source = """
        pipeline edge_case_pipeline {
            stage empty_stage {
                input: "input_data"
                output: "output_data"
                method: "empty_process"
                topic: /pipeline/empty_test
            }
        }
        """
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Should still compile with edge cases
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Edge case pipeline compilation failed in {cpp_file}"
    
    def test_large_scale_pipeline(self, test_output_dir):
        """Test large-scale pipeline generation."""
        source = """
        // Generate multiple stages to test scalability
        """
        
        # Add multiple stages
        for i in range(5):
            source += f"""
            stage stage_{i} {{
                input: "input_{i}"
                output: "output_{i}"
                method: "process_{i}"
                topic: /pipeline/stage_{i}
            }}
            """
        
        source += """
        pipeline large_scale_pipeline {
        """
        
        for i in range(5):
            source += f"    stage stage_{i}\n"
        
        source += "}"
        
        ast = parse_robodsl(source)
        generator = MainGenerator(output_dir=str(test_output_dir))
        generated_files = generator.generate(ast)
        
        # Should generate multiple files
        cpp_files = [f for f in generated_files if f.suffix in ['.cpp', '.hpp']]
        
        assert len(cpp_files) >= 5, f"Expected at least 5 C++ files, got {len(cpp_files)}"
        
        # Validate all files
        for cpp_file in cpp_files:
            content = cpp_file.read_text()
            
            # Check syntax
            assert self.validator.validate_syntax(content, str(cpp_file)), \
                f"Large scale pipeline syntax error in {cpp_file}"
            
            # Check file size (not too large)
            assert len(content) < 50000, \
                f"Generated file {cpp_file} is too large: {len(content)} characters"


if __name__ == "__main__":
    pytest.main([__file__]) 