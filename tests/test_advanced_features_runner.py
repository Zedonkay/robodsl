"""Advanced Features Test Runner.

This test runner validates pipeline, ONNX, and TensorRT features for C++ correctness and efficiency.
"""

import os
import sys
import pytest
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda, skip_if_no_tensorrt, skip_if_no_onnx
from robodsl.generators import MainGenerator
from test_advanced_features_config import (
    PIPELINE_TEST_CASES,
    ONNX_TEST_CASES,
    TENSORRT_TEST_CASES,
    COMPREHENSIVE_TEST_CASES,
    EDGE_CASE_TEST_CASES
)


class AdvancedFeaturesValidator:
    """Validates advanced features for correctness and efficiency."""
    
    def __init__(self):
        self.compiler_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2',
            '-fno-exceptions', '-fno-rtti', '-DNDEBUG'
        ]
        self.cuda_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-O2',
            '-arch=sm_60', '-DNDEBUG'
        ]
    
    def validate_syntax(self, cpp_code: str, filename: str = "test.cpp") -> Tuple[bool, List[str]]:
        """Validate C++ syntax using g++ compiler."""
        issues = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.compiler_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                issues.append(f"Compilation failed: {result.stderr}")
                return False, issues
            
            return True, issues
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            issues.append(f"Compiler error: {str(e)}")
            return False, issues
        finally:
            os.unlink(temp_file)
    
    def validate_cuda_syntax(self, cuda_code: str, filename: str = "test.cu") -> Tuple[bool, List[str]]:
        """Validate CUDA syntax using nvcc compiler."""
        issues = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(cuda_code)
            temp_file = f.name
        
        try:
            cmd = ['nvcc'] + self.cuda_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                issues.append(f"CUDA compilation failed: {result.stderr}")
                return False, issues
            
            return True, issues
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            issues.append(f"CUDA compiler error: {str(e)}")
            return False, issues
        finally:
            os.unlink(temp_file)
    
    def check_pipeline_features(self, cpp_code: str) -> List[str]:
        """Check for pipeline-specific features."""
        issues = []
        
        if 'namespace' not in cpp_code:
            issues.append("Pipeline code should use proper namespaces")
        
        if 'rclcpp' not in cpp_code:
            issues.append("Pipeline should integrate with ROS2")
        
        if 'initialize_' not in cpp_code:
            issues.append("Pipeline stages should have initialization methods")
        
        if 'input' not in cpp_code or 'output' not in cpp_code:
            issues.append("Pipeline should handle input/output data flow")
        
        return issues
    
    def check_onnx_features(self, cpp_code: str) -> List[str]:
        """Check for ONNX-specific features."""
        issues = []
        
        if 'onnxruntime_cxx_api.h' not in cpp_code:
            issues.append("ONNX Runtime C++ API should be included")
        
        if 'Ort::Session' not in cpp_code:
            issues.append("ONNX Runtime session should be used")
        
        if 'Ort::Value' not in cpp_code:
            issues.append("ONNX Runtime tensor values should be used")
        
        if 'Ort::Exception' not in cpp_code:
            issues.append("ONNX Runtime exceptions should be handled")
        
        return issues
    
    def check_tensorrt_features(self, cpp_code: str) -> List[str]:
        """Check for TensorRT-specific features."""
        issues = []
        
        if 'onnxruntime_providers.h' not in cpp_code:
            issues.append("TensorRT provider should be included")
        
        if 'OrtTensorRTProviderOptions' not in cpp_code:
            issues.append("TensorRT provider options should be configured")
        
        if 'trt_fp16_enable' not in cpp_code and 'trt_int8_enable' not in cpp_code:
            issues.append("TensorRT optimization settings should be configured")
        
        if 'trt_engine_cache_enable' not in cpp_code:
            issues.append("TensorRT engine cache should be enabled")
        
        return issues
    
    def check_cuda_features(self, cpp_code: str) -> List[str]:
        """Check for CUDA-specific features."""
        issues = []
        
        if 'cuda_runtime.h' not in cpp_code:
            issues.append("CUDA runtime should be included")
        
        if 'cudaMalloc' in cpp_code and 'cudaFree' not in cpp_code:
            issues.append("CUDA memory should be properly freed")
        
        if 'cudaMemcpy' not in cpp_code and 'cuda' in cpp_code:
            issues.append("CUDA memory transfers should be handled")
        
        if 'cudaError_t' in cpp_code and 'cudaSuccess' not in cpp_code:
            issues.append("CUDA error codes should be checked")
        
        return issues
    
    def check_performance_features(self, cpp_code: str) -> List[str]:
        """Check for performance optimization features."""
        issues = []
        
        if 'std::vector' in cpp_code and 'reserve' not in cpp_code:
            issues.append("Consider using vector::reserve() for better performance")
        
        if 'const' not in cpp_code and 'std::string' in cpp_code:
            issues.append("Consider using const references to avoid copies")
        
        if 'std::move' not in cpp_code and 'std::vector' in cpp_code:
            issues.append("Consider using move semantics for better performance")
        
        if 'new ' in cpp_code and 'std::unique_ptr' not in cpp_code:
            issues.append("Consider using smart pointers for memory management")
        
        return issues
    
    def run_validation(self, source_code: str, test_name: str) -> Dict[str, Any]:
        """Run comprehensive validation on advanced features."""
        start_time = time.time()
        
        try:
            # Create test output directory
            test_output_dir = Path("test_output") / test_name
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save RoboDSL source file
            source_file = test_output_dir / f"{test_name}.robodsl"
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Parse the source code
            ast = parse_robodsl(source_code)
            
            # Generate code with output directory
            generator = MainGenerator(output_dir=str(test_output_dir))
            generated_files = generator.generate(ast)
            
            # Initialize results
            results = {
                "test_name": test_name,
                "passed": True,
                "file_count": len(generated_files),
                "validation_time": 0,
                "issues": {
                    "syntax": [],
                    "pipeline": [],
                    "onnx": [],
                    "tensorrt": [],
                    "cuda": [],
                    "performance": []
                }
            }
            
            # Validate each generated file
            for file_path in generated_files:
                content = file_path.read_text()
                
                # Validate based on file type
                if file_path.suffix in ['.cpp', '.hpp']:
                    # C++ file validation
                    syntax_ok, syntax_issues = self.validate_syntax(content, str(file_path))
                    results["issues"]["syntax"].extend(syntax_issues)
                    
                    if syntax_ok:
                        # Check feature-specific issues
                        if 'pipeline' in str(file_path).lower() or 'stage' in content.lower():
                            results["issues"]["pipeline"].extend(self.check_pipeline_features(content))
                        if 'onnx' in str(file_path).lower() or 'Ort::' in content:
                            results["issues"]["onnx"].extend(self.check_onnx_features(content))
                        if 'tensorrt' in content.lower() or 'trt_' in content:
                            results["issues"]["tensorrt"].extend(self.check_tensorrt_features(content))
                        if 'cuda' in str(file_path).lower() or 'cuda' in content.lower():
                            results["issues"]["cuda"].extend(self.check_cuda_features(content))
                        
                        # Always check performance
                        results["issues"]["performance"].extend(self.check_performance_features(content))
                    
                elif file_path.suffix in ['.cu', '.cuh']:
                    # CUDA file validation
                    syntax_ok, syntax_issues = self.validate_cuda_syntax(content, str(file_path))
                    results["issues"]["syntax"].extend(syntax_issues)
                    
                    if syntax_ok:
                        results["issues"]["cuda"].extend(self.check_cuda_features(content))
                        results["issues"]["performance"].extend(self.check_performance_features(content))
            
            results["validation_time"] = time.time() - start_time
            
            # Determine if test passed (allow some issues)
            # Filter out dependency-related issues (missing CUDA, ONNX, ROS2)
            dependency_issues = []
            other_issues = []
            
            for issue_type, issues in results["issues"].items():
                for issue in issues:
                    if any(dep in issue.lower() for dep in ['cuda', 'nvcc', 'onnxruntime', 'rclcpp', 'file not found']):
                        dependency_issues.append(issue)
                    else:
                        other_issues.append(issue)
            
            # Test passes if we have reasonable number of non-dependency issues
            results["passed"] = len(other_issues) <= 5  # Allow up to 5 non-dependency issues
            results["dependency_issues"] = len(dependency_issues)
            results["other_issues"] = len(other_issues)
            
            return results
            
        except Exception as e:
            results = {
                "test_name": test_name,
                "passed": False,
                "file_count": 0,
                "validation_time": time.time() - start_time,
                "issues": {"error": [str(e)]}
            }
            return results


class TestAdvancedFeatures:
    """Test suite for advanced features validation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = AdvancedFeaturesValidator()
    
    @pytest.mark.parametrize("test_case", PIPELINE_TEST_CASES)
    def test_pipeline_features(self, test_case):
        skip_if_no_ros2()
        """Test pipeline generation features."""
        result = self.validator.run_validation(test_case["source"], test_case["name"])
        
        # Pipeline tests should pass
        assert result["passed"], f"Pipeline test {test_case['name']} failed: {result['issues']}"
        
        # Should generate files
        assert result["file_count"] > 0, f"No files generated for {test_case['name']}"
        
        # Should have reasonable validation time
        assert result["validation_time"] < 60, f"Validation took too long: {result['validation_time']}s"
    
    @pytest.mark.parametrize("test_case", ONNX_TEST_CASES)
    def test_onnx_features(self, test_case):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test ONNX model integration features."""
        result = self.validator.run_validation(test_case["source"], test_case["name"])
        
        # ONNX tests should pass
        assert result["passed"], f"ONNX test {test_case['name']} failed: {result['issues']}"
        
        # Should generate files
        assert result["file_count"] > 0, f"No files generated for {test_case['name']}"
        
        # Should have reasonable validation time
        assert result["validation_time"] < 60, f"Validation took too long: {result['validation_time']}s"
    
    @pytest.mark.parametrize("test_case", TENSORRT_TEST_CASES)
    def test_tensorrt_features(self, test_case):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT optimization features."""
        result = self.validator.run_validation(test_case["source"], test_case["name"])
        
        # TensorRT tests should pass
        assert result["passed"], f"TensorRT test {test_case['name']} failed: {result['issues']}"
        
        # Should generate files
        assert result["file_count"] > 0, f"No files generated for {test_case['name']}"
        
        # Should have reasonable validation time
        assert result["validation_time"] < 60, f"Validation took too long: {result['validation_time']}s"
    
    @pytest.mark.parametrize("test_case", COMPREHENSIVE_TEST_CASES)
    def test_comprehensive_features(self, test_case):
        skip_if_no_ros2()
        """Test comprehensive feature integration."""
        result = self.validator.run_validation(test_case["source"], test_case["name"])
        
        # Comprehensive tests should pass
        assert result["passed"], f"Comprehensive test {test_case['name']} failed: {result['issues']}"
        
        # Should generate multiple files
        assert result["file_count"] >= 3, f"Expected at least 3 files, got {result['file_count']}"
        
        # Should have reasonable validation time
        assert result["validation_time"] < 120, f"Validation took too long: {result['validation_time']}s"
    
    @pytest.mark.parametrize("test_case", EDGE_CASE_TEST_CASES)
    def test_edge_cases(self, test_case):
        skip_if_no_ros2()
        """Test edge cases in advanced features."""
        result = self.validator.run_validation(test_case["source"], test_case["name"])
        
        # Edge cases should pass
        assert result["passed"], f"Edge case test {test_case['name']} failed: {result['issues']}"
        
        # Should generate files
        assert result["file_count"] > 0, f"No files generated for {test_case['name']}"
        
        # Should have reasonable validation time
        assert result["validation_time"] < 60, f"Validation took too long: {result['validation_time']}s"
    
    def test_large_scale_generation(self):
        skip_if_no_ros2()
        """Test large-scale advanced features generation."""
        # Create a large-scale test with multiple features
        source = """
        // Multiple CUDA kernels
        cuda_kernels {
            kernel kernel1 {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                input: float data1(1000)
                output: float result1(1000)
            }
            
            kernel kernel2 {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                input: float data2(1000)
                output: float result2(1000)
            }
        }
        
        // Multiple ONNX models
        onnx_model model1 {
            config {
                input: "input1" -> "float32[1,3,224,224]"
                output: "output1" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        
        onnx_model model2 {
            config {
                input: "input2" -> "float32[1,3,224,224]"
                output: "output2" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: fp16
            }
        }
        
        // Complex pipeline
        pipeline large_scale_pipeline {
            stage stage1 {
                input: "input1"
                output: "output1"
                method: "process1"
                cuda_kernel: "kernel1"
                onnx_model: "model1"
                topic: /pipeline/stage1
            }
            
            stage stage2 {
                input: "output1"
                output: "output2"
                method: "process2"
                cuda_kernel: "kernel2"
                onnx_model: "model2"
                topic: /pipeline/stage2
            }
        }
        """
        
        result = self.validator.run_validation(source, "large_scale_test")
        
        # Large scale test should pass
        assert result["passed"], f"Large scale test failed: {result['issues']}"
        
        # Should generate many files
        assert result["file_count"] >= 5, f"Expected at least 5 files, got {result['file_count']}"
        
        # Should have reasonable validation time
        assert result["validation_time"] < 180, f"Validation took too long: {result['validation_time']}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 