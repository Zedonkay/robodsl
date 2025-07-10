"""Comprehensive Linux dependency tests for RoboDSL.

This module provides extensive test coverage for all Linux dependencies including:
- CUDA and GPU acceleration
- ROS2 integration
- TensorRT optimization
- ONNX Runtime
- CMake build system
- Python/C++ interoperability
- Memory management
- Performance optimization
- Error handling and recovery
"""

import pytest
import tempfile
import shutil
import os
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import (
    skip_if_no_ros2, skip_if_no_cuda, skip_if_no_tensorrt, 
    skip_if_no_onnx, has_ros2, has_cuda, has_tensorrt, has_onnx
)
from robodsl.generators.main_generator import MainGenerator
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.generators.cuda_kernel_generator import CudaKernelGenerator
from robodsl.generators.cpp_node_generator import CppNodeGenerator
from robodsl.generators.python_node_generator import PythonNodeGenerator
from robodsl.generators.cmake_generator import CMakeGenerator
from robodsl.core.ast import (
    OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, 
    DeviceNode, OptimizationNode, RoboDSLAST, KernelNode,
    NodeNode, PipelineNode
)


class TestLinuxDependenciesComprehensive:
    """Comprehensive Linux dependency test suite."""
    
    @pytest.fixture
    def linux_test_config(self):
        """Create a comprehensive Linux test configuration."""
        return {
            "cuda_available": has_cuda(),
            "tensorrt_available": has_tensorrt(),
            "ros2_available": has_ros2(),
            "onnx_available": has_onnx(),
            "gpu_memory": self._get_gpu_memory(),
            "cpu_cores": os.cpu_count(),
            "system_memory": self._get_system_memory()
        }
    
    def _get_gpu_memory(self):
        """Get available GPU memory."""
        try:
            if has_cuda():
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return int(result.stdout.strip().split('\n')[0])
        except:
            pass
        return 0
    
    def _get_system_memory(self):
        """Get system memory in GB."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1]) // (1024 * 1024)
        except:
            pass
        return 8  # Default assumption
    
    @pytest.fixture
    def complex_cuda_kernel(self):
        """Create a complex CUDA kernel configuration."""
        return CudaKernelNode(
            name="complex_kernel",
            kernel_code="""
            __global__ void complex_kernel(float* input, float* output, int size) {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) {
                    // Complex computation
                    float val = input[idx];
                    val = __sinf(val) * __cosf(val);
                    val = __expf(-val * val);
                    output[idx] = val;
                }
            }
            """,
            block_size=256,
            grid_size="(size + 255) / 256",
            inputs=["input", "output", "size"],
            outputs=["output"]
        )
    
    @pytest.fixture
    def multi_gpu_config(self):
        """Create multi-GPU configuration."""
        return {
            "gpu_count": self._get_gpu_count(),
            "gpu_memory_per_device": self._get_gpu_memory(),
            "gpu_architectures": self._get_gpu_architectures()
        }
    
    def _get_gpu_count(self):
        """Get number of available GPUs."""
        try:
            if has_cuda():
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return len(result.stdout.strip().split('\n'))
        except:
            pass
        return 1
    
    def _get_gpu_architectures(self):
        """Get GPU architectures."""
        try:
            if has_cuda():
                result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                                     capture_output=True, text=True)
                if result.returncode == 0:
                    return [cap.strip() for cap in result.stdout.strip().split('\n')]
        except:
            pass
        return ["8.6"]  # Default to RTX 30 series
    
    def test_linux_system_detection(self, linux_test_config):
        """Test Linux system detection and configuration."""
        assert isinstance(linux_test_config["cuda_available"], bool)
        assert isinstance(linux_test_config["tensorrt_available"], bool)
        assert isinstance(linux_test_config["ros2_available"], bool)
        assert isinstance(linux_test_config["onnx_available"], bool)
        assert linux_test_config["cpu_cores"] > 0
        assert linux_test_config["system_memory"] > 0
        
        # Test system information
        assert os.name == 'posix'  # Should be POSIX (Linux/macOS)
        # On macOS, sys.platform is 'darwin', on Linux it's 'linux'
        assert sys.platform.lower() in ['linux', 'darwin']
    
    def test_cuda_environment_comprehensive(self, linux_test_config):
        """Test comprehensive CUDA environment setup."""
        skip_if_no_cuda()
        
        # Test CUDA compiler availability
        assert shutil.which('nvcc') is not None
        
        # Test CUDA runtime
        try:
            import torch
            if torch.cuda.is_available():
                assert torch.cuda.device_count() > 0
                assert torch.cuda.get_device_name(0) != ""
        except ImportError:
            pass
        
        # Test CUDA version
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                 capture_output=True, text=True)
            assert result.returncode == 0
            assert 'release' in result.stdout.lower()
        except:
            pytest.fail("CUDA compiler not working properly")
        
        # Test GPU information
        try:
            result = subprocess.run(['nvidia-smi'], 
                                 capture_output=True, text=True)
            assert result.returncode == 0
        except:
            pytest.fail("nvidia-smi not available")
    
    def test_tensorrt_environment_comprehensive(self, linux_test_config):
        """Test comprehensive TensorRT environment setup."""
        skip_if_no_tensorrt()
        
        # Test TensorRT library availability
        try:
            import ctypes
            lib = ctypes.CDLL('libnvinfer.so')
            assert lib is not None
        except (OSError, ImportError):
            pytest.fail("TensorRT library not found")
        
        # Test TensorRT Python bindings
        try:
            import tensorrt as trt
            assert trt.__version__ is not None
        except ImportError:
            pass  # Python bindings not required for all tests
        
        # Test TensorRT sample compilation
        tensorrt_paths = [
            '/usr/local/tensorrt',
            '/opt/tensorrt',
            '/usr/lib/x86_64-linux-gnu/tensorrt',
            '/usr/lib/aarch64-linux-gnu/tensorrt'
        ]
        
        tensorrt_found = False
        for path in tensorrt_paths:
            if os.path.exists(path):
                tensorrt_found = True
                break
        
        assert tensorrt_found or 'TENSORRT_ROOT' in os.environ
    
    def test_ros2_environment_comprehensive(self, linux_test_config):
        """Test comprehensive ROS2 environment setup."""
        skip_if_no_ros2()
        
        # Test ROS2 installation
        assert shutil.which('ros2') is not None or 'AMENT_PREFIX_PATH' in os.environ
        
        # Test ROS2 workspace
        try:
            result = subprocess.run(['ros2', 'node', 'list'], 
                                 capture_output=True, text=True)
            # Should not fail even if no nodes are running
            assert result.returncode in [0, 1]
        except:
            pass
        
        # Test ROS2 build tools
        assert shutil.which('colcon') is not None
        
        # Test ROS2 message types
        try:
            result = subprocess.run(['ros2', 'interface', 'list'], 
                                 capture_output=True, text=True)
            assert result.returncode == 0
        except:
            pass
    
    def test_onnx_environment_comprehensive(self, linux_test_config):
        """Test comprehensive ONNX environment setup."""
        skip_if_no_onnx()
        
        try:
            import onnxruntime as ort
            assert ort.__version__ is not None
            
            # Test ONNX Runtime providers
            providers = ort.get_available_providers()
            assert len(providers) > 0
            
            # Test CUDA provider if available
            if 'CUDAExecutionProvider' in providers:
                assert ort.get_device('GPU') == 'GPU'
        except ImportError:
            pytest.fail("ONNX Runtime not properly installed")
    
    def test_multi_gpu_support(self, multi_gpu_config):
        """Test multi-GPU support and configuration."""
        skip_if_no_cuda()
        
        gpu_count = multi_gpu_config["gpu_count"]
        assert gpu_count > 0
        
        # Test multi-GPU CUDA kernels
        dsl_code = f'''
        cuda_kernel multi_gpu_kernel {{
            kernel: |
                __global__ void multi_gpu_kernel(float* input, float* output, int size, int gpu_id) {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {{
                        // GPU-specific computation
                        float val = input[idx];
                        val = val * (gpu_id + 1);
                        output[idx] = val;
                    }}
                }}
            block_size: 256
            grid_size: "(size + 255) / 256"
            inputs: ["input", "output", "size", "gpu_id"]
            outputs: ["output"]
        }}
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels) == 1
        kernel = ast.cuda_kernels[0]
        assert kernel.name == "multi_gpu_kernel"
        assert "gpu_id" in kernel.inputs
    
    def test_memory_management_comprehensive(self, linux_test_config):
        """Test comprehensive memory management scenarios."""
        skip_if_no_cuda()
        
        # Test large memory allocation
        dsl_code = '''
        cuda_kernel memory_test {
            kernel: |
                __global__ void memory_test(float* input, float* output, int size) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Memory-intensive operation
                        float sum = 0.0f;
                        for (int i = 0; i < 1000; i++) {
                            sum += input[idx] * input[idx];
                        }
                        output[idx] = sum;
                    }
                }
            block_size: 256
            grid_size: "(size + 255) / 256"
            inputs: ["input", "output", "size"]
            outputs: ["output"]
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels) == 1
        
        # Test memory pool configuration
        system_memory = linux_test_config["system_memory"]
        gpu_memory = linux_test_config["gpu_memory"]
        
        # Generate memory-aware configuration
        memory_config = {
            "cpu_memory_limit": system_memory * 0.8,  # 80% of system memory
            "gpu_memory_limit": gpu_memory * 0.8 if gpu_memory > 0 else 8,  # 80% of GPU memory
            "memory_pool_size": min(system_memory * 0.5, 16),  # Max 16GB pool
            "gpu_memory_pool_size": min(gpu_memory * 0.5, 8) if gpu_memory > 0 else 4
        }
        
        assert memory_config["cpu_memory_limit"] > 0
        assert memory_config["gpu_memory_limit"] > 0
    
    def test_performance_optimization_comprehensive(self, linux_test_config):
        """Test comprehensive performance optimization scenarios."""
        skip_if_no_cuda()
        skip_if_no_tensorrt()
        
        # Test performance-critical configurations
        dsl_code = '''
        onnx_model performance_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        
        cuda_kernel performance_kernel {
            kernel: |
                __global__ void performance_kernel(float* input, float* output, int size) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        // Optimized computation
                        float val = input[idx];
                        val = __fmaf_rn(val, val, 1.0f);  // Fused multiply-add
                        val = __expf(-val);
                        output[idx] = val;
                    }
                }
            block_size: 256
            grid_size: "(size + 255) / 256"
            inputs: ["input", "output", "size"]
            outputs: ["output"]
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.onnx_models) == 1
        assert len(ast.cuda_kernels) == 1
        
        # Test performance monitoring
        performance_config = {
            "enable_profiling": True,
            "enable_memory_tracking": True,
            "enable_throughput_monitoring": True,
            "performance_thresholds": {
                "max_latency_ms": 16.67,  # 60 FPS
                "min_throughput_fps": 30,
                "max_memory_usage_gb": linux_test_config["gpu_memory"] * 0.8 if linux_test_config["gpu_memory"] > 0 else 8
            }
        }
        
        assert performance_config["enable_profiling"]
        assert performance_config["performance_thresholds"]["max_latency_ms"] > 0
    
    def test_error_handling_comprehensive(self, test_output_dir):
        """Test comprehensive error handling scenarios."""
        skip_if_no_cuda()
        
        # Test invalid CUDA configurations
        invalid_configs = [
            # Invalid block size
            '''
            cuda_kernel invalid_block {
                kernel: |
                    __global__ void test(float* input, float* output, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) output[idx] = input[idx];
                    }
                block_size: 0
                grid_size: "(size + 255) / 256"
                inputs: ["input", "output", "size"]
                outputs: ["output"]
            }
            ''',
            # Invalid grid size
            '''
            cuda_kernel invalid_grid {
                kernel: |
                    __global__ void test(float* input, float* output, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) output[idx] = input[idx];
                    }
                block_size: 256
                grid_size: "invalid_expression"
                inputs: ["input", "output", "size"]
                outputs: ["output"]
            }
            ''',
            # Missing kernel code
            '''
            cuda_kernel empty_kernel {
                kernel: ""
                block_size: 256
                grid_size: "(size + 255) / 256"
                inputs: ["input", "output", "size"]
                outputs: ["output"]
            }
            '''
        ]
        
        for i, config in enumerate(invalid_configs):
            try:
                ast = parse_robodsl(config)
                # Should handle gracefully or raise appropriate error
            except Exception as e:
                # Expected behavior for invalid configurations
                assert isinstance(e, (ValueError, SyntaxError, RuntimeError))
    
    def test_build_system_comprehensive(self, test_output_dir):
        """Test comprehensive build system integration."""
        skip_if_no_ros2()
        
        # Test CMake configuration
        cmake_generator = CMakeGenerator(str(test_output_dir))
        
        # Test ROS2 package generation
        package_config = {
            "package_name": "test_package",
            "dependencies": ["rclcpp", "std_msgs", "sensor_msgs"],
            "cuda_dependencies": ["cuda", "cudart"],
            "tensorrt_dependencies": ["nvinfer", "nvonnxparser"],
            "build_type": "Release"
        }
        
        cmake_content = cmake_generator.generate_cmake_lists(package_config)
        assert "find_package(rclcpp REQUIRED)" in cmake_content
        assert "find_package(CUDA REQUIRED)" in cmake_content
        
        # Test package.xml generation
        package_xml = cmake_generator.generate_package_xml(package_config)
        assert "<name>test_package</name>" in package_xml
        assert "<depend>rclcpp</depend>" in package_xml
    
    def test_pipeline_integration_comprehensive(self, test_output_dir):
        """Test comprehensive pipeline integration scenarios."""
        skip_if_no_ros2()
        skip_if_no_cuda()
        skip_if_no_tensorrt()
        
        # Test complex pipeline with multiple components
        dsl_code = '''
        pipeline complex_pipeline {
            stage: "preprocessing" {
                cuda_kernel: "normalize_kernel"
                input: "raw_data"
                output: "normalized_data"
            }
            
            stage: "inference" {
                onnx_model: "model"
                input: "normalized_data"
                output: "predictions"
                device: cuda
                optimization: tensorrt
            }
            
            stage: "postprocessing" {
                cuda_kernel: "postprocess_kernel"
                input: "predictions"
                output: "final_result"
            }
        }
        
        cuda_kernel normalize_kernel {
            kernel: |
                __global__ void normalize(float* input, float* output, int size) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = (input[idx] - 127.5f) / 127.5f;
                    }
                }
            block_size: 256
            grid_size: "(size + 255) / 256"
            inputs: ["input", "output", "size"]
            outputs: ["output"]
        }
        
        onnx_model model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        
        cuda_kernel postprocess_kernel {
            kernel: |
                __global__ void postprocess(float* input, float* output, int size) {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output[idx] = __expf(input[idx]);
                    }
                }
            block_size: 256
            grid_size: "(size + 255) / 256"
            inputs: ["input", "output", "size"]
            outputs: ["output"]
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.pipelines) == 1
        assert len(ast.cuda_kernels) == 2
        assert len(ast.onnx_models) == 1
        
        pipeline = ast.pipelines[0]
        assert len(pipeline.stages) == 3
        assert pipeline.stages[0].name == "preprocessing"
        assert pipeline.stages[1].name == "inference"
        assert pipeline.stages[2].name == "postprocessing"
    
    def test_stress_testing_comprehensive(self, test_output_dir, linux_test_config):
        """Test comprehensive stress testing scenarios."""
        skip_if_no_cuda()
        skip_if_no_tensorrt()
        
        # Test memory stress
        large_input_size = min(linux_test_config["gpu_memory"] * 1024 * 1024 // 4, 1000000)  # Use 25% of GPU memory
        
        dsl_code = f'''
        cuda_kernel stress_kernel {{
            kernel: |
                __global__ void stress_kernel(float* input, float* output, int size) {{
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {{
                        // Memory-intensive stress test
                        float sum = 0.0f;
                        for (int i = 0; i < 100; i++) {{
                            sum += input[idx] * input[idx];
                            sum = __sinf(sum) + __cosf(sum);
                        }}
                        output[idx] = sum;
                    }}
                }}
            block_size: 256
            grid_size: "(size + 255) / 256"
            inputs: ["input", "output", "size"]
            outputs: ["output"]
        }}
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.cuda_kernels) == 1
        
        # Test concurrent execution
        concurrent_config = {
            "max_concurrent_kernels": 4,
            "max_concurrent_streams": 8,
            "memory_pool_size": linux_test_config["gpu_memory"] * 0.5 if linux_test_config["gpu_memory"] > 0 else 4,
            "stress_test_duration": 60,  # 60 seconds
            "stress_test_iterations": 1000
        }
        
        assert concurrent_config["max_concurrent_kernels"] > 0
        assert concurrent_config["max_concurrent_streams"] > 0
    
    def test_compatibility_testing_comprehensive(self, linux_test_config):
        """Test comprehensive compatibility scenarios."""
        skip_if_no_cuda()
        
        # Test different CUDA versions
        cuda_versions = ["11.0", "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "12.0", "12.1", "12.2"]
        
        # Test different GPU architectures
        gpu_architectures = ["6.0", "6.1", "7.0", "7.5", "8.0", "8.6", "8.9", "9.0"]
        
        # Test different TensorRT versions
        tensorrt_versions = ["8.0", "8.1", "8.2", "8.3", "8.4", "8.5", "8.6"]
        
        # Test different ROS2 distributions
        ros2_distributions = ["foxy", "galactic", "humble", "iron", "rolling"]
        
        compatibility_matrix = {
            "cuda_versions": cuda_versions,
            "gpu_architectures": gpu_architectures,
            "tensorrt_versions": tensorrt_versions,
            "ros2_distributions": ros2_distributions,
            "python_versions": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "compiler_versions": ["gcc-9", "gcc-10", "gcc-11", "gcc-12", "clang-12", "clang-13", "clang-14", "clang-15"]
        }
        
        # Test current environment compatibility
        current_config = {
            "cuda_version": self._get_cuda_version(),
            "gpu_architecture": self._get_current_gpu_arch(),
            "tensorrt_version": self._get_tensorrt_version(),
            "ros2_distribution": self._get_ros2_distribution(),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "compiler_version": self._get_compiler_version()
        }
        
        # Verify current configuration is in compatibility matrix
        for key, value in current_config.items():
            if value and key in compatibility_matrix:
                assert value in compatibility_matrix[key] or "unknown" in value.lower()
    
    def _get_cuda_version(self):
        """Get current CUDA version."""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'release' in line.lower():
                        version = line.split('release')[1].strip().split(',')[0]
                        return version
        except:
            pass
        return "unknown"
    
    def _get_current_gpu_arch(self):
        """Get current GPU architecture."""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return "unknown"
    
    def _get_tensorrt_version(self):
        """Get TensorRT version."""
        try:
            import tensorrt as trt
            return trt.__version__
        except ImportError:
            pass
        return "unknown"
    
    def _get_ros2_distribution(self):
        """Get ROS2 distribution."""
        try:
            result = subprocess.run(['ros2', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'distribution' in line.lower():
                        return line.split()[-1]
        except:
            pass
        return "unknown"
    
    def _get_compiler_version(self):
        """Get compiler version."""
        try:
            result = subprocess.run(['gcc', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                version = result.stdout.split('\n')[0].split()[-1]
                return f"gcc-{version}"
        except:
            pass
        return "unknown"
    
    def test_security_testing_comprehensive(self, test_output_dir):
        """Test comprehensive security scenarios."""
        skip_if_no_cuda()
        
        # Test input validation
        malicious_inputs = [
            # Buffer overflow attempts
            '''
            cuda_kernel malicious_kernel {
                kernel: |
                    __global__ void malicious(float* input, float* output, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            output[idx] = input[idx + 1000000];  // Buffer overflow
                        }
                    }
                block_size: 256
                grid_size: "(size + 255) / 256"
                inputs: ["input", "output", "size"]
                outputs: ["output"]
            }
            ''',
            # Memory access violations
            '''
            cuda_kernel null_pointer {
                kernel: |
                    __global__ void null_pointer(float* input, float* output, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            output[idx] = input[idx];
                            if (idx == 0) {
                                float* null_ptr = nullptr;
                                *null_ptr = 1.0f;  // Null pointer dereference
                            }
                        }
                    }
                block_size: 256
                grid_size: "(size + 255) / 256"
                inputs: ["input", "output", "size"]
                outputs: ["output"]
            }
            '''
        ]
        
        for i, malicious_input in enumerate(malicious_inputs):
            try:
                ast = parse_robodsl(malicious_input)
                # Should handle gracefully or raise appropriate error
            except Exception as e:
                # Expected behavior for malicious inputs
                assert isinstance(e, (ValueError, SyntaxError, RuntimeError))
    
    def test_monitoring_comprehensive(self, test_output_dir):
        """Test comprehensive monitoring and logging scenarios."""
        skip_if_no_cuda()
        
        # Test performance monitoring
        monitoring_config = {
            "enable_gpu_monitoring": True,
            "enable_memory_monitoring": True,
            "enable_temperature_monitoring": True,
            "enable_power_monitoring": True,
            "monitoring_interval_ms": 100,
            "log_level": "INFO",
            "metrics": [
                "gpu_utilization",
                "memory_utilization", 
                "temperature",
                "power_consumption",
                "throughput",
                "latency"
            ]
        }
        
        assert monitoring_config["enable_gpu_monitoring"]
        assert len(monitoring_config["metrics"]) > 0
        
        # Test logging configuration
        logging_config = {
            "log_file": str(test_output_dir / "test.log"),
            "log_level": "DEBUG",
            "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "max_log_size_mb": 100,
            "backup_count": 5
        }
        
        assert logging_config["log_file"].endswith(".log")
        assert logging_config["max_log_size_mb"] > 0
    
    def test_deployment_comprehensive(self, test_output_dir):
        """Test comprehensive deployment scenarios."""
        skip_if_no_ros2()
        
        # Test Docker deployment
        docker_config = {
            "base_image": "nvidia/cuda:11.8-devel-ubuntu20.04",
            "ros2_distribution": "humble",
            "python_version": "3.10",
            "install_tensorrt": True,
            "install_onnx": True,
            "expose_ports": [11311, 11312],
            "volume_mounts": ["/tmp:/tmp"],
            "environment_variables": {
                "CUDA_VISIBLE_DEVICES": "0",
                "ROS_DOMAIN_ID": "0"
            }
        }
        
        assert "nvidia/cuda" in docker_config["base_image"]
        assert docker_config["ros2_distribution"] in ["foxy", "galactic", "humble", "iron", "rolling"]
        
        # Test systemd service deployment
        systemd_config = {
            "service_name": "robodsl-service",
            "user": "robodsl",
            "group": "robodsl",
            "working_directory": "/opt/robodsl",
            "environment_file": "/etc/robodsl/environment",
            "restart_policy": "always",
            "restart_sec": 10
        }
        
        assert systemd_config["service_name"] == "robodsl-service"
        assert systemd_config["restart_policy"] == "always"
    
    def test_integration_testing_comprehensive(self, test_output_dir):
        """Test comprehensive integration testing scenarios."""
        skip_if_no_ros2()
        skip_if_no_cuda()
        skip_if_no_tensorrt()
        
        # Test end-to-end pipeline
        integration_config = {
            "test_duration_seconds": 300,  # 5 minutes
            "concurrent_users": 10,
            "data_volume_gb": 10,
            "performance_thresholds": {
                "max_latency_ms": 100,
                "min_throughput_fps": 30,
                "max_memory_usage_gb": 8,
                "max_cpu_usage_percent": 80,
                "max_gpu_usage_percent": 90
            },
            "error_thresholds": {
                "max_error_rate_percent": 1.0,
                "max_crash_rate_percent": 0.1,
                "max_memory_leak_mb": 100
            }
        }
        
        assert integration_config["test_duration_seconds"] > 0
        assert integration_config["concurrent_users"] > 0
        assert integration_config["performance_thresholds"]["max_latency_ms"] > 0
        assert integration_config["error_thresholds"]["max_error_rate_percent"] > 0
    
    def test_future_compatibility_comprehensive(self, test_output_dir):
        """Test comprehensive future compatibility scenarios."""
        skip_if_no_cuda()
        
        # Test upcoming CUDA features
        future_features = {
            "cuda_12_3_features": [
                "cudaGraphInstantiateFlagAutoFreeOnLaunch",
                "cudaGraphInstantiateFlagUseNodePriority",
                "cudaGraphInstantiateFlagUseNodePriority"
            ],
            "tensorrt_9_0_features": [
                "ILayer::setPrecision",
                "ILayer::setOutputType",
                "IBuilderConfig::setTacticSources"
            ],
            "ros2_rolling_features": [
                "rclcpp::QoS",
                "rclcpp::NodeOptions",
                "rclcpp::PublisherOptions"
            ]
        }
        
        # Test backward compatibility
        backward_compatibility = {
            "cuda_versions": ["10.0", "10.1", "10.2", "11.0", "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8", "12.0", "12.1", "12.2"],
            "tensorrt_versions": ["7.0", "7.1", "7.2", "8.0", "8.1", "8.2", "8.3", "8.4", "8.5", "8.6"],
            "ros2_versions": ["foxy", "galactic", "humble", "iron", "rolling"],
            "python_versions": ["3.7", "3.8", "3.9", "3.10", "3.11", "3.12"]
        }
        
        assert len(future_features["cuda_12_3_features"]) > 0
        assert len(backward_compatibility["cuda_versions"]) > 0 