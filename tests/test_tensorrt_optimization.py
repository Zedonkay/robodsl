"""Tests for TensorRT optimization functionality."""

import pytest
from pathlib import Path
import tempfile
import shutil

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda, skip_if_no_tensorrt, skip_if_no_onnx
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.core.ast import OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, DeviceNode, OptimizationNode


class TestTensorRTOptimization:
    """Test TensorRT optimization parsing and generation."""
    
    def test_tensorrt_model_parsing(self, parser):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test parsing of ONNX model with TensorRT optimization."""
        dsl_code = '''
        onnx_model resnet50 {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert model.name == "resnet50"
        assert model.config.device.device == "cuda"
        assert len(model.config.optimizations) == 1
        assert model.config.optimizations[0].optimization == "tensorrt"
    
    def test_tensorrt_generation(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT integration code generation."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Create a test model with TensorRT optimization
        model = OnnxModelNode(
            name="resnet50",
            config=ModelConfigNode(
                inputs=[
                    InputDefNode(name="input", type="float32[1,3,224,224]")
                ],
                outputs=[
                    OutputDefNode(name="output", type="float32[1,1000]")
                ],
                device=DeviceNode(device="cuda"),
                optimizations=[
                    OptimizationNode(optimization="tensorrt")
                ]
            )
        )
        
        # Generate integration files
        generated_files = generator.generate_onnx_integration(model, "test_node")
        
        # Check that files were generated
        assert len(generated_files) == 2  # header and implementation
        
        # Check header file for TensorRT includes
        header_path = str(test_output_dir / "test_node_onnx.hpp")
        assert header_path in generated_files
        
        header_content = generated_files[header_path]
        assert "#include <onnxruntime_providers.h>" in header_content
        assert "run_inference_tensorrt" in header_content
        assert "enable_tensorrt_fp16" in header_content
        assert "enable_tensorrt_int8" in header_content
        assert "tensorrt_enabled_" in header_content
        
        # Check implementation file for TensorRT configuration
        impl_path = str(test_output_dir / "test_node_onnx.cpp")
        assert impl_path in generated_files
        
        impl_content = generated_files[impl_path]
        assert "OrtTensorRTProviderOptions" in impl_content
        assert "AppendExecutionProvider_TensorRT" in impl_content
        assert "trt_fp16_enable = true" in impl_content
        assert "trt_engine_cache_enable = true" in impl_content
        assert "trt_builder_optimization_level = 3" in impl_content
    
    def test_cmake_tensorrt_integration(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test CMake integration with TensorRT."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="resnet50",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        cmake_content = generator.generate_cmake_integration(model, "test_node")
        
        # Check for TensorRT-specific CMake configuration
        assert "find_package(TensorRT REQUIRED)" in cmake_content
        assert "find_package(CUDA REQUIRED)" in cmake_content
        assert "${TensorRT_INCLUDE_DIRS}" in cmake_content
        assert "${TensorRT_LIBRARIES}" in cmake_content
        assert "TENSORRT_CACHE_PATH" in cmake_content
        assert "CUDA_ARCHITECTURES" in cmake_content
        assert "CUDA_SEPARABLE_COMPILATION" in cmake_content
    
    def test_multiple_optimizations(self, parser):
        skip_if_no_ros2()
        """Test ONNX model with multiple optimizations."""
        dsl_code = '''
        onnx_model yolo_model {
            input: "input" -> "float32[1,3,640,640]"
            output: "output" -> "float32[1,25200,85]"
            device: cuda
            optimization: tensorrt
            optimization: openvino
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert len(model.config.optimizations) == 2
        optimization_names = [opt.optimization for opt in model.config.optimizations]
        assert "tensorrt" in optimization_names
        assert "openvino" in optimization_names
    
    def test_tensorrt_without_cuda(self, parser):
        skip_if_no_cuda()
        skip_if_no_tensorrt()
        """Test TensorRT optimization without CUDA device (should still work)."""
        dsl_code = '''
        onnx_model cpu_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cpu
            optimization: tensorrt
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        # TensorRT can still be specified, but it won't be used effectively on CPU
        assert model.config.device.device == "cpu"
        assert len(model.config.optimizations) == 1
        assert model.config.optimizations[0].optimization == "tensorrt"
    
    def test_tensorrt_methods_generation(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test that TensorRT-specific methods are generated."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "test_node")
        impl_content = generated_files[str(test_output_dir / "test_node_onnx.cpp")]
        
        # Check for TensorRT-specific method implementations
        assert "run_inference_tensorrt(" in impl_content
        assert "run_inference_tensorrt_cuda(" in impl_content
        assert "enable_tensorrt_fp16()" in impl_content
        assert "enable_tensorrt_int8()" in impl_content
        assert "set_tensorrt_workspace_size(" in impl_content
        assert "clear_tensorrt_cache()" in impl_content
        assert "initialize_tensorrt_options(" in impl_content
        assert "create_tensorrt_cache_directory()" in impl_content
        assert "validate_tensorrt_engine()" in impl_content
    
    def test_tensorrt_initialization(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT initialization in constructor."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "test_node")
        impl_content = generated_files[str(test_output_dir / "test_node_onnx.cpp")]
        
        # Check for TensorRT initialization in constructor
        assert "tensorrt_enabled_ = true" in impl_content
        assert "tensorrt_fp16_enabled_ = true" in impl_content
        assert "tensorrt_int8_enabled_ = false" in impl_content
        assert "tensorrt_workspace_size_ = 1 << 30" in impl_content
        assert "tensorrt_cache_path_ = \"./trt_cache\"" in impl_content 