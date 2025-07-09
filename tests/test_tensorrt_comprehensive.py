"""Comprehensive TensorRT optimization tests for RoboDSL.

This module provides extensive test coverage for all TensorRT optimization features,
including edge cases, error conditions, performance scenarios, and integration testing.
"""

import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda, skip_if_no_tensorrt, skip_if_no_onnx
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.core.ast import (
    OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, 
    DeviceNode, OptimizationNode, RoboDSLAST
)


class TestTensorRTComprehensive:
    """Comprehensive TensorRT optimization test suite."""
    
    @pytest.fixture
    def complex_model_config(self):
        """Create a complex ONNX model configuration for testing."""
        return OnnxModelNode(
            name="complex_model",
            config=ModelConfigNode(
                inputs=[
                    InputDefNode(name="input1", type="float32[1,3,224,224]"),
                    InputDefNode(name="input2", type="float32[1,1,224,224]"),
                    InputDefNode(name="input3", type="float32[1,512]")
                ],
                outputs=[
                    OutputDefNode(name="output1", type="float32[1,1000]"),
                    OutputDefNode(name="output2", type="float32[1,10]"),
                    OutputDefNode(name="output3", type="float32[1,512]")
                ],
                device=DeviceNode(device="cuda"),
                optimizations=[
                    OptimizationNode(optimization="tensorrt"),
                    OptimizationNode(optimization="openvino")
                ]
            )
        )
    
    @pytest.fixture
    def edge_case_models(self):
        """Create edge case model configurations."""
        return {
            "minimal": OnnxModelNode(
                name="minimal_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float32[1,1,1,1]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,1]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            ),
            "large": OnnxModelNode(
                name="large_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float32[1,3,1024,1024]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,10000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            ),
            "mixed_precision": OnnxModelNode(
                name="mixed_precision_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float16[1,3,224,224]")],
                    outputs=[OutputDefNode(name="output", type="float16[1,1000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
        }
    
    def test_tensorrt_parsing_edge_cases(self, parser):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test parsing of edge case TensorRT configurations."""
        
        # Test 1: Minimal TensorRT configuration
        dsl_code = '''
        onnx_model minimal {
            input: "input" -> "float32[1,1,1,1]"
            output: "output" -> "float32[1,1]"
            device: cuda
            optimization: tensorrt
        }
        '''
        ast = parse_robodsl(dsl_code)
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        assert model.name == "minimal"
        assert len(model.config.inputs) == 1
        assert len(model.config.outputs) == 1
        assert model.config.inputs[0].type == "float32[1,1,1,1]"
        
        # Test 2: Multiple optimizations
        dsl_code = '''
        onnx_model multi_opt {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
            optimization: openvino
            optimization: onednn
        }
        '''
        ast = parse_robodsl(dsl_code)
        model = ast.onnx_models[0]
        assert len(model.config.optimizations) == 3
        opt_names = [opt.optimization for opt in model.config.optimizations]
        assert "tensorrt" in opt_names
        assert "openvino" in opt_names
        assert "onednn" in opt_names
        
        # Test 3: TensorRT with CPU device (should still work)
        dsl_code = '''
        onnx_model cpu_tensorrt {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cpu
            optimization: tensorrt
        }
        '''
        ast = parse_robodsl(dsl_code)
        model = ast.onnx_models[0]
        assert model.config.device.device == "cpu"
        assert len(model.config.optimizations) == 1
        assert model.config.optimizations[0].optimization == "tensorrt"
        
        # Test 4: TensorRT with quoted and unquoted names
        dsl_code = '''
        onnx_model mixed_names {
            input: "input_tensor" -> "float32[1,3,224,224]"
            input: input_data -> "float32[1,1,224,224]"
            output: "output_tensor" -> "float32[1,1000]"
            output: output_data -> "float32[1,10]"
            device: cuda
            optimization: tensorrt
        }
        '''
        ast = parse_robodsl(dsl_code)
        model = ast.onnx_models[0]
        assert len(model.config.inputs) == 2
        assert len(model.config.outputs) == 2
        assert model.config.inputs[0].name == "input_tensor"
        assert model.config.inputs[1].name == "input_data"
    
    def test_tensorrt_generation_edge_cases(self, test_output_dir, edge_case_models):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test code generation for edge case scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Minimal model generation
        minimal_model = edge_case_models["minimal"]
        generated_files = generator.generate_onnx_integration(minimal_model, "minimal_node")
        
        header_content = generated_files[str(test_output_dir / "minimal_node_onnx.hpp")]
        impl_content = generated_files[str(test_output_dir / "minimal_node_onnx.cpp")]
        
        # Check TensorRT includes and methods are present
        assert "#include <onnxruntime_providers.h>" in header_content
        assert "run_inference_tensorrt" in header_content
        assert "OrtTensorRTProviderOptions" in impl_content
        assert "AppendExecutionProvider_TensorRT" in impl_content
        
        # Test 2: Large model generation
        large_model = edge_case_models["large"]
        generated_files = generator.generate_onnx_integration(large_model, "large_node")
        
        impl_content = generated_files[str(test_output_dir / "large_node_onnx.cpp")]
        assert "trt_max_workspace_size = 1 << 30" in impl_content  # 1GB workspace
        
        # Test 3: Mixed precision model
        mixed_model = edge_case_models["mixed_precision"]
        generated_files = generator.generate_onnx_integration(mixed_model, "mixed_node")
        
        impl_content = generated_files[str(test_output_dir / "mixed_node_onnx.cpp")]
        assert "trt_fp16_enable = true" in impl_content
    
    def test_tensorrt_cmake_integration_comprehensive(self, test_output_dir, complex_model_config):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test comprehensive CMake integration scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Complex model with TensorRT
        cmake_content = generator.generate_cmake_integration(complex_model_config, "complex_node")
        
        # Check all TensorRT dependencies
        assert "find_package(TensorRT REQUIRED)" in cmake_content
        assert "find_package(CUDA REQUIRED)" in cmake_content
        assert "${TensorRT_INCLUDE_DIRS}" in cmake_content
        assert "${TensorRT_LIBRARIES}" in cmake_content
        assert "${CUDA_INCLUDE_DIRS}" in cmake_content
        assert "${CUDA_LIBRARIES}" in cmake_content
        
        # Check CUDA properties
        assert "CUDA_SEPARABLE_COMPILATION ON" in cmake_content
        assert "CUDA_ARCHITECTURES" in cmake_content
        assert "60;70;75;80;86" in cmake_content
        
        # Check TensorRT cache configuration
        assert "TENSORRT_CACHE_PATH" in cmake_content
        assert "file(MAKE_DIRECTORY" in cmake_content
        
        # Test 2: Model without TensorRT (should not include TensorRT)
        non_tensorrt_model = OnnxModelNode(
            name="cpu_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cpu")
            )
        )
        
        cmake_content = generator.generate_cmake_integration(non_tensorrt_model, "cpu_node")
        assert "find_package(TensorRT REQUIRED)" not in cmake_content
        assert "find_package(CUDA REQUIRED)" not in cmake_content
        assert "CUDA_SEPARABLE_COMPILATION" not in cmake_content
    
    def test_tensorrt_method_generation_comprehensive(self, test_output_dir, complex_model_config):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test comprehensive method generation for TensorRT."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        generated_files = generator.generate_onnx_integration(complex_model_config, "complex_node")
        
        impl_content = generated_files[str(test_output_dir / "complex_node_onnx.cpp")]
        
        # Test 1: All TensorRT methods are generated
        expected_methods = [
            "run_inference_tensorrt(",
            "run_inference_tensorrt_cuda(",
            "enable_tensorrt_fp16()",
            "enable_tensorrt_int8()",
            "set_tensorrt_workspace_size(",
            "clear_tensorrt_cache()",
            "initialize_tensorrt_options(",
            "create_tensorrt_cache_directory()",
            "validate_tensorrt_engine()"
        ]
        
        for method in expected_methods:
            assert method in impl_content, f"Method {method} not found in generated code"
        
        # Test 2: TensorRT configuration options
        expected_configs = [
            "trt_max_workspace_size = 1 << 30",
            "trt_fp16_enable = true",
            "trt_int8_enable = false",
            "trt_engine_cache_enable = true",
            "trt_engine_cache_path = \"./trt_cache\"",
            "trt_builder_optimization_level = 3",
            "trt_optimization_level = 3"
        ]
        
        for config in expected_configs:
            assert config in impl_content, f"Config {config} not found in generated code"
        
        # Test 3: Error handling in methods
        assert "if (!tensorrt_enabled_)" in impl_content
        assert "std::cerr << \"TensorRT is not enabled" in impl_content
        assert "return false;" in impl_content
    
    def test_tensorrt_initialization_comprehensive(self, test_output_dir, complex_model_config):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test comprehensive initialization scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        generated_files = generator.generate_onnx_integration(complex_model_config, "complex_node")
        
        impl_content = generated_files[str(test_output_dir / "complex_node_onnx.cpp")]
        
        # Test 1: Constructor initialization
        assert "tensorrt_enabled_ = true" in impl_content
        assert "tensorrt_fp16_enabled_ = true" in impl_content
        assert "tensorrt_int8_enabled_ = false" in impl_content
        assert "tensorrt_workspace_size_ = 1 << 30" in impl_content
        assert "tensorrt_cache_path_ = \"./trt_cache\"" in impl_content
        
        # Test 2: Session options initialization
        assert "OrtTensorRTProviderOptions trt_options" in impl_content
        assert "session_options.AppendExecutionProvider_TensorRT" in impl_content
        assert "session_options.AppendExecutionProvider_CUDA" in impl_content
        
        # Test 3: CUDA options configuration
        assert "OrtCUDAProviderOptions cuda_options" in impl_content
        assert "cuda_options.device_id = 0" in impl_content
        assert "cuda_options.gpu_mem_limit = SIZE_MAX" in impl_content
    
    def test_tensorrt_error_handling(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test comprehensive error handling scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Model with invalid configuration
        invalid_model = OnnxModelNode(
            name="invalid_model",
            config=ModelConfigNode(
                inputs=[],  # No inputs
                outputs=[],  # No outputs
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        # Should still generate valid code
        generated_files = generator.generate_onnx_integration(invalid_model, "invalid_node")
        assert len(generated_files) == 2
        
        # Test 2: Model with conflicting optimizations
        conflicting_model = OnnxModelNode(
            name="conflicting_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cpu"),  # CPU with TensorRT
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(conflicting_model, "conflicting_node")
        impl_content = generated_files[str(test_output_dir / "conflicting_node_onnx.cpp")]
        
        # Should still generate TensorRT code but with appropriate warnings
        assert "OrtTensorRTProviderOptions" in impl_content
    
    def test_tensorrt_performance_scenarios(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test performance-related scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: High-performance model configuration
        perf_model = OnnxModelNode(
            name="perf_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,512,512]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(perf_model, "perf_node")
        impl_content = generated_files[str(test_output_dir / "perf_node_onnx.cpp")]
        
        # Check performance optimizations
        assert "trt_builder_optimization_level = 3" in impl_content
        assert "trt_optimization_level = 3" in impl_content
        assert "trt_fp16_enable = true" in impl_content
        assert "trt_engine_cache_enable = true" in impl_content
        assert "trt_timing_cache_enable = true" in impl_content
        
        # Test 2: Memory optimization settings
        assert "trt_context_memory_sharing_enable = true" in impl_content
        assert "trt_max_workspace_size = 1 << 30" in impl_content
    
    def test_tensorrt_integration_with_nodes(self, parser):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT integration within node contexts."""
        
        # Test 1: Node with TensorRT model
        dsl_code = '''
        node tensorrt_node {
            subscriber /input: "sensor_msgs/msg/Image"
            publisher /output: "std_msgs/msg/Float32MultiArray"
            
            onnx_model resnet50 {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "tensorrt_node"
        assert len(node.content.onnx_models) == 1
        assert node.content.onnx_models[0].name == "resnet50"
        
        # Test 2: Node with multiple TensorRT models
        dsl_code = '''
        node multi_tensorrt_node {
            subscriber /input: "sensor_msgs/msg/Image"
            publisher /output1: "std_msgs/msg/Float32MultiArray"
            publisher /output2: "std_msgs/msg/Float32MultiArray"
            
            onnx_model model1 {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
            
            onnx_model model2 {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,10]"
                device: cuda
                optimization: tensorrt
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        node = ast.nodes[0]
        assert len(node.content.onnx_models) == 2
        assert node.content.onnx_models[0].name == "model1"
        assert node.content.onnx_models[1].name == "model2"
    
    def test_tensorrt_pipeline_integration(self, parser):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT integration within pipeline contexts."""
        
        # Test 1: Pipeline with TensorRT stages
        dsl_code = '''
        onnx_model pipeline_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        
        pipeline tensorrt_pipeline {
            stage preprocessing {
                input: "raw_input"
                output: "preprocessed_input"
                method: "preprocess"
                topic: /preprocessing/result
            }
            
            stage inference {
                input: "preprocessed_input"
                output: "inference_result"
                method: "run_inference"
                onnx_model: "pipeline_model"
                topic: /inference/result
            }
            
            stage postprocessing {
                input: "inference_result"
                output: "final_result"
                method: "postprocess"
                topic: /postprocessing/result
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.onnx_models) == 1
        assert len(ast.pipelines) == 1
        
        pipeline = ast.pipelines[0]
        assert pipeline.name == "tensorrt_pipeline"
        assert len(pipeline.content.stages) == 3
        
        # Check that inference stage has ONNX model reference
        inference_stage = pipeline.content.stages[1]
        assert len(inference_stage.content.onnx_models) == 1
        assert inference_stage.content.onnx_models[0].model_name == "pipeline_model"
    
    def test_tensorrt_memory_management(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT memory management features."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Memory management methods
        model = OnnxModelNode(
            name="memory_test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "memory_node")
        impl_content = generated_files[str(test_output_dir / "memory_node_onnx.cpp")]
        
        # Check memory management features
        assert "allocate_cuda_memory" in impl_content
        assert "free_cuda_memory" in impl_content
        assert "copy_to_cuda" in impl_content
        assert "copy_from_cuda" in impl_content
        assert "cuda_memory_pool_" in impl_content
        
        # Check CUDA memory allocation
        assert "cudaMalloc" in impl_content
        assert "cudaFree" in impl_content
        assert "cudaMemcpy" in impl_content
    
    def test_tensorrt_configuration_validation(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT configuration validation."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Valid configuration
        valid_model = OnnxModelNode(
            name="valid_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(valid_model, "valid_node")
        impl_content = generated_files[str(test_output_dir / "valid_node_onnx.cpp")]
        
        # Check validation methods
        assert "validate_tensorrt_engine()" in impl_content
        assert "create_tensorrt_cache_directory()" in impl_content
        
        # Test 2: Configuration with different workspace sizes
        large_workspace_model = OnnxModelNode(
            name="large_workspace_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        # Should generate same workspace size (default)
        generated_files = generator.generate_onnx_integration(large_workspace_model, "large_workspace_node")
        impl_content = generated_files[str(test_output_dir / "large_workspace_node_onnx.cpp")]
        assert "trt_max_workspace_size = 1 << 30" in impl_content
    
    def test_tensorrt_runtime_controls(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT runtime control methods."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="runtime_control_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "runtime_node")
        impl_content = generated_files[str(test_output_dir / "runtime_node_onnx.cpp")]
        
        # Test 1: FP16 control
        assert "enable_tensorrt_fp16()" in impl_content
        assert "tensorrt_fp16_enabled_ = true" in impl_content
        
        # Test 2: INT8 control
        assert "enable_tensorrt_int8()" in impl_content
        assert "tensorrt_int8_enabled_ = true" in impl_content
        assert "tensorrt_fp16_enabled_ = false" in impl_content  # INT8 takes precedence
        
        # Test 3: Workspace size control
        assert "set_tensorrt_workspace_size(" in impl_content
        assert "tensorrt_workspace_size_ = size" in impl_content
        
        # Test 4: Cache control
        assert "clear_tensorrt_cache()" in impl_content
        assert "rm -rf" in impl_content  # Cache clearing command
    
    def test_tensorrt_error_recovery(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT error recovery scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="error_recovery_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "error_node")
        impl_content = generated_files[str(test_output_dir / "error_node_onnx.cpp")]
        
        # Test 1: Error handling in inference methods
        assert "try {" in impl_content
        assert "catch (const Ort::Exception& e)" in impl_content
        assert "catch (const std::exception& e)" in impl_content
        assert "std::cerr <<" in impl_content
        assert "return false;" in impl_content
        
        # Test 2: CUDA error handling
        assert "cudaError_t" in impl_content
        assert "cudaSuccess" in impl_content
        assert "cudaGetErrorString" in impl_content
        
        # Test 3: Memory allocation error handling
        assert "if (!cuda_input || !cuda_output)" in impl_content
        assert "return false;" in impl_content
    
    def test_tensorrt_performance_monitoring(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT performance monitoring features."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="perf_monitor_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "perf_monitor_node")
        impl_content = generated_files[str(test_output_dir / "perf_monitor_node_onnx.cpp")]
        
        # Test 1: Performance logging
        assert "std::cout <<" in impl_content
        assert "ONNX model initialized successfully" in impl_content
        assert "TensorRT FP16 optimization enabled" in impl_content
        assert "TensorRT INT8 optimization enabled" in impl_content
        assert "TensorRT workspace size set to" in impl_content
        assert "TensorRT cache cleared successfully" in impl_content
        
        # Test 2: Model information logging
        assert "Model:" in impl_content
        assert "Device:" in impl_content
        assert "Input:" in impl_content
        assert "Output:" in impl_content
        assert "shape:" in impl_content
    
    def test_tensorrt_integration_compatibility(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT integration compatibility with other features."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: TensorRT with CUDA kernels
        model = OnnxModelNode(
            name="cuda_kernel_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "cuda_kernel_node")
        impl_content = generated_files[str(test_output_dir / "cuda_kernel_node_onnx.cpp")]
        
        # Should include both TensorRT and CUDA features
        assert "OrtTensorRTProviderOptions" in impl_content
        assert "OrtCUDAProviderOptions" in impl_content
        assert "cuda_runtime.h" in impl_content
        assert "cudaSetDevice" in impl_content
        
        # Test 2: TensorRT with OpenCV integration
        header_content = generated_files[str(test_output_dir / "cuda_kernel_node_onnx.hpp")]
        assert "opencv2/opencv.hpp" in header_content
        
        # Test 3: TensorRT with ROS2 integration
        node_integration = generator.generate_node_integration(model, "cuda_kernel_node")
        assert "rclcpp::Node" in node_integration
        assert "RCLCPP_INFO" in node_integration
        assert "RCLCPP_ERROR" in node_integration
    
    def test_tensorrt_stress_testing(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT under stress conditions."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Multiple models with TensorRT
        models = []
        for i in range(5):
            model = OnnxModelNode(
                name=f"stress_model_{i}",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
            models.append(model)
        
        # Generate all models
        all_files = {}
        for i, model in enumerate(models):
            files = generator.generate_onnx_integration(model, f"stress_node_{i}")
            all_files.update(files)
        
        # Should generate 10 files (5 headers + 5 implementations)
        assert len(all_files) == 10
        
        # Test 2: Large model with TensorRT
        large_model = OnnxModelNode(
            name="stress_large_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,1024,1024]")],
                outputs=[OutputDefNode(name="output", type="float32[1,10000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(large_model, "stress_large_node")
        impl_content = generated_files[str(test_output_dir / "stress_large_node_onnx.cpp")]
        
        # Should handle large models correctly
        assert "trt_max_workspace_size = 1 << 30" in impl_content  # Still 1GB default
    
    def test_tensorrt_future_compatibility(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_tensorrt()
        """Test TensorRT features for future compatibility."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test 1: Model with potential future TensorRT features
        future_model = OnnxModelNode(
            name="future_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(future_model, "future_node")
        impl_content = generated_files[str(test_output_dir / "future_node_onnx.cpp")]
        
        # Check for extensible configuration structure
        assert "initialize_tensorrt_options" in impl_content
        assert "OrtTensorRTProviderOptions trt_options" in impl_content
        
        # Test 2: Configuration that can be extended
        assert "trt_options." in impl_content  # Shows extensible structure
        
        # Test 3: Error handling that can accommodate new features
        assert "catch (const Ort::Exception& e)" in impl_content
        assert "catch (const std::exception& e)" in impl_content 