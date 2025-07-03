"""Performance tests for TensorRT optimization in RoboDSL.

This module focuses on performance testing, benchmarking, and optimization validation.
"""

import pytest
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from robodsl.parsers.lark_parser import RoboDSLParser
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.core.ast import (
    OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, 
    DeviceNode, OptimizationNode
)


class TestTensorRTPerformance:
    """Performance testing for TensorRT optimization."""
    
    @pytest.fixture
    def performance_models(self):
        """Create models for performance testing."""
        return {
            "resnet50": OnnxModelNode(
                name="resnet50",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            ),
            "yolo_v8": OnnxModelNode(
                name="yolo_v8",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="images", type="float32[1,3,640,640]")],
                    outputs=[OutputDefNode(name="output0", type="float32[1,84,8400]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            ),
            "bert": OnnxModelNode(
                name="bert",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input_ids", type="int64[1,512]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,512,768]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
        }
    
    def test_tensorrt_code_generation_performance(self, test_output_dir, performance_models):
        """Test performance of code generation for TensorRT models."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        generation_times = {}
        
        for model_name, model in performance_models.items():
            start_time = time.time()
            
            # Generate TensorRT integration code
            generated_files = generator.generate_onnx_integration(model, f"{model_name}_node")
            
            end_time = time.time()
            generation_time = end_time - start_time
            generation_times[model_name] = generation_time
            
            # Verify generation completed successfully
            assert len(generated_files) == 2  # header + implementation
            assert f"{model_name}_node_onnx.hpp" in str(list(generated_files.keys())[0])
            assert f"{model_name}_node_onnx.cpp" in str(list(generated_files.keys())[1])
        
        # Performance assertions
        for model_name, gen_time in generation_times.items():
            assert gen_time < 1.0, f"Code generation for {model_name} took too long: {gen_time:.3f}s"
        
        # Verify all models generated TensorRT code
        for model_name, model in performance_models.items():
            # Get the generated files for this specific model
            model_files = generator.generate_onnx_integration(model, f"{model_name}_node")
            impl_content = model_files[str(test_output_dir / f"{model_name}_node_onnx.cpp")]
            assert "OrtTensorRTProviderOptions" in impl_content
            assert "AppendExecutionProvider_TensorRT" in impl_content
    
    def test_tensorrt_optimization_levels(self, test_output_dir):
        """Test different TensorRT optimization levels."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test maximum optimization configuration
        max_opt_model = OnnxModelNode(
            name="max_opt_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(max_opt_model, "max_opt_node")
        impl_content = generated_files[str(test_output_dir / "max_opt_node_onnx.cpp")]
        
        # Check for maximum optimization settings
        assert "trt_builder_optimization_level = 3" in impl_content
        assert "trt_optimization_level = 3" in impl_content
        assert "trt_fp16_enable = true" in impl_content
        assert "trt_engine_cache_enable = true" in impl_content
        assert "trt_timing_cache_enable = true" in impl_content
        assert "trt_build_heuristics_enable = true" in impl_content
    
    def test_tensorrt_memory_optimization(self, test_output_dir):
        """Test TensorRT memory optimization features."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test memory optimization configuration
        memory_opt_model = OnnxModelNode(
            name="memory_opt_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(memory_opt_model, "memory_opt_node")
        impl_content = generated_files[str(test_output_dir / "memory_opt_node_onnx.cpp")]
        
        # Check memory optimization features
        assert "trt_context_memory_sharing_enable = true" in impl_content
        assert "trt_max_workspace_size = 1 << 30" in impl_content  # 1GB
        assert "cuda_memory_pool_" in impl_content
        assert "allocate_cuda_memory" in impl_content
        assert "free_cuda_memory" in impl_content
    
    def test_tensorrt_precision_optimization(self, test_output_dir):
        """Test TensorRT precision optimization features."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test FP16 optimization
        fp16_model = OnnxModelNode(
            name="fp16_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(fp16_model, "fp16_node")
        impl_content = generated_files[str(test_output_dir / "fp16_node_onnx.cpp")]
        
        # Check FP16 optimization
        assert "trt_fp16_enable = true" in impl_content
        assert "enable_tensorrt_fp16()" in impl_content
        assert "tensorrt_fp16_enabled_ = true" in impl_content
        
        # Test INT8 optimization
        int8_model = OnnxModelNode(
            name="int8_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(int8_model, "int8_node")
        impl_content = generated_files[str(test_output_dir / "int8_node_onnx.cpp")]
        
        # Check INT8 optimization
        assert "trt_int8_enable = false" in impl_content  # Default disabled
        assert "enable_tensorrt_int8()" in impl_content
        assert "tensorrt_int8_enabled_ = true" in impl_content
        assert "tensorrt_fp16_enabled_ = false" in impl_content  # INT8 takes precedence
    
    def test_tensorrt_cache_performance(self, test_output_dir):
        """Test TensorRT cache performance features."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        cache_model = OnnxModelNode(
            name="cache_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(cache_model, "cache_node")
        impl_content = generated_files[str(test_output_dir / "cache_node_onnx.cpp")]
        
        # Check cache optimization features
        assert "trt_engine_cache_enable = true" in impl_content
        assert "trt_engine_cache_path = \"./trt_cache\"" in impl_content
        assert "trt_timing_cache_enable = true" in impl_content
        assert "clear_tensorrt_cache()" in impl_content
        assert "create_tensorrt_cache_directory()" in impl_content
        assert "validate_tensorrt_engine()" in impl_content
    
    def test_tensorrt_batch_processing(self, test_output_dir):
        """Test TensorRT batch processing capabilities."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test different batch sizes
        batch_sizes = [1, 4, 8, 16, 32]
        
        for batch_size in batch_sizes:
            batch_model = OnnxModelNode(
                name=f"batch_{batch_size}_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type=f"float32[{batch_size},3,224,224]")],
                    outputs=[OutputDefNode(name="output", type=f"float32[{batch_size},1000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
            
            generated_files = generator.generate_onnx_integration(batch_model, f"batch_{batch_size}_node")
            impl_content = generated_files[str(test_output_dir / f"batch_{batch_size}_node_onnx.cpp")]
            
            # Verify TensorRT configuration works for all batch sizes
            assert "OrtTensorRTProviderOptions" in impl_content
            assert "AppendExecutionProvider_TensorRT" in impl_content
            assert "trt_builder_optimization_level = 3" in impl_content
    
    def test_tensorrt_multi_gpu_support(self, test_output_dir):
        """Test TensorRT multi-GPU support."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test different GPU devices
        for device_id in [0, 1, 2, 3]:
            multi_gpu_model = OnnxModelNode(
                name=f"gpu_{device_id}_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
            
            generated_files = generator.generate_onnx_integration(multi_gpu_model, f"gpu_{device_id}_node")
            impl_content = generated_files[str(test_output_dir / f"gpu_{device_id}_node_onnx.cpp")]
            
            # Check device configuration
            assert "trt_options.device_id = 0" in impl_content  # Default to GPU 0
            assert "cuda_options.device_id = 0" in impl_content
    
    def test_tensorrt_workspace_optimization(self, test_output_dir):
        """Test TensorRT workspace size optimization."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Test different workspace sizes
        workspace_sizes = [1 << 28, 1 << 29, 1 << 30, 1 << 31]  # 256MB, 512MB, 1GB, 2GB
        
        for workspace_size in workspace_sizes:
            workspace_model = OnnxModelNode(
                name=f"workspace_{workspace_size}_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                    outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
            
            generated_files = generator.generate_onnx_integration(workspace_model, f"workspace_{workspace_size}_node")
            impl_content = generated_files[str(test_output_dir / f"workspace_{workspace_size}_node_onnx.cpp")]
            
            # Check workspace configuration (should use default 1GB)
            assert "trt_max_workspace_size = 1 << 30" in impl_content
            assert "set_tensorrt_workspace_size(" in impl_content
    
    def test_tensorrt_inference_methods_performance(self, test_output_dir):
        """Test performance of different TensorRT inference methods."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="inference_methods_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "inference_methods_node")
        impl_content = generated_files[str(test_output_dir / "inference_methods_node_onnx.cpp")]
        
        # Check all inference methods are generated
        inference_methods = [
            "run_inference_tensorrt(",
            "run_inference_tensorrt_cuda(",
            "run_inference(",
            "run_inference_cuda("
        ]
        
        for method in inference_methods:
            assert method in impl_content, f"Method {method} not found"
        
        # Check TensorRT-specific optimizations in inference methods
        assert "session_.Run(" in impl_content
        assert "Ort::RunOptions{nullptr}" in impl_content
        assert "input_names.data()" in impl_content
        assert "output_names.data()" in impl_content
    
    def test_tensorrt_error_handling_performance(self, test_output_dir):
        """Test performance of TensorRT error handling."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="error_handling_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "error_handling_node")
        impl_content = generated_files[str(test_output_dir / "error_handling_node_onnx.cpp")]
        
        # Check comprehensive error handling
        error_handling_features = [
            "try {",
            "catch (const Ort::Exception& e)",
            "catch (const std::exception& e)",
            "std::cerr <<",
            "return false;",
            "cudaError_t",
            "cudaSuccess",
            "cudaGetErrorString"
        ]
        
        for feature in error_handling_features:
            assert feature in impl_content, f"Error handling feature {feature} not found"
    
    def test_tensorrt_initialization_performance(self, test_output_dir):
        """Test performance of TensorRT initialization."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="init_performance_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "init_performance_node")
        impl_content = generated_files[str(test_output_dir / "init_performance_node_onnx.cpp")]
        
        # Check initialization performance features
        init_features = [
            "session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL)",
            "session_options.EnableMemoryPattern()",
            "session_options.EnableCpuMemArena()",
            "session_options.SetIntraOpNumThreads(1)",
            "session_options.SetInterOpNumThreads(1)",
            "session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL)"
        ]
        
        for feature in init_features:
            assert feature in impl_content, f"Initialization feature {feature} not found"
    
    def test_tensorrt_memory_management_performance(self, test_output_dir):
        """Test performance of TensorRT memory management."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="memory_performance_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda"),
                optimizations=[OptimizationNode(optimization="tensorrt")]
            )
        )
        
        generated_files = generator.generate_onnx_integration(model, "memory_performance_node")
        impl_content = generated_files[str(test_output_dir / "memory_performance_node_onnx.cpp")]
        
        # Check memory management performance features
        memory_features = [
            "allocate_cuda_memory(",
            "free_cuda_memory(",
            "copy_to_cuda(",
            "copy_from_cuda(",
            "cuda_memory_pool_",
            "cudaMalloc(",
            "cudaFree(",
            "cudaMemcpy("
        ]
        
        for feature in memory_features:
            assert feature in impl_content, f"Memory management feature {feature} not found"
    
    def test_tensorrt_benchmark_scenarios(self, test_output_dir):
        """Test TensorRT benchmark scenarios."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Benchmark different model configurations
        benchmark_configs = [
            ("small", "float32[1,3,112,112]", "float32[1,100]"),
            ("medium", "float32[1,3,224,224]", "float32[1,1000]"),
            ("large", "float32[1,3,512,512]", "float32[1,1000]"),
            ("xlarge", "float32[1,3,1024,1024]", "float32[1,1000]")
        ]
        
        for size, input_type, output_type in benchmark_configs:
            benchmark_model = OnnxModelNode(
                name=f"benchmark_{size}_model",
                config=ModelConfigNode(
                    inputs=[InputDefNode(name="input", type=input_type)],
                    outputs=[OutputDefNode(name="output", type=output_type)],
                    device=DeviceNode(device="cuda"),
                    optimizations=[OptimizationNode(optimization="tensorrt")]
                )
            )
            
            start_time = time.time()
            generated_files = generator.generate_onnx_integration(benchmark_model, f"benchmark_{size}_node")
            generation_time = time.time() - start_time
            
            # Performance assertion
            assert generation_time < 0.5, f"Benchmark generation for {size} took too long: {generation_time:.3f}s"
            
            # Verify TensorRT configuration
            impl_content = generated_files[str(test_output_dir / f"benchmark_{size}_node_onnx.cpp")]
            assert "OrtTensorRTProviderOptions" in impl_content
            assert "trt_builder_optimization_level = 3" in impl_content 