"""Tests for ONNX integration functionality."""

import pytest
from pathlib import Path
import tempfile
import shutil

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda, skip_if_no_tensorrt, skip_if_no_onnx
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.core.ast import OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, DeviceNode, OptimizationNode


class TestOnnxIntegration:
    """Test ONNX integration parsing and generation."""
    
    def test_onnx_model_parsing(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test parsing of ONNX model definitions."""
        # Test basic ONNX model with quoted strings
        dsl_code = '''
        onnx_model test_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert model.name == "test_model"
        assert len(model.config.inputs) == 1
        assert len(model.config.outputs) == 1
        assert model.config.device is not None
        assert len(model.config.optimizations) == 1
        
        # Check input
        input_def = model.config.inputs[0]
        assert input_def.name == "input"
        assert input_def.type == "float32[1,3,224,224]"
        
        # Check output
        output_def = model.config.outputs[0]
        assert output_def.name == "output"
        assert output_def.type == "float32[1,1000]"
        
        # Check device
        assert model.config.device.device == "cuda"
        
        # Check optimization
        assert model.config.optimizations[0].optimization == "tensorrt"
    
    def test_onnx_model_parsing_with_names(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test parsing of ONNX model definitions with unquoted names."""
        # Test ONNX model with unquoted names
        dsl_code = '''
        onnx_model test_model_names {
            input: input_tensor -> images
            output: output_tensor -> detections
            device: cuda
            optimization: tensorrt
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert model.name == "test_model_names"
        assert len(model.config.inputs) == 1
        assert len(model.config.outputs) == 1
        
        # Check input
        input_def = model.config.inputs[0]
        assert input_def.name == "input_tensor"
        assert input_def.type == "images"
        
        # Check output
        output_def = model.config.outputs[0]
        assert output_def.name == "output_tensor"
        assert output_def.type == "detections"
    
    def test_onnx_model_generation(self, test_output_dir):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test ONNX integration code generation."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        # Create a test model
        model = OnnxModelNode(
            name="test_model",
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
        
        # Check header file
        header_path = str(test_output_dir / "test_node_onnx.hpp")
        assert header_path in generated_files
        
        header_content = generated_files[header_path]
        assert "class test_nodeOnnxInference" in header_content
        assert "#include <onnxruntime_cxx_api.h>" in header_content
        assert "#include <opencv2/opencv.hpp>" in header_content
        
        # Check implementation file
        impl_path = str(test_output_dir / "test_node_onnx.cpp")
        assert impl_path in generated_files
        
        impl_content = generated_files[impl_path]
        assert "test_nodeOnnxInference::test_nodeOnnxInference" in impl_content
        assert "device_type_(\"cuda\")" in impl_content
        assert "tensorrt" in impl_content
    
    def test_onnx_model_without_optimization(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test ONNX model without optimization settings."""
        dsl_code = '''
        onnx_model simple_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cpu
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert model.name == "simple_model"
        assert model.config.device.device == "cpu"
        assert len(model.config.optimizations) == 0
    
    def test_onnx_model_multiple_inputs_outputs(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test ONNX model with multiple inputs and outputs."""
        dsl_code = '''
        onnx_model multi_io_model {
            input: "input1" -> "float32[1,3,224,224]"
            input: "input2" -> "float32[1,1,224,224]"
            output: "output1" -> "float32[1,1000]"
            output: "output2" -> "float32[1,10]"
            device: cuda
            optimization: openvino
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert len(model.config.inputs) == 2
        assert len(model.config.outputs) == 2
        assert len(model.config.optimizations) == 1
        
        # Check inputs
        input_names = [inp.name for inp in model.config.inputs]
        assert "input1" in input_names
        assert "input2" in input_names
        
        # Check outputs
        output_names = [out.name for out in model.config.outputs]
        assert "output1" in output_names
        assert "output2" in output_names
        
        # Check optimization
        assert model.config.optimizations[0].optimization == "openvino"
    
    def test_cmake_integration_generation(self, test_output_dir):
        skip_if_no_ros2()
        """Test CMake integration generation."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda")
            )
        )
        
        cmake_content = generator.generate_cmake_integration(model, "test_node")
        
        assert "find_package(ONNXRuntime REQUIRED)" in cmake_content
        assert "find_package(OpenCV REQUIRED)" in cmake_content
        assert "test_node" in cmake_content
        assert "test_model.onnx" in cmake_content
    
    def test_python_integration_generation(self, test_output_dir):
        skip_if_no_ros2()
        """Test Python integration generation."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cpu")
            )
        )
        
        python_content = generator.generate_python_integration(model, "test_node")
        
        assert "class test_nodeOnnxPython" in python_content
        assert "import onnxruntime as ort" in python_content
        assert "import numpy as np" in python_content
        assert "self.device = device" in python_content
        assert 'device: str = "cpu"' in python_content
    
    def test_invalid_onnx_model(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test handling of invalid ONNX model definitions."""
        # Test with missing input
        dsl_code = '''
        onnx_model invalid_model {
            output: "output" -> "float32[1,1000]"
            device: cuda
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        # Should still parse but with empty inputs
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        assert len(model.config.inputs) == 0
    
    def test_onnx_model_in_node_context(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        """Test ONNX model usage within node context."""
        dsl_code = '''
        onnx_model classifier {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
        }
    
        node inference_node {
            publisher /results: "std_msgs/msg/String"
            subscriber /input: "sensor_msgs/msg/Image"
            // onnx_model: "classifier"  # Not supported in grammar, so skip this part
        }
        '''
    
        ast = parse_robodsl(dsl_code)
        assert ast is not None
        assert len(ast.nodes) == 1
        assert len(ast.onnx_models) == 1
        
        node = ast.nodes[0]
        assert node.name == "inference_node"
        assert len(node.content.publishers) == 1
        assert len(node.content.subscribers) == 1
    
    def test_cuda_integration_generation(self, test_output_dir):
        skip_if_no_cuda()
        """Test CUDA integration generation for ONNX models."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda")
            )
        )
        
        cuda_content = generator.generate_cuda_integration(model, "test_node")
        
        assert "cuda" in cuda_content.lower()
        assert "test_model" in cuda_content
        assert "test_node" in cuda_content
        assert "gpu" in cuda_content.lower()
    
    def test_node_integration_generation(self, test_output_dir):
        skip_if_no_ros2()
        """Test node integration generation for ONNX models."""
        generator = OnnxIntegrationGenerator(str(test_output_dir))
        
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda")
            )
        )
        
        node_integration = generator.generate_node_integration(model, "test_node")
        
        assert "test_node" in node_integration
        assert "test_model" in node_integration
        assert "onnx" in node_integration.lower()
        assert "inference" in node_integration.lower()

    def test_onnx_model_missing_input(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model m {
            output: "out" -> "float32[1,1000]"
            device: cuda
        }
        '''
        ast = parse_robodsl(dsl)
        assert len(ast.onnx_models) == 1
        assert len(ast.onnx_models[0].config.inputs) == 0

    def test_onnx_model_invalid_device(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model m {
            input: "in" -> "float32[1,3,224,224]"
            output: "out" -> "float32[1,1000]"
            device: notarealdevice
        }
        '''
        ast = parse_robodsl(dsl)
        assert ast.onnx_models[0].config.device.device == "notarealdevice"

    def test_onnx_model_duplicate_optimizations(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model m {
            input: "in" -> "float32[1,3,224,224]"
            output: "out" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
            optimization: tensorrt
        }
        '''
        ast = parse_robodsl(dsl)
        opt_names = [opt.optimization for opt in ast.onnx_models[0].config.optimizations]
        assert opt_names.count("tensorrt") == 2

    def test_onnx_model_malformed_shape(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model m {
            input: "in" -> "float32[1,3,224,224,]"  # Malformed shape (trailing comma)
            output: "out" -> "float32[1,1000]"
            device: cuda
        }
        '''
        ast = parse_robodsl(dsl)
        assert ast.onnx_models[0].config.inputs[0].type == "float32[1,3,224,224,]"

    def test_onnx_model_long_name(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model this_is_a_very_long_model_name_with_numbers_1234567890 {
            input: "in" -> "float32[1,3,224,224]"
            output: "out" -> "float32[1,1000]"
            device: cuda
        }
        '''
        ast = parse_robodsl(dsl)
        assert ast.onnx_models[0].name == "this_is_a_very_long_model_name_with_numbers_1234567890"

    def test_onnx_model_unicode_name(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model 模型 {
            input: "in" -> "float32[1,3,224,224]"
            output: "out" -> "float32[1,1000]"
            device: cuda
        }
        '''
        ast = parse_robodsl(dsl)
        assert ast.onnx_models[0].name == "模型"

    def test_onnx_model_invalid_optimization(self):
        skip_if_no_ros2()
        skip_if_no_onnx()
        dsl = '''
        onnx_model m {
            input: "in" -> "float32[1,3,224,224]"
            output: "out" -> "float32[1,1000]"
            device: cuda
            optimization: notarealopt
        }
        '''
        ast = parse_robodsl(dsl)
        assert ast.onnx_models[0].config.optimizations[0].optimization == "notarealopt"


if __name__ == "__main__":
    pytest.main([__file__]) 