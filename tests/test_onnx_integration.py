"""Tests for ONNX integration functionality."""

import pytest
from pathlib import Path
import tempfile
import shutil

from robodsl.parsers.lark_parser import RoboDSLParser
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.core.ast import OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, DeviceNode, OptimizationNode


class TestOnnxIntegration:
    """Test ONNX integration parsing and generation."""
    
    def setup_method(self):
        """Set up test environment."""
        self.parser = RoboDSLParser()
        self.temp_dir = tempfile.mkdtemp()
        self.generator = OnnxIntegrationGenerator(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_onnx_model_parsing(self):
        """Test parsing of ONNX model definitions."""
        # Test basic ONNX model
        dsl_code = '''
        onnx_model "test_model" {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        '''
        
        ast = self.parser.parse(dsl_code)
        
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
    
    def test_onnx_model_generation(self):
        """Test ONNX integration code generation."""
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
        generated_files = self.generator.generate_onnx_integration(model, "test_node")
        
        # Check that files were generated
        assert len(generated_files) == 2  # header and implementation
        
        # Check header file
        header_path = str(Path(self.temp_dir) / "test_node_onnx.hpp")
        assert header_path in generated_files
        
        header_content = generated_files[header_path]
        assert "class test_nodeOnnxInference" in header_content
        assert "#include <onnxruntime_cxx_api.h>" in header_content
        assert "#include <opencv2/opencv.hpp>" in header_content
        
        # Check implementation file
        impl_path = str(Path(self.temp_dir) / "test_node_onnx.cpp")
        assert impl_path in generated_files
        
        impl_content = generated_files[impl_path]
        assert "test_nodeOnnxInference::test_nodeOnnxInference" in impl_content
        assert "device_type_(\"cuda\")" in impl_content
        assert "tensorrt" in impl_content
    
    def test_onnx_model_without_optimization(self):
        """Test ONNX model without optimization settings."""
        dsl_code = '''
        onnx_model "simple_model" {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cpu
        }
        '''
        
        ast = self.parser.parse(dsl_code)
        
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        
        assert model.name == "simple_model"
        assert model.config.device.device == "cpu"
        assert len(model.config.optimizations) == 0
    
    def test_onnx_model_multiple_inputs_outputs(self):
        """Test ONNX model with multiple inputs and outputs."""
        dsl_code = '''
        onnx_model "multi_io_model" {
            input: "input1" -> "float32[1,3,224,224]"
            input: "input2" -> "float32[1,1,224,224]"
            output: "output1" -> "float32[1,1000]"
            output: "output2" -> "float32[1,10]"
            device: cuda
            optimization: openvino
        }
        '''
        
        ast = self.parser.parse(dsl_code)
        
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
    
    def test_cmake_integration_generation(self):
        """Test CMake integration generation."""
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda")
            )
        )
        
        cmake_content = self.generator.generate_cmake_integration(model, "test_node")
        
        assert "find_package(ONNXRuntime REQUIRED)" in cmake_content
        assert "find_package(OpenCV REQUIRED)" in cmake_content
        assert "test_node" in cmake_content
        assert "test_model.onnx" in cmake_content
    
    def test_python_integration_generation(self):
        """Test Python integration generation."""
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cpu")
            )
        )
        
        python_content = self.generator.generate_python_integration(model, "test_node")
        
        assert "class test_nodeOnnxInference" in python_content
        assert "import onnxruntime as ort" in python_content
        assert "import numpy as np" in python_content
        assert "self.device_type = \"cpu\"" in python_content
    
    def test_invalid_onnx_model(self):
        """Test handling of invalid ONNX model definitions."""
        # Test missing model name
        dsl_code = '''
        onnx_model {
            input: "input" -> "float32[1,3,224,224]"
        }
        '''
        
        with pytest.raises(Exception):
            self.parser.parse(dsl_code)
    
    def test_onnx_model_in_node_context(self):
        """Test ONNX model definition in the context of a complete DSL file."""
        dsl_code = '''
        include <rclcpp/rclcpp.hpp>
        
        node image_classifier {
            subscriber /camera/image_raw: "sensor_msgs/msg/Image"
            publisher /classification/result: "std_msgs/msg/Float32MultiArray"
            parameter "model_path": "resnet50.onnx"
            
            onnx_model "resnet50" {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        '''
        
        ast = self.parser.parse(dsl_code)
        
        # Check that node is parsed
        assert len(ast.nodes) == 1
        
        # Check node
        node = ast.nodes[0]
        assert node.name == "image_classifier"
        assert len(node.content.subscribers) == 1
        assert len(node.content.publishers) == 1
        assert len(node.content.parameters) == 1
        
        # Check ONNX model within node
        assert len(node.content.onnx_models) == 1
        model = node.content.onnx_models[0]
        assert model.name == "resnet50"
        assert model.config.device.device == "cuda"
        assert model.config.optimizations[0].optimization == "tensorrt"
    
    def test_cuda_integration_generation(self):
        """Test CUDA-specific ONNX integration generation."""
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cuda")
            )
        )
        
        generated_files = self.generator.generate_onnx_integration(model, "test_node")
        
        # Check header file for CUDA support
        header_path = str(Path(self.temp_dir) / "test_node_onnx.hpp")
        assert header_path in generated_files
        
        header_content = generated_files[header_path]
        assert "#include <cuda_runtime.h>" in header_content
        assert "run_inference_cuda" in header_content
        assert "allocate_cuda_memory" in header_content
        assert "free_cuda_memory" in header_content
        
        # Check implementation file for CUDA support
        impl_path = str(Path(self.temp_dir) / "test_node_onnx.cpp")
        assert impl_path in generated_files
        
        impl_content = generated_files[impl_path]
        assert "#include <cuda_runtime.h>" in impl_content
        assert "cudaSetDevice" in impl_content
        assert "cudaMalloc" in impl_content
        assert "cudaMemcpy" in impl_content
    
    def test_node_integration_generation(self):
        """Test node integration code generation."""
        model = OnnxModelNode(
            name="test_model",
            config=ModelConfigNode(
                inputs=[InputDefNode(name="input", type="float32[1,3,224,224]")],
                outputs=[OutputDefNode(name="output", type="float32[1,1000]")],
                device=DeviceNode(device="cpu")
            )
        )
        
        node_integration = self.generator.generate_node_integration(model, "test_node")
        
        assert "class test_nodeNode : public rclcpp::Node" in node_integration
        assert "std::unique_ptr<test_nodeOnnxInference> onnx_inference_" in node_integration
        assert "this->declare_parameter(\"model_path\"" in node_integration
        assert "onnx_inference_->initialize()" in node_integration
        assert "process_with_onnx" in node_integration
        assert "rclcpp::init(argc, argv)" in node_integration


if __name__ == "__main__":
    pytest.main([__file__]) 