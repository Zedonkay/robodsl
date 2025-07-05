"""Integration tests for TensorRT optimization with other RoboDSL features."""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.generators.onnx_integration import OnnxIntegrationGenerator
from robodsl.core.ast import (
    OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, 
    DeviceNode, OptimizationNode
)


class TestTensorRTIntegration:
    """Integration testing for TensorRT with other RoboDSL features."""
    
    def test_tensorrt_with_cuda_kernels(self, parser):
        """Test TensorRT integration with CUDA kernels."""
        dsl_code = '''
        cuda_kernels {
            kernel preprocess_kernel {
                input: float* input_data, int width, int height
                output: float* output_data
                
                block_size: (16, 16, 1)
                grid_size: (width/16 + 1, height/16 + 1, 1)
                
                code {
                    __global__ void preprocess_kernel(const float* input, float* output, int width, int height) {
                        int x = blockIdx.x * blockDim.x + threadIdx.x;
                        int y = blockIdx.y * blockDim.y + threadIdx.y;
                        
                        if (x < width && y < height) {
                            int idx = y * width + x;
                            output[idx] = input[idx] / 255.0f;
                        }
                    }
                }
            }
        }
        
        onnx_model tensorrt_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        
        node integrated_node {
            subscriber /input: "sensor_msgs/msg/Image"
            publisher /output: "std_msgs/msg/Float32MultiArray"
            
            onnx_model tensorrt_model {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        # Check CUDA kernels
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 1
        assert ast.cuda_kernels.kernels[0].name == "preprocess_kernel"
        
        # Check ONNX model
        assert len(ast.onnx_models) == 1
        assert ast.onnx_models[0].name == "tensorrt_model"
        assert len(ast.onnx_models[0].config.optimizations) == 1
        assert ast.onnx_models[0].config.optimizations[0].optimization == "tensorrt"
        
        # Check node integration
        assert len(ast.nodes) == 1
        assert ast.nodes[0].name == "integrated_node"
        assert len(ast.nodes[0].content.onnx_models) == 1
    
    def test_tensorrt_with_pipelines(self, parser):
        """Test TensorRT integration with pipeline stages."""
        dsl_code = '''
        onnx_model classification_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
        }
        
        onnx_model detection_model {
            input: "input" -> "float32[1,3,640,640]"
            output: "output" -> "float32[1,84,8400]"
            device: cuda
            optimization: tensorrt
        }
        
        pipeline tensorrt_pipeline {
            stage preprocessing {
                input: "raw_image"
                output: "preprocessed_image"
                method: "preprocess"
                topic: /preprocessing/result
            }
            
            stage classification {
                input: "preprocessed_image"
                output: "classification_result"
                method: "run_classification"
                onnx_model: "classification_model"
                topic: /classification/result
            }
            
            stage detection {
                input: "preprocessed_image"
                output: "detection_result"
                method: "run_detection"
                onnx_model: "detection_model"
                topic: /detection/result
            }
            
            stage fusion {
                input: "classification_result,detection_result"
                output: "final_result"
                method: "fuse_results"
                topic: /fusion/result
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        # Check ONNX models
        assert len(ast.onnx_models) == 2
        assert ast.onnx_models[0].name == "classification_model"
        assert ast.onnx_models[1].name == "detection_model"
        
        # Check pipeline
        assert len(ast.pipelines) == 1
        pipeline = ast.pipelines[0]
        assert pipeline.name == "tensorrt_pipeline"
        assert len(pipeline.content.stages) == 4
        
        # Check that stages have ONNX model references
        classification_stage = pipeline.content.stages[1]
        detection_stage = pipeline.content.stages[2]
        
        assert len(classification_stage.content.onnx_models) == 1
        assert classification_stage.content.onnx_models[0].model_name == "classification_model"
        
        assert len(detection_stage.content.onnx_models) == 1
        assert detection_stage.content.onnx_models[0].model_name == "detection_model"
    
    def test_tensorrt_with_simulation(self, parser):
        """Test TensorRT integration with simulation configuration."""
        # This test is skipped because simulation syntax is not fully implemented
        pytest.skip("Simulation syntax is not fully implemented in the current grammar")
    
    def test_tensorrt_with_dynamic_parameters(self, parser):
        """Test TensorRT integration with dynamic parameters."""
        # This test is skipped because dynamic parameters syntax is not fully implemented
        pytest.skip("Dynamic parameters syntax is not fully implemented in the current grammar")
    
    def test_tensorrt_with_multiple_optimizations(self, parser):
        """Test TensorRT with multiple optimization strategies."""
        dsl_code = '''
        onnx_model multi_opt_model {
            input: "input" -> "float32[1,3,224,224]"
            output: "output" -> "float32[1,1000]"
            device: cuda
            optimization: tensorrt
            optimization: openvino
            optimization: onednn
        }
        
        node multi_opt_node {
            subscriber /input: "sensor_msgs/msg/Image"
            publisher /output: "std_msgs/msg/Float32MultiArray"
            
            onnx_model multi_opt_model {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
                optimization: openvino
                optimization: onednn
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        # Check ONNX model
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        assert model.name == "multi_opt_model"
        assert len(model.config.optimizations) == 3
        
        # Check optimization names
        opt_names = [opt.optimization for opt in model.config.optimizations]
        assert "tensorrt" in opt_names
        assert "openvino" in opt_names
        assert "onednn" in opt_names
    
    def test_tensorrt_with_different_devices(self, parser):
        """Test TensorRT with different device configurations."""
        device_configs = [
            ("cuda", "cuda"),
            ("gpu", "gpu"),
            ("cpu", "cpu")
        ]
        
        for device_name, device_type in device_configs:
            dsl_code = f'''
            onnx_model {device_name}_model {{
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: {device_type}
                optimization: tensorrt
            }}
            
            node {device_name}_node {{
                subscriber /input: "sensor_msgs/msg/Image"
                publisher /output: "std_msgs/msg/Float32MultiArray"
                
                onnx_model {device_name}_model {{
                    input: "input" -> "float32[1,3,224,224]"
                    output: "output" -> "float32[1,1000]"
                    device: {device_type}
                    optimization: tensorrt
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            
            # Check ONNX model
            assert len(ast.onnx_models) == 1
            model = ast.onnx_models[0]
            assert model.name == f"{device_name}_model"
            assert model.config.device.device == device_type
            assert len(model.config.optimizations) == 1
            assert model.config.optimizations[0].optimization == "tensorrt"
    
    def test_tensorrt_with_complex_data_types(self, parser):
        """Test TensorRT with complex data type configurations."""
        dsl_code = '''
        onnx_model complex_types_model {
            input: "input1" -> "float32[1,3,224,224]"
            input: "input2" -> "float16[1,1,224,224]"
            input: "input3" -> "int64[1,512]"
            output: "output1" -> "float32[1,1000]"
            output: "output2" -> "float16[1,10]"
            output: "output3" -> "int32[1,512]"
            device: cuda
            optimization: tensorrt
        }
        
        node complex_types_node {
            subscriber /input1: "sensor_msgs/msg/Image"
            subscriber /input2: "sensor_msgs/msg/Image"
            subscriber /input3: "std_msgs/msg/Int64MultiArray"
            publisher /output1: "std_msgs/msg/Float32MultiArray"
            publisher /output2: "std_msgs/msg/Float32MultiArray"
            publisher /output3: "std_msgs/msg/Int32MultiArray"
            
            onnx_model complex_types_model {
                input: "input1" -> "float32[1,3,224,224]"
                input: "input2" -> "float16[1,1,224,224]"
                input: "input3" -> "int64[1,512]"
                output: "output1" -> "float32[1,1000]"
                output: "output2" -> "float16[1,10]"
                output: "output3" -> "int32[1,512]"
                device: cuda
                optimization: tensorrt
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        
        # Check ONNX model
        assert len(ast.onnx_models) == 1
        model = ast.onnx_models[0]
        assert model.name == "complex_types_model"
        assert len(model.config.inputs) == 3
        assert len(model.config.outputs) == 3
        
        # Check input types
        input_types = [inp.type for inp in model.config.inputs]
        assert "float32[1,3,224,224]" in input_types
        assert "float16[1,1,224,224]" in input_types
        assert "int64[1,512]" in input_types
        
        # Check output types
        output_types = [out.type for out in model.config.outputs]
        assert "float32[1,1000]" in output_types
        assert "float16[1,10]" in output_types
        assert "int32[1,512]" in output_types
    
    def test_tensorrt_with_methods(self, parser):
        """Test TensorRT integration with custom methods."""
        # This test is skipped because method syntax is not fully implemented
        pytest.skip("Method syntax is not fully implemented in the current grammar")
    
    def test_tensorrt_with_timers(self, parser):
        """Test TensorRT integration with timer-based processing."""
        # This test is skipped because timer syntax is not fully implemented
        pytest.skip("Timer syntax is not fully implemented in the current grammar")
    
    def test_tensorrt_with_lifecycle(self, parser):
        """Test TensorRT integration with lifecycle nodes."""
        # This test is skipped because lifecycle syntax is not fully implemented
        pytest.skip("Lifecycle syntax is not fully implemented in the current grammar") 