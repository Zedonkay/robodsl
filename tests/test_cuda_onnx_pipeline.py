import pytest
import tempfile
import os
from robodsl.parsers.lark_parser import RoboDSLParser
from robodsl.generators.main_generator import MainGenerator
from robodsl.core.ast import StageCudaKernelNode, StageOnnxModelNode, KernelNode, OnnxModelNode


class TestCudaOnnxPipelineIntegration:
    """Test CUDA and ONNX integration in pipeline stages."""
    
    def test_pipeline_with_cuda_kernels(self, parser, generator):
        """Test pipeline generation with CUDA kernels."""
        robodsl_code = """
        cuda_kernels {
            kernel test_kernel {
                input: float* input_data, int size
                output: float* output_data
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output_data[idx] = input_data[idx] * 2.0f;
                    }
                }
            }
        }
        
        pipeline test_pipeline {
            stage processing {
                input: "input_data"
                output: "output_data"
                method: "process"
                cuda_kernel: "test_kernel"
                topic: /test/topic
            }
        }
        """
        
        # Parse the code
        ast = parser.parse(robodsl_code)
        assert ast is not None
        
        # Check that CUDA kernels are parsed
        assert len(ast.cuda_kernels.kernels) == 1
        assert ast.cuda_kernels.kernels[0].name == "test_kernel"
        
        # Check that pipeline stages have CUDA kernels
        pipeline = ast.pipelines[0]
        stage = pipeline.content.stages[0]
        assert len(stage.content.cuda_kernels) == 1
        assert stage.content.cuda_kernels[0].kernel_name == "test_kernel"
        
        # Generate code
        files = generator.generate(ast)
        
        # Check that CUDA integration files are generated
        # The files are now Path objects, so we need to check the string representation
        file_paths = [str(f) for f in files]
        print('DEBUG: Generated file paths:', file_paths)
        cuda_header_path = "include/robodsl_project/processing_cuda.hpp"
        cuda_impl_path = "src/processing_cuda.cpp"
        
        assert any(p.endswith(cuda_header_path) for p in file_paths)
        assert any(p.endswith(cuda_impl_path) for p in file_paths)
        
        # Check CUDA header content
        cuda_header_file = next(f for f in files if str(f).endswith("processing_cuda.hpp"))
        cuda_header = cuda_header_file.read_text()
        assert "class processingCudaManager" in cuda_header
        assert "test_kernel_kernel_" in cuda_header
        assert "cuda_runtime.h" in cuda_header
        
        # Check CUDA implementation content
        cuda_impl_file = next(f for f in files if str(f).endswith("processing_cuda.cpp"))
        cuda_impl = cuda_impl_file.read_text()
        assert "processingCudaManager::processingCudaManager" in cuda_impl
        assert "cudaSetDevice" in cuda_impl
        assert "test_kernel" in cuda_impl
    
    def test_pipeline_with_onnx_models(self, parser, generator):
        """Test pipeline generation with ONNX models."""
        robodsl_code = """
        onnx_model test_model {
            config {
                input: "input" -> "float32"
                output: "output" -> "float32"
                device: gpu
                optimization: tensorrt
            }
        }
        
        pipeline test_pipeline {
            stage inference {
                input: "input_data"
                output: "output_data"
                method: "run_inference"
                onnx_model: "test_model"
                topic: /inference/topic
            }
        }
        """
        
        # Parse the code
        ast = parser.parse(robodsl_code)
        assert ast is not None
        
        # Check that ONNX models are parsed
        assert len(ast.onnx_models) == 1
        assert ast.onnx_models[0].name == "test_model"
        
        # Check that pipeline stages have ONNX models
        pipeline = ast.pipelines[0]
        stage = pipeline.content.stages[0]
        assert len(stage.content.onnx_models) == 1
        assert stage.content.onnx_models[0].model_name == "test_model"
        
        # Generate code
        files = generator.generate(ast)
        
        # Check that ONNX integration files are generated
        file_paths = [str(f) for f in files]
        print('DEBUG: Generated file paths:', file_paths)
        onnx_header_path = "include/robodsl_project/inference_onnx.hpp"
        onnx_impl_path = "src/inference_onnx.cpp"
        
        assert any(p.endswith(onnx_header_path) for p in file_paths)
        assert any(p.endswith(onnx_impl_path) for p in file_paths)
        
        # Check ONNX header content
        onnx_header_file = next(f for f in files if str(f).endswith("inference_onnx.hpp"))
        onnx_header = onnx_header_file.read_text()
        assert "class inferenceOnnxManager" in onnx_header
        assert "onnxruntime_cxx_api.h" in onnx_header
        assert "test_model_model_path_" in onnx_header
        
        # Check ONNX implementation content
        onnx_impl_file = next(f for f in files if str(f).endswith("inference_onnx.cpp"))
        onnx_impl = onnx_impl_file.read_text()
        assert "inferenceOnnxManager::inferenceOnnxManager" in onnx_impl
        assert "Ort::Env" in onnx_impl
        assert "test_model" in onnx_impl
    
    def test_pipeline_with_both_cuda_and_onnx(self, parser, generator):
        """Test pipeline generation with both CUDA kernels and ONNX models."""
        robodsl_code = """
        cuda_kernels {
            kernel preprocess {
                input: float* input_data, int size
                output: float* output_data
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output_data[idx] = input_data[idx] - 127.5f;
                    }
                }
            }
        }
        
        onnx_model detector {
            config {
                input: "input" -> "float32"
                output: "output" -> "float32"
                device: gpu
            }
        }
        
        pipeline ml_pipeline {
            stage preprocessing {
                input: "raw_data"
                output: "preprocessed_data"
                method: "preprocess"
                cuda_kernel: "preprocess"
                topic: /preprocessing/topic
            }
            
            stage detection {
                input: "preprocessed_data"
                output: "detection_results"
                method: "detect"
                onnx_model: "detector"
                topic: /detection/topic
            }
        }
        """
        
        # Parse the code
        ast = parser.parse(robodsl_code)
        assert ast is not None
        
        # Check that both CUDA and ONNX are parsed
        assert len(ast.cuda_kernels.kernels) == 1
        assert len(ast.onnx_models) == 1
        
        # Check pipeline stages
        pipeline = ast.pipelines[0]
        assert len(pipeline.content.stages) == 2
        
        preprocessing_stage = pipeline.content.stages[0]
        detection_stage = pipeline.content.stages[1]
        
        assert len(preprocessing_stage.content.cuda_kernels) == 1
        assert len(detection_stage.content.onnx_models) == 1
        
        # Generate code
        files = generator.generate(ast)
        
        # Check that all expected files are generated
        file_paths = [str(f) for f in files]
        
        # CUDA files
        assert any(p.endswith("include/robodsl_project/preprocessing_cuda.hpp") for p in file_paths)
        assert any(p.endswith("src/preprocessing_cuda.cpp") for p in file_paths)
        
        # ONNX files
        assert any(p.endswith("include/robodsl_project/detection_onnx.hpp") for p in file_paths)
        assert any(p.endswith("src/detection_onnx.cpp") for p in file_paths)
        
        # Check CUDA header content
        cuda_header_file = next(f for f in files if str(f).endswith("preprocessing_cuda.hpp"))
        cuda_header = cuda_header_file.read_text()
        assert "class preprocessingCudaManager" in cuda_header
        assert "preprocess_kernel_" in cuda_header
        
        # Check ONNX header content
        onnx_header_file = next(f for f in files if str(f).endswith("detection_onnx.hpp"))
        onnx_header = onnx_header_file.read_text()
        assert "class detectionOnnxManager" in onnx_header
        assert "detector_model_path_" in onnx_header
    
    def test_ast_builder_cuda_onnx_nodes(self, parser):
        """Test AST builder creates correct CUDA and ONNX node types."""
        robodsl_code = """
        cuda_kernels {
            kernel test_kernel {
                input: float* data, int size
                output: float* result
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        result[idx] = data[idx] * 2.0f;
                    }
                }
            }
        }
        
        onnx_model test_model {
            config {
                input: "input" -> "float32"
                output: "output" -> "float32"
                device: gpu
            }
        }
        
        pipeline test_pipeline {
            stage cuda_stage {
                input: "input_data"
                output: "cuda_output"
                method: "cuda_process"
                cuda_kernel: "test_kernel"
                topic: /cuda/topic
            }
            
            stage onnx_stage {
                input: "cuda_output"
                output: "final_output"
                method: "onnx_inference"
                onnx_model: "test_model"
                topic: /onnx/topic
            }
        }
        """
        
        ast = parser.parse(robodsl_code)
        assert ast is not None
        
        # Check CUDA kernel node (standalone kernels)
        cuda_kernel = ast.cuda_kernels.kernels[0]
        assert isinstance(cuda_kernel, KernelNode)
        assert cuda_kernel.name == "test_kernel"
        assert len(cuda_kernel.content.parameters) == 3  # input, input, output
        
        # Check ONNX model node
        onnx_model = ast.onnx_models[0]
        assert isinstance(onnx_model, OnnxModelNode)
        assert onnx_model.name == "test_model"
        assert len(onnx_model.config.inputs) == 1
        assert len(onnx_model.config.outputs) == 1
        
        # Check pipeline stage CUDA kernel references
        pipeline = ast.pipelines[0]
        cuda_stage = pipeline.content.stages[0]
        assert len(cuda_stage.content.cuda_kernels) == 1
        assert isinstance(cuda_stage.content.cuda_kernels[0], StageCudaKernelNode)
        assert cuda_stage.content.cuda_kernels[0].kernel_name == "test_kernel"
    
    def test_comprehensive_ml_pipeline(self, parser, generator):
        """Test a comprehensive ML pipeline with preprocessing, inference, and postprocessing."""
        robodsl_code = """
        cuda_kernels {
            kernel preprocess {
                input: uint8* raw_image, int width, int height
                output: float* normalized_image
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    int total_pixels = width * height * 3;
                    if (idx < total_pixels) {
                        normalized_image[idx] = (float)raw_image[idx] / 255.0f;
                    }
                }
            }
            
            kernel postprocess {
                input: float* raw_detections, int num_detections
                output: float* filtered_detections
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < num_detections) {
                        if (raw_detections[idx * 6 + 4] > 0.5f) {
                            filtered_detections[idx] = raw_detections[idx];
                        }
                    }
                }
            }
        }
        
        onnx_model yolo_detector {
            config {
                input: "images" -> "float32"
                output: "output0" -> "float32"
                device: gpu
                optimization: tensorrt
            }
        }
        
        pipeline ml_detection_pipeline {
            stage preprocessing {
                input: "camera_image"
                output: "normalized_image"
                method: "normalize_image"
                cuda_kernel: "preprocess"
                topic: /preprocessing/image
            }
            
            stage detection {
                input: "normalized_image"
                output: "raw_detections"
                method: "detect_objects"
                onnx_model: "yolo_detector"
                topic: /detection/results
            }
            
            stage postprocessing {
                input: "raw_detections"
                output: "final_detections"
                method: "filter_detections"
                cuda_kernel: "postprocess"
                topic: /postprocessing/filtered
            }
        }
        """
        
        # Parse the code
        ast = parser.parse(robodsl_code)
        assert ast is not None
        
        # Check that all components are parsed correctly
        assert len(ast.cuda_kernels.kernels) == 2
        assert len(ast.onnx_models) == 1
        assert len(ast.pipelines) == 1
        
        pipeline = ast.pipelines[0]
        assert len(pipeline.content.stages) == 3
        
        # Check stage configurations
        preprocessing_stage = pipeline.content.stages[0]
        detection_stage = pipeline.content.stages[1]
        postprocessing_stage = pipeline.content.stages[2]
        
        assert len(preprocessing_stage.content.cuda_kernels) == 1
        assert len(detection_stage.content.onnx_models) == 1
        assert len(postprocessing_stage.content.cuda_kernels) == 1
        
        # Generate code
        files = generator.generate(ast)
        
        # Check that all expected files are generated
        file_paths = [str(f) for f in files]
        
        # CUDA files
        assert any(p.endswith("include/robodsl_project/preprocessing_cuda.hpp") for p in file_paths)
        assert any(p.endswith("src/preprocessing_cuda.cpp") for p in file_paths)
        assert any(p.endswith("include/robodsl_project/postprocessing_cuda.hpp") for p in file_paths)
        assert any(p.endswith("src/postprocessing_cuda.cpp") for p in file_paths)
        
        # ONNX files
        assert any(p.endswith("include/robodsl_project/detection_onnx.hpp") for p in file_paths)
        assert any(p.endswith("src/detection_onnx.cpp") for p in file_paths)
        
        # Check CUDA header content
        cuda_header_file = next(f for f in files if str(f).endswith("preprocessing_cuda.hpp"))
        cuda_header = cuda_header_file.read_text()
        assert "class preprocessingCudaManager" in cuda_header
        assert "preprocess_kernel_" in cuda_header

        # Check ONNX integration
        onnx_header_file = next(f for f in files if str(f).endswith("detection_onnx.hpp"))
        onnx_header = onnx_header_file.read_text()
        assert "class detectionOnnxManager" in onnx_header
        assert "yolo_detector_model_path_" in onnx_header 