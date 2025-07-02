"""Test cases for use_kernel syntax and Thrust usage in RoboDSL."""

import pytest
from pathlib import Path
import tempfile
import shutil

from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.generators.main_generator import MainGenerator
from robodsl.core.ast import RoboDSLAST


class TestUseKernelSyntax:
    """Test cases for use_kernel syntax and Thrust integration."""

    def test_use_kernel_basic_parsing(self):
        """Test basic parsing of use_kernel syntax."""
        robodsl_code = """
        cuda_kernels {
            kernel test_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: true
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void test_kernel(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
        }

        node test_node_1 {
            use_kernel: "test_kernel"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        # Check that the global kernel was parsed
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "test_kernel"
        assert kernel.content.use_thrust is True
        
        # Check that the node was parsed with used_kernels
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "test_node_1"
        assert hasattr(node.content, 'used_kernels')
        assert len(node.content.used_kernels) == 1
        assert node.content.used_kernels[0] == "test_kernel"

    def test_use_kernel_multiple_kernels(self):
        """Test using multiple global kernels in a single node."""
        robodsl_code = """
        cuda_kernels {
            kernel kernel1 {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: false
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void kernel1(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
            
            kernel kernel2 {
                block_size: (128, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: true
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void kernel2(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] + 1.0f;
                        }
                    }
                }
            }
        }

        node multi_kernel_node {
            use_kernel: "kernel1"
            use_kernel: "kernel2"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        # Check global kernels
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 2
        
        kernel_names = [k.name for k in ast.cuda_kernels.kernels]
        assert "kernel1" in kernel_names
        assert "kernel2" in kernel_names
        
        # Check kernel1 doesn't use Thrust
        kernel1 = next(k for k in ast.cuda_kernels.kernels if k.name == "kernel1")
        assert kernel1.content.use_thrust is False
        
        # Check kernel2 uses Thrust
        kernel2 = next(k for k in ast.cuda_kernels.kernels if k.name == "kernel2")
        assert kernel2.content.use_thrust is True
        
        # Check node uses both kernels
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "multi_kernel_node"
        assert len(node.content.used_kernels) == 2
        assert "kernel1" in node.content.used_kernels
        assert "kernel2" in node.content.used_kernels

    def test_use_kernel_with_embedded_kernels(self):
        """Test using both embedded and global kernels in the same node."""
        robodsl_code = """
        cuda_kernels {
            kernel global_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: true
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void global_kernel(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
        }

        node mixed_kernel_node {
            use_kernel: "global_kernel"
            
            cuda_kernels {
                kernel embedded_kernel {
                    block_size: (128, 1, 1)
                    grid_size: (1, 1, 1)
                    use_thrust: false
                    input: float* data, int size
                    output: float* result
                    code: {
                        __global__ void embedded_kernel(float* data, float* result, int size) {
                            int idx = blockIdx.x * blockDim.x + threadIdx.x;
                            if (idx < size) {
                                result[idx] = data[idx] + 1.0f;
                            }
                        }
                    }
                }
            }
            
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        # Check global kernel
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 1
        global_kernel = ast.cuda_kernels.kernels[0]
        assert global_kernel.name == "global_kernel"
        assert global_kernel.content.use_thrust is True
        
        # Check node has both embedded and referenced kernels
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "mixed_kernel_node"
        
        # Check embedded kernel
        assert len(node.content.cuda_kernels) == 1
        embedded_kernel = node.content.cuda_kernels[0]
        assert embedded_kernel.name == "embedded_kernel"
        assert embedded_kernel.content.use_thrust is False
        
        # Check referenced kernel
        assert len(node.content.used_kernels) == 1
        assert node.content.used_kernels[0] == "global_kernel"

    def test_use_kernel_code_generation(self):
        """Test that use_kernel generates correct C++ code."""
        robodsl_code = """
        cuda_kernels {
            kernel test_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: true
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void test_kernel(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
        }

        node test_node_1 {
            use_kernel: "test_kernel"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        # Generate code
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MainGenerator(output_dir=temp_dir, debug=True)
            generated_files = generator.generate(ast)
            
            # Check that files were generated
            assert len(generated_files) > 0
            
            # Check for C++ node files
            cpp_files = [f for f in generated_files if f.name.endswith('.cpp') or f.name.endswith('.hpp')]
            assert len(cpp_files) >= 2  # At least header and source
            
            # Check for CUDA kernel files
            cuda_files = [f for f in generated_files if f.name.endswith('.cu') or f.name.endswith('.cuh')]
            assert len(cuda_files) >= 2  # At least .cu and .cuh for the kernel
            
            # Check that the generated C++ header includes Thrust
            header_file = next(f for f in cpp_files if f.name.endswith('.hpp'))
            header_content = header_file.read_text()
            
            # Should include Thrust headers since the kernel uses Thrust
            assert "#include <thrust/device_vector.h>" in header_content
            assert "#include <thrust/transform.h>" in header_content
            assert "#include <thrust/functional.h>" in header_content
            assert "This node uses Thrust algorithms" in header_content
            
            # Should reference the global kernel
            assert "Referenced global CUDA kernels:" in header_content
            assert "- test_kernel" in header_content
            
            # Check that the CUDA kernel file includes Thrust
            cu_file = next(f for f in cuda_files if f.name.endswith('.cu'))
            cu_content = cu_file.read_text()
            
            assert "#include <thrust/device_vector.h>" in cu_content
            assert "Thrust algorithms are used in this kernel" in cu_content

    def test_use_kernel_nonexistent_kernel(self):
        """Test that referencing a non-existent kernel doesn't break parsing."""
        robodsl_code = """
        node test_node_2 {
            use_kernel: "nonexistent_kernel"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        # This should parse successfully even though the kernel doesn't exist
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "test_node_2"
        assert len(node.content.used_kernels) == 1
        assert node.content.used_kernels[0] == "nonexistent_kernel"

    def test_thrust_usage_detection(self):
        """Test that Thrust usage is correctly detected and handled."""
        robodsl_code = """
        cuda_kernels {
            kernel no_thrust_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: false
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void no_thrust_kernel(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
            
            kernel thrust_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: true
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void thrust_kernel(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
        }

        node test_node_3 {
            use_kernel: "no_thrust_kernel"
            use_kernel: "thrust_kernel"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        # Check kernels
        assert len(ast.cuda_kernels.kernels) == 2
        
        no_thrust_kernel = next(k for k in ast.cuda_kernels.kernels if k.name == "no_thrust_kernel")
        thrust_kernel = next(k for k in ast.cuda_kernels.kernels if k.name == "thrust_kernel")
        
        assert no_thrust_kernel.content.use_thrust is False
        assert thrust_kernel.content.use_thrust is True
        
        # Generate code and verify Thrust includes are present
        with tempfile.TemporaryDirectory() as temp_dir:
            generator = MainGenerator(output_dir=temp_dir, debug=True)
            generated_files = generator.generate(ast)
            
            # Find the generated header file
            header_file = next(f for f in generated_files if f.name.endswith('.hpp'))
            header_content = header_file.read_text()
            
            # Should include Thrust headers since at least one kernel uses Thrust
            assert "#include <thrust/device_vector.h>" in header_content
            assert "This node uses Thrust algorithms" in header_content

    def test_use_kernel_semantic_validation(self):
        """Test semantic validation of use_kernel references."""
        robodsl_code = """
        cuda_kernels {
            kernel kernel1 {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: false
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void kernel1(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
        }

        node test_node_4 {
            use_kernel: "kernel1"
            use_kernel: "kernel1"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        # This should parse successfully (duplicates are allowed)
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        node = ast.nodes[0]
        assert len(node.content.used_kernels) == 2
        assert node.content.used_kernels[0] == "kernel1"
        assert node.content.used_kernels[1] == "kernel1"

    def test_use_kernel_in_pipeline_stage(self):
        """Test that use_kernel works in pipeline stages as well."""
        robodsl_code = """
        cuda_kernels {
            kernel pipeline_kernel {
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                use_thrust: true
                input: float* data, int size
                output: float* result
                code: {
                    __global__ void pipeline_kernel(float* data, float* result, int size) {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                        if (idx < size) {
                            result[idx] = data[idx] * 2.0f;
                        }
                    }
                }
            }
        }

        pipeline test_pipeline {
            stage processing_stage {
                input: "/input"
                output: "/output"
                method: "process"
                cuda_kernel: "pipeline_kernel"
                topic: /pipeline/status
            }
        }

        node test_node_5 {
            use_kernel: "pipeline_kernel"
            parameter int input_size = 1024
            publisher /test/output: "std_msgs/Float32MultiArray"
        }
        """
        
        ast = parse_robodsl(robodsl_code)
        assert ast is not None
        
        # Check global kernel
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert kernel.name == "pipeline_kernel"
        assert kernel.content.use_thrust is True
        
        # Check pipeline
        assert len(ast.pipelines) == 1
        pipeline = ast.pipelines[0]
        assert len(pipeline.content.stages) == 1
        stage = pipeline.content.stages[0]
        assert len(stage.content.cuda_kernels) == 1
        assert stage.content.cuda_kernels[0].kernel_name == "pipeline_kernel"
        
        # Check node
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.used_kernels) == 1
        assert node.content.used_kernels[0] == "pipeline_kernel"


if __name__ == "__main__":
    pytest.main([__file__]) 