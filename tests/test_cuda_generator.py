import os
import sys
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.core.ast import RoboDSLAST, KernelNode, KernelContentNode, KernelParamNode, KernelParameterDirection
from conftest import skip_if_no_ros2, skip_if_no_cuda
from robodsl.generators import CudaKernelGenerator


class TestCudaGenerator:
    def test_generate_cuda_kernel_files(self, test_output_dir):
        skip_if_no_cuda()
        """Test CUDA kernel file generation."""
        generator = CudaKernelGenerator(str(test_output_dir))
        
        # Create a sample CUDA kernel AST
        kernel_content = KernelContentNode(
            block_size=(256, 1, 1),
            grid_size=(1, 1, 1),
            shared_memory=0,
            use_thrust=False,
            parameters=[
                KernelParamNode(
                    direction="in",
                    param_type="Image",
                    param_name="input",
                    size_expr=["width", "height"]
                ),
                KernelParamNode(
                    direction="out", 
                    param_type="Image",
                    param_name="output",
                    size_expr=["width", "height"]
                )
            ]
        )
        
        kernel = KernelNode(
            name="test_kernel",
            content=kernel_content
        )
        
        # Create a complete AST
        ast = RoboDSLAST()
        ast.cuda_kernels = MagicMock()
        ast.cuda_kernels.kernels = [kernel]
        
        generated_files = generator.generate(ast)
        
        # Check that files were generated
        assert len(generated_files) > 0
        
        # Check that expected files exist
        cuh_file = test_output_dir / 'include' / 'test_kernel_kernel.cuh'
        cu_file = test_output_dir / 'src' / 'test_kernel_kernel.cu'
        hpp_file = test_output_dir / 'include' / 'test_kernel_wrapper.hpp'
        
        assert cuh_file.exists()
        assert cu_file.exists()
        assert hpp_file.exists()
        
        # Check file contents
        cuh_content = cuh_file.read_text()
        assert "test_kernel" in cuh_content
        assert "Image" in cuh_content
        
        cu_content = cu_file.read_text()
        assert "__global__" in cu_content
        assert "test_kernel" in cu_content
        
        hpp_content = hpp_file.read_text()
        assert "test_kernel" in hpp_content

    def test_generate_with_multiple_kernels(self, test_output_dir):
        skip_if_no_ros2()
        """Test generation with multiple kernels."""
        generator = CudaKernelGenerator(str(test_output_dir))
        
        # Create first kernel
        kernel1_content = KernelContentNode(
            block_size=(256, 1, 1),
            grid_size=(1, 1, 1),
            shared_memory=0,
            use_thrust=False,
            parameters=[
                KernelParamNode(
                    direction="in",
                    param_type="Image",
                    param_name="input",
                    size_expr=["width", "height"]
                ),
                KernelParamNode(
                    direction="out", 
                    param_type="Image",
                    param_name="output",
                    size_expr=["width", "height"]
                )
            ]
        )
        
        kernel1 = KernelNode(
            name="test_kernel",
            content=kernel1_content
        )
        
        # Create second kernel
        kernel2_content = KernelContentNode(
            block_size=(32, 32, 1),
            grid_size=(1, 1, 1),
            shared_memory=1024,
            use_thrust=True,
            parameters=[
                KernelParamNode(
                    direction="in",
                    param_type="float",
                    param_name="data",
                    size_expr=["size"]
                )
            ]
        )
        
        kernel2 = KernelNode(
            name="filter_kernel",
            content=kernel2_content
        )
        
        # Create a complete AST
        ast = RoboDSLAST()
        ast.cuda_kernels = MagicMock()
        ast.cuda_kernels.kernels = [kernel1, kernel2]
        
        # Generate files
        generated_files = generator.generate(ast)
        
        # Check that files for both kernels were generated
        test_kernel_files = [f for f in generated_files if 'test_kernel' in str(f)]
        filter_kernel_files = [f for f in generated_files if 'filter_kernel' in str(f)]
        
        assert len(test_kernel_files) > 0
        assert len(filter_kernel_files) > 0

    def test_kernel_parameter_direction_enum(self):
        skip_if_no_ros2()
        """Test that kernel parameter directions are correctly handled."""
        # Test input parameter
        input_param = KernelParamNode(
            direction="in",
            param_type="Image",
            param_name="input",
            size_expr=["width", "height"]
        )
        assert input_param.direction == "in"
        
        # Test output parameter
        output_param = KernelParamNode(
            direction="out",
            param_type="Image", 
            param_name="output",
            size_expr=["width", "height"]
        )
        assert output_param.direction == "out"

    def test_kernel_content_validation(self):
        skip_if_no_ros2()
        """Test that kernel content is properly validated."""
        # Test with valid content
        valid_content = KernelContentNode(
            block_size=(256, 1, 1),
            grid_size=(1, 1, 1),
            shared_memory=0,
            use_thrust=False,
            parameters=[]
        )
        assert valid_content is not None
        
        # Test block size validation
        assert valid_content.block_size == (256, 1, 1)
        assert valid_content.grid_size == (1, 1, 1)

    def test_generator_without_kernels(self, test_output_dir):
        skip_if_no_ros2()
        """Test generator behavior when no kernels are present."""
        generator = CudaKernelGenerator(str(test_output_dir))
        
        empty_ast = RoboDSLAST()
        empty_ast.cuda_kernels = None
        
        generated_files = generator.generate(empty_ast)
        assert len(generated_files) == 0

    def test_new_kernel_syntax(self, test_output_dir):
        skip_if_no_ros2()
        """Test the new input: and output: syntax for CUDA kernels."""
        generator = CudaKernelGenerator(str(test_output_dir))
        
        # Create a sample CUDA kernel AST with new syntax
        kernel_content = KernelContentNode(
            block_size=(256, 1, 1),
            grid_size=(1, 1, 1),
            shared_memory=0,
            use_thrust=False,
            parameters=[
                KernelParamNode(
                    direction=KernelParameterDirection.IN,
                    param_type="float",
                    param_name="input_data",
                    size_expr=["size"]
                ),
                KernelParamNode(
                    direction=KernelParameterDirection.OUT, 
                    param_type="float",
                    param_name="output_data",
                    size_expr=["size"]
                )
            ]
        )
        
        kernel = KernelNode(
            name="test_new_syntax",
            content=kernel_content
        )
        
        # Create a complete AST
        ast = RoboDSLAST()
        ast.cuda_kernels = MagicMock()
        ast.cuda_kernels.kernels = [kernel]
        
        # Generate files
        generated_files = generator.generate(ast)
        
        # Check that files were generated
        assert len(generated_files) > 0
        
        # Check that expected files exist
        cuh_file = test_output_dir / 'include' / 'test_new_syntax_kernel.cuh'
        cu_file = test_output_dir / 'src' / 'test_new_syntax_kernel.cu'
        hpp_file = test_output_dir / 'include' / 'test_new_syntax_wrapper.hpp'
        
        assert cuh_file.exists()
        assert cu_file.exists()
        assert hpp_file.exists()
        
        # Check file contents for new syntax
        cuh_content = cuh_file.read_text()
        assert "test_new_syntax" in cuh_content
        assert "input_data" in cuh_content
        assert "output_data" in cuh_content
        
        cu_content = cu_file.read_text()
        assert "__global__" in cu_content
        assert "test_new_syntax" in cu_content
        assert "input_data" in cu_content
        assert "output_data" in cu_content
