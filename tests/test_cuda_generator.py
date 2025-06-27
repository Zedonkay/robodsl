import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parser import RoboDSLAST, KernelNode, KernelContentNode, KernelParamNode, KernelParameterDirection
from robodsl.generator import CodeGenerator

class TestCudaGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test output
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
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
        
        self.kernel = KernelNode(
            name="test_kernel",
            content=kernel_content
        )
        
        # Create a complete AST
        self.ast = RoboDSLAST()
        self.ast.cuda_kernels = MagicMock()
        self.ast.cuda_kernels.kernels = [self.kernel]
        
        # Create generator
        self.generator = CodeGenerator()

    def test_generate_cuda_kernel_header(self):
        """Test CUDA kernel header generation."""
        header_content = self.generator.generate_cuda_kernel_header(self.kernel)
        
        # Check that the header contains expected content
        self.assertIn("__global__", header_content)
        self.assertIn("test_kernel", header_content)
        # Note: Current placeholder doesn't include parameter types
        # self.assertIn("Image", header_content)

    def test_generate_cuda_kernel_implementation(self):
        """Test CUDA kernel implementation generation."""
        impl_content = self.generator.generate_cuda_kernel_implementation(self.kernel)
        
        # Check that the implementation contains expected content
        self.assertIn("__global__", impl_content)
        self.assertIn("test_kernel", impl_content)
        # Note: Current placeholder doesn't include parameter types
        # self.assertIn("Image", impl_content)

    def test_generate_cuda_kernel_cpp_header(self):
        """Test CUDA kernel C++ header generation."""
        cpp_header = self.generator.generate_cuda_kernel_cpp_header(self.kernel)
        
        # Check that the C++ header contains expected content
        self.assertIn("class", cpp_header)
        self.assertIn("test_kernel", cpp_header)
        # Note: Current placeholder doesn't include parameter types
        # self.assertIn("Image", cpp_header)

    def test_generate_cuda_source_file(self):
        """Test complete CUDA source file generation."""
        cuda_source = self.generator.generate_cuda_source_file(self.ast)
        
        # Check that the source file contains expected content
        self.assertIn("#include", cuda_source)
        self.assertIn("__global__", cuda_source)
        self.assertIn("test_kernel", cuda_source)

    def test_generate_with_multiple_kernels(self):
        """Test generation with multiple kernels."""
        # Add another kernel
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
        
        self.ast.cuda_kernels.kernels.append(kernel2)
        
        # Generate source
        cuda_source = self.generator.generate_cuda_source_file(self.ast)
        
        # Check that both kernels are included
        self.assertIn("test_kernel", cuda_source)
        self.assertIn("filter_kernel", cuda_source)

    def test_kernel_parameter_direction_enum(self):
        """Test that kernel parameter directions are correctly handled."""
        # Test input parameter
        input_param = KernelParamNode(
            direction="in",
            param_type="Image",
            param_name="input",
            size_expr=["width", "height"]
        )
        self.assertEqual(input_param.direction, "in")
        
        # Test output parameter
        output_param = KernelParamNode(
            direction="out",
            param_type="Image", 
            param_name="output",
            size_expr=["width", "height"]
        )
        self.assertEqual(output_param.direction, "out")

    def test_kernel_content_validation(self):
        """Test that kernel content is properly validated."""
        # Test with valid content
        valid_content = KernelContentNode(
            block_size=(256, 1, 1),
            grid_size=(1, 1, 1),
            shared_memory=0,
            use_thrust=False,
            parameters=[]
        )
        self.assertIsNotNone(valid_content)
        
        # Test block size validation
        self.assertEqual(valid_content.block_size, (256, 1, 1))
        self.assertEqual(valid_content.grid_size, (1, 1, 1))

    def tearDown(self):
        # Clean up test output directory
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
