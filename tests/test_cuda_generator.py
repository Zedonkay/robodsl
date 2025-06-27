import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.ast import RoboDSLAST, KernelNode, KernelContentNode, KernelParamNode, KernelParameterDirection
from robodsl.generators import CudaKernelGenerator

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
        self.generator = CudaKernelGenerator(str(self.test_dir))

    def test_generate_cuda_kernel_files(self):
        """Test CUDA kernel file generation."""
        generated_files = self.generator.generate(self.ast)
        
        # Check that files were generated
        self.assertGreater(len(generated_files), 0)
        
        # Check that expected files exist
        cuh_file = self.test_dir / 'include' / 'test_kernel_kernel.cuh'
        cu_file = self.test_dir / 'src' / 'test_kernel_kernel.cu'
        hpp_file = self.test_dir / 'include' / 'test_kernel_wrapper.hpp'
        
        self.assertTrue(cuh_file.exists())
        self.assertTrue(cu_file.exists())
        self.assertTrue(hpp_file.exists())
        
        # Check file contents
        with open(cuh_file, 'r') as f:
            cuh_content = f.read()
            self.assertIn("test_kernel", cuh_content)
            self.assertIn("Image", cuh_content)
        
        with open(cu_file, 'r') as f:
            cu_content = f.read()
            self.assertIn("__global__", cu_content)
            self.assertIn("test_kernel", cu_content)
        
        with open(hpp_file, 'r') as f:
            hpp_content = f.read()
            self.assertIn("test_kernel", hpp_content)

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
        
        # Generate files
        generated_files = self.generator.generate(self.ast)
        
        # Check that files for both kernels were generated
        test_kernel_files = [f for f in generated_files if 'test_kernel' in str(f)]
        filter_kernel_files = [f for f in generated_files if 'filter_kernel' in str(f)]
        
        self.assertGreater(len(test_kernel_files), 0)
        self.assertGreater(len(filter_kernel_files), 0)

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

    def test_generator_without_kernels(self):
        """Test generator behavior when no kernels are present."""
        empty_ast = RoboDSLAST()
        empty_ast.cuda_kernels = None
        
        generated_files = self.generator.generate(empty_ast)
        self.assertEqual(len(generated_files), 0)

    def tearDown(self):
        # Clean up test output directory
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)

if __name__ == '__main__':
    unittest.main()
