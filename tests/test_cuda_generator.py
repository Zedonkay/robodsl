import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parser import CudaKernelConfig, RoboDSLConfig
from robodsl.generator import CodeGenerator

class TestCudaGenerator(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test output
        self.test_dir = Path("test_output")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a sample CUDA kernel configuration
        self.kernel = CudaKernelConfig(
            name="vector_add",
            block_size=(256, 1, 1),
            grid_size=(1, 1, 1),
            shared_mem_bytes=0,
            use_thrust=False,
            inputs=[{"name": "a", "type": "float"}, {"name": "b", "type": "float"}],
            outputs=[{"name": "c", "type": "float"}],
            code="int idx = blockIdx.x * blockDim.x + threadIdx.x;\nif (idx < N) {\n    c[idx] = a[idx] + b[idx];\n}",
            includes=["<vector>", "<cuda_runtime.h>"],
            defines=["N 1024"]
        )
        
        # Create a minimal config
        self.config = RoboDSLConfig(
            project_name="test_project",
            nodes=[],
            cuda_kernels=[self.kernel]
        )
        
        # Initialize the code generator
        self.generator = CodeGenerator(self.config, output_dir=str(self.test_dir))

    def test_generate_cuda_source(self):
        """Test CUDA source file generation."""
        # Generate the source file
        self.generator._generate_cuda_kernel(self.kernel)
        
        # Check if the source file was created
        source_file = self.test_dir / "cuda" / "vector_add_kernel.cu"
        self.assertTrue(source_file.exists(), "CUDA source file was not created")
        
        # Check the content of the source file
        with open(source_file, 'r') as f:
            content = f.read()
            
        # Check for important parts in the generated code
        self.assertIn("__global__ void vector_add_kernel", content)
        self.assertIn("const float* a", content)
        self.assertIn("const float* b", content)
        self.assertIn("float* c", content)
        self.assertIn("cudaMalloc", content)
        self.assertIn("cudaMemcpy", content)
        self.assertIn("vector_add_kernel<<<", content)

    def tearDown(self):
        # Clean up test files
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
