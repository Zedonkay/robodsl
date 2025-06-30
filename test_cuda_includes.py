#!/usr/bin/env python3
"""Test script to verify cuda_includes functionality."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from robodsl.parsers.lark_parser import RoboDSLParser
from robodsl.generators.cpp_node_generator import CppNodeGenerator
from robodsl.generators.cuda_kernel_generator import CudaKernelGenerator
from pathlib import Path
import tempfile

def test_cuda_includes():
    """Test that cuda_includes are properly parsed and included only in CUDA files."""
    
    # Test RoboDSL code with cuda_includes
    robodsl_code = """
    cuda_kernels {
        kernel test_kernel {
            input: float* data, int size
            output: float* result
            include <cuda_runtime.h>
            include <device_launch_parameters.h>
            include <thrust/sort.h>
            code: {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < size) {
                    result[idx] = data[idx] * 2.0f;
                }
            }
        }
    }
    
    node test_node {
        parameter int count = 0
        publisher /chatter: "std_msgs/msg/String"
    }
    """
    
    # Parse the code
    parser = RoboDSLParser()
    ast = parser.parse(robodsl_code)
    
    print("AST parsed successfully")
    print(f"Number of CUDA kernels: {len(ast.cuda_kernels.kernels)}")
    
    # Check that cuda_includes are parsed
    kernel = ast.cuda_kernels.kernels[0]
    print(f"Kernel name: {kernel.name}")
    print(f"CUDA includes: {kernel.content.cuda_includes}")
    
    # Verify the includes are correct
    expected_includes = [
        "cuda_runtime.h",
        "device_launch_parameters.h", 
        "thrust/sort.h"
    ]
    
    assert len(kernel.content.cuda_includes) == 3, f"Expected 3 includes, got {len(kernel.content.cuda_includes)}"
    for include in expected_includes:
        assert include in kernel.content.cuda_includes, f"Expected include '{include}' not found"
    
    print("✓ CUDA includes parsed correctly")
    
    # Create output directory
    output_dir = Path("test_output/cuda_includes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate code
    node_generator = CppNodeGenerator(output_dir=output_dir)
    cuda_generator = CudaKernelGenerator(output_dir=output_dir)
    
    node_files = node_generator.generate(ast)
    cuda_files = cuda_generator.generate(ast)
    
    all_files = node_files + cuda_files
    
    print(f"Generated {len(all_files)} files")
    print("Generated files:")
    for f in all_files:
        print(f"  {f}")
    
    # Check that CUDA includes are NOT in node files (.hpp, .cpp)
    for file_path in node_files:
        if file_path.suffix in ['.hpp', '.cpp']:
            content = file_path.read_text()
            print(f"\nChecking node file: {file_path.name}")
            for include in expected_includes:
                assert f"#include <{include}>" not in content, f"CUDA include '#include <{include}>' found in node file {file_path.name}"
            print(f"✓ No CUDA includes found in {file_path.name}")
    
    # Check that CUDA includes ARE in CUDA files (.cu, .cuh)
    for file_path in cuda_files:
        if file_path.suffix in ['.cu', '.cuh']:
            content = file_path.read_text()
            print(f"\nChecking CUDA file: {file_path.name}")
            print(f"Content preview:")
            print(content[:500] + "..." if len(content) > 500 else content)
            
            for include in expected_includes:
                assert f"#include <{include}>" in content, f"Expected '#include <{include}>' not found in CUDA file {file_path.name}"
            print(f"✓ All CUDA includes found in {file_path.name}")
    
    print("\n✓ All tests passed! CUDA includes only appear in .cu/.cuh files")

if __name__ == "__main__":
    test_cuda_includes()
    print("Test completed successfully!") 