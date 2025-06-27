"""
Code generation for RoboDSL.

This module handles the generation of C++ and CUDA source files from the parsed DSL configuration.
"""

import jinja2
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from .parser import RoboDSLAST, NodeNode, KernelNode

class CodeGenerator:
    """Generates C++ and CUDA source code from RoboDSL configuration."""
    
    def __init__(self, output_dir: str = "."):
        """Initialize the code generator.
        
        Args:
            output_dir: Base directory for generated files (default: current directory)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up Jinja2 environment with template directories
        template_dirs = [
            Path(__file__).parent / 'templates/cpp',
            Path(__file__).parent / 'templates/py',
            Path(__file__).parent / 'templates/cmake',
            Path(__file__).parent / 'templates/launch',
            Path(__file__).parent / 'templates',  # For backward compatibility
        ]
        template_loaders = [
            jinja2.FileSystemLoader(str(d)) for d in template_dirs if d.exists()
        ]
        if template_loaders:
            self.env = jinja2.Environment(
                loader=jinja2.ChoiceLoader(template_loaders),
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
            self.env.filters['indent'] = lambda text, n: '\n'.join(' ' * n + line if line.strip() else line \
                                                         for line in text.split('\n'))
        else:
            self.env = None
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate all source files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for all generated files
        """
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include' / 'robodsl').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'src').mkdir(exist_ok=True)
        (self.output_dir / 'launch').mkdir(exist_ok=True)
        
        # Generate code for each node
        for node in ast.nodes:
            # Generate node header and source files
            header_path = self._generate_node_header(node)
            source_path = self._generate_node_source(node)
            generated_files.extend([header_path, source_path])
            
            # Generate launch file if this is a ROS2 node
            if node.content.publishers or node.content.subscribers or node.content.services or node.content.parameters:
                launch_path = self._generate_launch_file(node)
                generated_files.append(launch_path)
        
        # Generate CUDA kernels
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                kernel_files = self._generate_cuda_kernel(kernel)
                generated_files.extend(kernel_files)
        
        # Generate package.xml if this is a ROS2 project
        package_path = self._generate_package_xml(ast)
        if package_path:
            generated_files.append(package_path)
        
        # Generate CMakeLists.txt
        cmake_path = self._generate_cmakelists(ast)
        generated_files.append(cmake_path)
        
        return generated_files
    
    def generate_cuda_kernel_header(self, kernel: KernelNode) -> str:
        """Generate CUDA kernel header content."""
        # This is a placeholder - implement actual header generation
        return f"__global__ void {kernel.name}_kernel();"
    
    def generate_cuda_kernel_implementation(self, kernel: KernelNode) -> str:
        """Generate CUDA kernel implementation content."""
        # This is a placeholder - implement actual implementation generation
        return f"__global__ void {kernel.name}_kernel() {{\n  // Implementation\n}}"
    
    def generate_cuda_kernel_cpp_header(self, kernel: KernelNode) -> str:
        """Generate C++ header for CUDA kernel wrapper."""
        # This is a placeholder - implement actual C++ header generation
        return f"class {kernel.name}_wrapper {{\npublic:\n  void call();\n}};"
    
    def generate_cuda_source_file(self, ast: RoboDSLAST) -> str:
        """Generate complete CUDA source file."""
        # This is a placeholder - implement actual CUDA source generation
        content = ["#include <cuda_runtime.h>"]
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                content.append(self.generate_cuda_kernel_implementation(kernel))
        return "\n".join(content)
    
    def _generate_node_header(self, node: NodeNode) -> Path:
        """Generate a C++ header file for a ROS2 node."""
        # Placeholder implementation
        header_path = self.output_dir / 'include' / f'{node.name}_node.hpp'
        with open(header_path, 'w') as f:
            f.write(f"// Generated header for {node.name}\n")
        return header_path
    
    def _generate_node_source(self, node: NodeNode) -> Path:
        """Generate a C++ source file for a ROS2 node."""
        # Placeholder implementation
        source_path = self.output_dir / 'src' / f'{node.name}_node.cpp'
        with open(source_path, 'w') as f:
            f.write(f"// Generated source for {node.name}\n")
        return source_path
    
    def _generate_launch_file(self, node: NodeNode) -> Optional[Path]:
        """Generate a ROS2 launch file for a node."""
        # Placeholder implementation
        launch_path = self.output_dir / 'launch' / f'{node.name}.launch.py'
        with open(launch_path, 'w') as f:
            f.write(f"# Generated launch file for {node.name}\n")
        return launch_path
    
    def _generate_cuda_kernel(self, kernel: KernelNode) -> List[Path]:
        """Generate CUDA kernel files."""
        # Placeholder implementation
        kernel_path = self.output_dir / 'src' / f'{kernel.name}_kernel.cu'
        with open(kernel_path, 'w') as f:
            f.write(f"// Generated CUDA kernel for {kernel.name}\n")
        return [kernel_path]
    
    def _generate_package_xml(self, ast: RoboDSLAST) -> Optional[Path]:
        """Generate package.xml for ROS2 project."""
        # Placeholder implementation
        package_path = self.output_dir / 'package.xml'
        with open(package_path, 'w') as f:
            f.write("<?xml version=\"1.0\"?>\n<package format=\"3\">\n</package>\n")
        return package_path
    
    def _generate_cmakelists(self, ast: RoboDSLAST) -> Path:
        """Generate CMakeLists.txt."""
        # Placeholder implementation
        cmake_path = self.output_dir / 'CMakeLists.txt'
        with open(cmake_path, 'w') as f:
            f.write("cmake_minimum_required(VERSION 3.8)\nproject(robodsl_generated)\n")
        return cmake_path

