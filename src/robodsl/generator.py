"""Main generator module for RoboDSL.

This module provides the main entry point for code generation using the modular
generator system.
"""

from pathlib import Path
from typing import List, Optional

from .ast import RoboDSLAST
from .generators import MainGenerator


def generate_code(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate all code files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for all generated files
    """
    generator = MainGenerator(output_dir, template_dirs)
    return generator.generate(ast)


def generate_cpp_nodes(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate only C++ node files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for generated C++ files
    """
    from .generators import CppNodeGenerator
    generator = CppNodeGenerator(output_dir, template_dirs)
    return generator.generate(ast)


def generate_cuda_kernels(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate only CUDA kernel files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for generated CUDA files
    """
    from .generators import CudaKernelGenerator
    generator = CudaKernelGenerator(output_dir, template_dirs)
    return generator.generate(ast)


def generate_python_nodes(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate only Python node files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for generated Python files
    """
    from .generators import PythonNodeGenerator
    generator = PythonNodeGenerator(output_dir, template_dirs)
    return generator.generate(ast)


def generate_cmake_files(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate only CMake files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for generated CMake files
    """
    from .generators import CMakeGenerator
    generator = CMakeGenerator(output_dir, template_dirs)
    return generator.generate(ast)


def generate_launch_files(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate only launch files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for generated launch files
    """
    from .generators import LaunchGenerator
    generator = LaunchGenerator(output_dir, template_dirs)
    return generator.generate(ast)


def generate_package_files(ast: RoboDSLAST, output_dir: str = ".", template_dirs: Optional[List[Path]] = None) -> List[Path]:
    """Generate only package files from a RoboDSL AST.
    
    Args:
        ast: The parsed RoboDSL AST
        output_dir: Directory to output generated files
        template_dirs: Additional template directories to search
        
    Returns:
        List of Path objects for generated package files
    """
    from .generators import PackageGenerator
    generator = PackageGenerator(output_dir, template_dirs)
    return generator.generate(ast)


# Backward compatibility - keep the old CodeGenerator class for existing code
class CodeGenerator:
    """Legacy code generator for backward compatibility."""
    
    def __init__(self, output_dir: str = "."):
        """Initialize the legacy code generator.
        
        Args:
            output_dir: Base directory for generated files
        """
        self.output_dir = output_dir
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate all code files from the AST using the new modular system.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for all generated files
        """
        return generate_code(ast, self.output_dir)

