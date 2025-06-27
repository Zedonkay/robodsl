"""CUDA Kernel Generator for RoboDSL.

This generator creates CUDA kernel files (.cu, .cuh) for standalone kernels
defined outside of nodes.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST, KernelNode, KernelParameterDirection


class CudaKernelGenerator(BaseGenerator):
    """Generates CUDA kernel files for standalone kernels."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate CUDA kernel files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'src').mkdir(exist_ok=True)
        
        # Generate files for standalone kernels
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                kernel_files = self._generate_kernel_files(kernel)
                generated_files.extend(kernel_files)
        
        return generated_files
    
    def _generate_kernel_files(self, kernel: KernelNode) -> List[Path]:
        """Generate all files for a single kernel."""
        generated_files = []
        
        # Generate CUDA header (.cuh)
        cuh_path = self._generate_kernel_header(kernel)
        generated_files.append(cuh_path)
        
        # Generate CUDA implementation (.cu)
        cu_path = self._generate_kernel_implementation(kernel)
        generated_files.append(cu_path)
        
        # Generate C++ wrapper header (.hpp)
        hpp_path = self._generate_kernel_wrapper_header(kernel)
        generated_files.append(hpp_path)
        
        return generated_files
    
    def _generate_kernel_header(self, kernel: KernelNode) -> Path:
        """Generate CUDA kernel header (.cuh) file."""
        context = self._prepare_kernel_context(kernel)
        
        try:
            content = self.render_template('kernel.cuh.jinja2', context)
            cuh_path = self.get_output_path('include', f'{kernel.name}_kernel.cuh')
            return self.write_file(cuh_path, content)
        except Exception as e:
            print(f"Template error for kernel {kernel.name}: {e}")
            # Fallback to simple header
            content = f"__global__ void {kernel.name}_kernel();"
            cuh_path = self.get_output_path('include', f'{kernel.name}_kernel.cuh')
            return self.write_file(cuh_path, content)
    
    def _generate_kernel_implementation(self, kernel: KernelNode) -> Path:
        """Generate CUDA kernel implementation (.cu) file."""
        context = self._prepare_kernel_context(kernel)
        
        try:
            content = self.render_template('kernel.cu.jinja2', context)
            cu_path = self.get_output_path('src', f'{kernel.name}_kernel.cu')
            return self.write_file(cu_path, content)
        except Exception as e:
            print(f"Template error for kernel {kernel.name}: {e}")
            # Fallback to simple implementation
            content = f"__global__ void {kernel.name}_kernel() {{\n  // Implementation\n}}"
            cu_path = self.get_output_path('src', f'{kernel.name}_kernel.cu')
            return self.write_file(cu_path, content)
    
    def _generate_kernel_wrapper_header(self, kernel: KernelNode) -> Path:
        """Generate C++ wrapper header (.hpp) file."""
        context = self._prepare_kernel_context(kernel)
        
        try:
            content = self.render_template('kernel.cuh.jinja2', context)  # Use same template as .cuh
            hpp_path = self.get_output_path('include', f'{kernel.name}_wrapper.hpp')
            return self.write_file(hpp_path, content)
        except Exception as e:
            print(f"Template error for kernel {kernel.name}: {e}")
            # Fallback to simple wrapper
            content = f"class {kernel.name}_wrapper {{\npublic:\n  void call();\n}};"
            hpp_path = self.get_output_path('include', f'{kernel.name}_wrapper.hpp')
            return self.write_file(hpp_path, content)
    
    def _prepare_kernel_context(self, kernel: KernelNode) -> Dict[str, Any]:
        """Prepare context for kernel template rendering."""
        # Extract parameters from kernel content
        kernel_parameters = []
        input_params = []
        output_params = []
        
        if kernel.content.parameters:
            for param in kernel.content.parameters:
                param_info = {
                    'name': param.param_name or f"param_{len(kernel_parameters)}",
                    'type': param.param_type,
                    'direction': param.direction,
                    'size_expr': param.size_expr,
                    'device_name': f"d_{param.param_name or f'param_{len(kernel_parameters)}'}_" if param.param_name else f"d_param_{len(kernel_parameters)}_"
                }
                kernel_parameters.append(param_info)
                
                if param.direction == KernelParameterDirection.IN:
                    input_params.append(param_info)
                elif param.direction == KernelParameterDirection.OUT:
                    output_params.append(param_info)
        
        # Determine input and output types
        input_type = input_params[0]['type'] if input_params else "float"
        output_type = output_params[0]['type'] if output_params else "float"
        param_type = "KernelParameters"  # Default parameter struct name
        
        # Prepare member variables for the wrapper class
        members = []
        for param in kernel_parameters:
            members.append({
                'name': param['device_name'],
                'type': f"{param['type']}*",
                'original_name': param['name']
            })
        
        return {
            'kernel_name': kernel.name,
            'namespace': 'robodsl',
            'include_guard': f"{kernel.name.upper()}_KERNEL_HPP",
            'include_path': f"{kernel.name}_kernel.cuh",
            'kernel_parameters': kernel_parameters,
            'input_params': input_params,
            'output_params': output_params,
            'input_type': input_type,
            'output_type': output_type,
            'param_type': param_type,
            'members': members,
            'block_size': kernel.content.block_size[0] if kernel.content.block_size else 256,
            'grid_size': kernel.content.grid_size,
            'shared_memory': kernel.content.shared_memory or 0,
            'use_thrust': kernel.content.use_thrust,
            'cuda_enabled': True
        } 