"""C++ Node Generator for RoboDSL.

This generator creates C++ header (.hpp) and source (.cpp) files for ROS2 nodes,
including virtual methods for CUDA kernels defined within the node.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..ast import RoboDSLAST, NodeNode, KernelNode


class CppNodeGenerator(BaseGenerator):
    """Generates C++ node files with CUDA kernel virtual methods."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate C++ node files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include' / 'robodsl').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'src').mkdir(exist_ok=True)
        
        # Generate files for each node
        for node in ast.nodes:
            # Generate node header and source files
            header_path = self._generate_node_header(node)
            source_path = self._generate_node_source(node)
            generated_files.extend([header_path, source_path])
        
        return generated_files
    
    def _generate_node_header(self, node: NodeNode) -> Path:
        """Generate a C++ header file for a ROS2 node."""
        context = self._prepare_node_context(node)
        
        try:
            content = self.render_template('node.hpp.jinja2', context)
            header_path = self.get_output_path('include', f'{node.name}_node.hpp')
            return self.write_file(header_path, content)
        except Exception as e:
            print(f"Template error for node {node.name}: {e}")
            # Fallback to simple header
            content = f"// Generated header for {node.name}\n"
            header_path = self.get_output_path('include', f'{node.name}_node.hpp')
            return self.write_file(header_path, content)
    
    def _generate_node_source(self, node: NodeNode) -> Path:
        """Generate a C++ source file for a ROS2 node."""
        context = self._prepare_node_context(node)
        
        try:
            content = self.render_template('node.cpp.jinja2', context)
            source_path = self.get_output_path('src', f'{node.name}_node.cpp')
            return self.write_file(source_path, content)
        except Exception as e:
            print(f"Template error for node {node.name}: {e}")
            # Fallback to simple source
            content = f"// Generated source for {node.name}\n"
            source_path = self.get_output_path('src', f'{node.name}_node.cpp')
            return self.write_file(source_path, content)
    
    def _prepare_node_context(self, node: NodeNode) -> Dict[str, Any]:
        """Prepare context for node template rendering."""
        # Prepare includes based on node content
        includes = []
        ros2_includes = []
        cuda_includes = []
        
        # Add message includes for publishers/subscribers
        for pub in node.content.publishers:
            includes.append(f"#include <{pub.msg_type}.hpp>")
        for sub in node.content.subscribers:
            includes.append(f"#include <{sub.msg_type}.hpp>")
        for srv in node.content.services:
            includes.append(f"#include <{srv.srv_type}.hpp>")
        for action in node.content.actions:
            includes.append(f"#include <{action.action_type}.hpp>")
        
        # Add CUDA includes if node has kernels
        if node.content.cuda_kernels:
            cuda_includes.extend([
                "#include <cuda_runtime.h>",
                "#include <vector>",
                "#include <memory>"
            ])
        
        # Prepare CUDA kernel information
        cuda_kernels = []
        for kernel in node.content.cuda_kernels:
            kernel_info = self._prepare_kernel_context(kernel)
            cuda_kernels.append(kernel_info)
        
        # Determine if this is a lifecycle node
        is_lifecycle = node.content.lifecycle is not None
        
        # Prepare C++ methods
        cpp_methods = []
        if hasattr(node.content, 'cpp_methods'):
            for method in node.content.cpp_methods:
                method_info = {
                    'name': method.name,
                    'inputs': [{
                        'type': f"const {param.param_type}" if getattr(param, 'is_const', False) else param.param_type, 
                        'name': param.param_name, 
                        'size_expr': param.size_expr
                    } for param in method.inputs],
                    'outputs': [{
                        'type': f"const {param.param_type}" if getattr(param, 'is_const', False) else param.param_type, 
                        'name': param.param_name, 
                        'size_expr': param.size_expr
                    } for param in method.outputs],
                    'code': method.code
                }
                cpp_methods.append(method_info)
        
        return {
            'class_name': f"{node.name.capitalize()}Node",
            'base_class': 'rclcpp_lifecycle::LifecycleNode' if is_lifecycle else 'rclcpp::Node',
            'namespace': 'robodsl',
            'include_guard': f"{node.name.upper()}_NODE_HPP",
            'includes': list(set(includes)),  # Remove duplicates
            'ros2_includes': ros2_includes,
            'cuda_includes': cuda_includes,
            'is_lifecycle': is_lifecycle,
            'publishers': [{'name': pub.topic.split('/')[-1], 'msg_type': pub.msg_type} for pub in node.content.publishers],
            'subscribers': [{'name': sub.topic.split('/')[-1], 'msg_type': sub.msg_type, 'callback_name': f"on_{sub.topic.split('/')[-1]}"} for sub in node.content.subscribers],
            'services': [{'name': srv.service.split('/')[-1], 'srv_type': srv.srv_type, 'callback_name': f"on_{srv.service.split('/')[-1]}"} for srv in node.content.services],
            'actions': [{'name': action.name, 'action_type': action.action_type} for action in node.content.actions],
            'timers': [{'name': timer.name, 'callback_name': f"on_{timer.name}"} for timer in node.content.timers],
            'parameters': [{'name': param.name, 'type': 'auto'} for param in node.content.parameters],
            'cuda_kernels': cuda_kernels,
            'cuda_default_input_type': 'float',
            'cuda_default_output_type': 'float',
            'cuda_default_param_type': 'CudaParams',
            'cpp_methods': cpp_methods
        }
    
    def _prepare_kernel_context(self, kernel: KernelNode) -> Dict[str, Any]:
        """Prepare context for kernel template rendering."""
        # Extract parameters from kernel content
        kernel_parameters = []
        input_params = []
        output_params = []
        param_signature = []
        
        if kernel.content.parameters:
            for param in kernel.content.parameters:
                param_type = param.param_type
                if getattr(param, 'is_const', False):
                    param_type = f"const {param_type}"
                param_info = {
                    'name': param.param_name or f"param_{len(kernel_parameters)}",
                    'type': param_type,
                    'direction': param.direction,
                    'size_expr': param.size_expr
                }
                kernel_parameters.append(param_info)
                param_signature.append(f"{param_type} {param.param_name}")
                if param.direction == "in":
                    input_params.append(param_info)
                elif param.direction == "out":
                    output_params.append(param_info)
        
        # Determine input and output types
        input_type = input_params[0]['type'] if input_params else "float"
        output_type = output_params[0]['type'] if output_params else "float"
        param_type = "KernelParameters"  # Default parameter struct name
        
        # Prepare member variables for the wrapper class
        members = []
        for param in kernel_parameters:
            members.append({
                'name': param['name'],
                'type': param['type']
            })
        
        return {
            'kernel_name': kernel.name,
            'namespace': 'robodsl',
            'include_guard': f"{kernel.name.upper()}_KERNEL_HPP",
            'include_path': f"{kernel.name}_kernel.cuh",
            'kernel_parameters': kernel_parameters,
            'parameters': kernel_parameters,  # For template
            'signature': ', '.join(param_signature),
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