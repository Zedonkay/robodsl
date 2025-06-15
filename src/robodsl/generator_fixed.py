"""
Code generation for RoboDSL.

This module handles the generation of C++ and CUDA source files from the parsed DSL configuration.
"""

import dataclasses
import os
import jinja2
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, fields, asdict

from .parser import RoboDSLConfig, NodeConfig, CudaKernelConfig

# Set up Jinja2 environment with the templates directory
template_dir = Path(__file__).parent / 'templates'
if template_dir.exists():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    env.filters['indent'] = lambda text, n: '\n'.join(' ' * n + line if line.strip() else line for line in text.split('\n'))


class CodeGenerator:
    """Generates C++ and CUDA source code from RoboDSL configuration."""
    
    def __init__(self, config: RoboDSLConfig, output_dir: str = "."):
        """Initialize the code generator with the parsed DSL configuration.
        
        Args:
            config: The parsed RoboDSL configuration
            output_dir: Base directory for generated files (default: current directory)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _generate_message_includes(self, node: NodeConfig) -> List[str]:
        """Generate message includes with ROS2 conditional compilation."""
        uses_ros2 = bool(node.publishers or node.subscribers or node.services or node.parameters)
        includes = []
        
        if uses_ros2:
            # In ROS2 mode, include the actual message headers
            msg_includes = set()
            for pub in node.publishers:
                msg_includes.add(f'#include "{pub["msg_type"].replace(".", "/")}.hpp"')
            for sub in node.subscribers:
                msg_includes.add(f'#include "{sub["msg_type"].replace(".", "/")}.hpp"')
            for srv in node.services:
                msg_includes.add(f'#include "{srv["srv_type"].replace(".", "/")}.hpp"')
            
            if msg_includes:
                includes.append('// ROS2 message includes')
                includes.extend(sorted(msg_includes))
        else:
            # In non-ROS2 mode, provide stub message types
            includes.append('// Message type stubs for non-ROS2 builds')
            includes.append('#include <memory>')
            
            # Add stubs for all message types
            msg_stubs = set()
            for pub in node.publishers:
                ns, name = pub["msg_type"].split('.')
                msg_stubs.add(f'namespace {ns} {{ namespace msg {{ struct {name} {{ using SharedPtr = std::shared_ptr<{name}>; using ConstSharedPtr = std::shared_ptr<const {name}>; }}; }}}}')
            for sub in node.subscribers:
                ns, name = sub["msg_type"].split('.')
                msg_stubs.add(f'namespace {ns} {{ namespace msg {{ struct {name} {{ using SharedPtr = std::shared_ptr<{name}>; using ConstSharedPtr = std::shared_ptr<const {name}>; }}; }}}}')
            for srv in node.services:
                ns, name = srv["srv_type"].split('.')
                msg_stubs.add(f'namespace {ns} {{ namespace srv {{ struct {name} {{ struct Request {{ using SharedPtr = std::shared_ptr<Request>; }}; struct Response {{ using SharedPtr = std::shared_ptr<Response>; }}; using Request = Request; using Response = Response; using SharedPtr = std::shared_ptr<{name}>; }}; }}}}')
            
            includes.extend(sorted(msg_stubs))
        
        return includes
    
    def _generate_node_header(self, node: NodeConfig) -> Path:
        """Generate a C++ header file for a ROS2 node.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated header file
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Generate include guard
        include_guard = f"ROBODSL_{node.name.upper()}_NODE_HPP_"
        
        # Check if this node uses ROS2 features
        uses_ros2 = bool(node.publishers or node.subscribers or node.services or node.parameters)
        
        # Generate base includes
        includes = [
            '#include <string>',
            '#include <vector>',
            '#include <map>',
            '',  # For better formatting
        ]
        
        # Conditionally include ROS2 headers
        if uses_ros2:
            includes.insert(0, '#if ENABLE_ROS2')
            includes.insert(1, '#include "rclcpp/rclcpp.hpp"')
            includes.insert(2, '#else')
            includes.insert(3, '// ROS2 stubs for non-ROS2 builds')
            includes.insert(4, 'namespace rclcpp { class Node {}; }')
            includes.insert(5, '#endif')
            includes.insert(6, '')  # Extra newline for formatting
        
        # Add message includes with conditional compilation
        includes.extend(self._generate_message_includes(node))
        includes.append('')  # Extra newline for formatting
        
        # Generate the class declaration parts
        ros_publishers = self._generate_publisher_declarations(node.publishers)
        ros_subscribers = self._generate_subscriber_declarations(node.subscribers)
        ros_services = self._generate_service_declarations(node.services)
        parameters = self._generate_parameter_declarations(node.parameters)
        cuda_kernels = self._generate_cuda_kernel_declarations(node.name, self.config.cuda_kernels)
        
        # Generate the class declaration with conditional ROS2 support
        class_declaration = []
        
        if uses_ros2:
            class_declaration.append(f'class {class_name} : public rclcpp::Node {{')
            class_declaration.append('public:')
            class_declaration.append(f'    {class_name}() : Node("{node.name}") {{}}')
        else:
            class_declaration.append(f'class {class_name} {{')
            class_declaration.append('public:')
            class_declaration.append(f'    {class_name}() = default;')
        
        class_declaration.extend([
            f'    virtual ~{class_name}() = default;',
            '    ', 
            '    // Common interface methods',
            '    void initialize();',
            '    void update();',
            '    void cleanup();',
            '',
            'private:',
            '    // ROS2 Publishers',
            f'{ros_publishers}',
            '    ', 
            '    // ROS2 Subscribers',
            f'{ros_subscribers}',
            '    ', 
            '    // ROS2 Services',
            f'{ros_services}',
            '    ', 
            '    // Parameters',
            f'{parameters}',
            '',
            '    // CUDA Kernels',
            f'{cuda_kernels}',
            '};'
        ])
        
        class_declaration = '\n'.join(class_declaration)
        
        # Write the header file
        header_content = [
            f'// Generated by RoboDSL - DO NOT EDIT',
            f'#ifndef {include_guard}',
            f'#define {include_guard}',
            ''
        ]
        
        # Add includes
        header_content.extend(includes)
        header_content.append('')
        
        # Add class declaration
        header_content.append(class_declaration)
        
        # Add footer
        header_content.extend([
            '',
            f'#endif // {include_guard}',
            ''
        ])
        
        header_content = '\n'.join(header_content)
        
        # Create output directory if it doesn't exist
        header_path = self.output_dir / 'include' / 'robodsl' / f"{node.name}_node.hpp"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(header_path, 'w') as f:
            f.write(header_content)
        
        return header_path
    
    # Add other methods from the original generator here...
    # (This is a simplified version focusing on the changes needed for message includes)
    
    def _generate_publisher_declarations(self, publishers: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 publishers."""
        if not publishers:
            return "    // No publishers defined\n"
            
        decls = []
        for pub in publishers:
            topic = pub['topic']
            msg_type = pub['msg_type'].replace('.', '::')
            var_name = f"{topic.lstrip('/').replace('/', '_')}_pub_"
            decls.append(f"    rclcpp::Publisher<{msg_type}>::{type} {var_name};")
            
        return '\n'.join(decls)
    
    def _generate_subscriber_declarations(self, subscribers: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 subscribers."""
        if not subscribers:
            return "    // No subscribers defined\n"
            
        decls = []
        for sub in subscribers:
            topic = sub['topic']
            msg_type = sub['msg_type'].replace('.', '::')
            callback_name = f"{topic.lstrip('/').replace('/', '_')}_callback"
            decls.append(f"    void {callback_name}(const {msg_type}::ConstSharedPtr msg) const;")
            
        return '\n'.join(decls)
    
    def _generate_service_declarations(self, services: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 services."""
        if not services:
            return "    // No services defined\n"
            
        decls = []
        for srv in services:
            service_name = srv['service']
            srv_type = srv['srv_type'].replace('.', '::')
            callback_name = f"{service_name.lstrip('/').replace('/', '_')}_callback"
            decls.append(f"    void {callback_name}(" \
                        f"const std::shared_ptr<{srv_type}::Request> request, " \
                        f"std::shared_ptr<{srv_type}::Response> response);")
            
        return '\n'.join(decls)
    
    def _generate_parameter_declarations(self, parameters: Dict[str, str]) -> str:
        """Generate C++ declarations for node parameters."""
        if not parameters:
            return "    // No parameters defined\n"
            
        decls = ["    // Parameters"]
        for name, type_info in parameters.items():
            cpp_type = self._map_parameter_type(type_info)
            decls.append(f"    {cpp_type} {name}_;")
            
        return '\n'.join(decls)
    
    def _generate_cuda_kernel_declarations(self, node_name: str, kernels: List[CudaKernelConfig]) -> str:
        """Generate C++ declarations for CUDA kernels used by this node."""
        if not kernels:
            return "    // No CUDA kernels defined\n"
            
        decls = ["    // CUDA Kernels"]
        for kernel in kernels:
            if kernel.node == node_name:
                decls.append(f"    // CUDA kernel: {kernel.name}")
                decls.append(f"    // Add kernel declarations here")
                
        return '\n'.join(decls)
    
    def _map_parameter_type(self, type_info: str) -> str:
        """Map parameter type from DSL to C++ type."""
        type_map = {
            'int': 'int',
            'float': 'float',
            'double': 'double',
            'bool': 'bool',
            'string': 'std::string',
            'int[]': 'std::vector<int>',
            'float[]': 'std::vector<float>',
            'double[]': 'std::vector<double>',
            'bool[]': 'std::vector<bool>',
            'string[]': 'std::vector<std::string>',
        }
        return type_map.get(type_info, 'auto')  # Default to auto if type not recognized
