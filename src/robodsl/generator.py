"""
Code generation for RoboDSL.

This module handles the generation of C++ and CUDA source files from the parsed DSL configuration.
"""

import jinja2
from pathlib import Path
from typing import Dict, List, Any, Optional, Set

from .parser import RoboDSLConfig, NodeConfig, CudaKernelConfig, QoSConfig

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
        
    def generate(self) -> List[Path]:
        """Generate all source files from the DSL configuration.
        
        Returns:
            List of Path objects for all generated files
        """
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include' / 'robodsl').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'src').mkdir(exist_ok=True)
        (self.output_dir / 'launch').mkdir(exist_ok=True)
        
        # Generate code for each node
        for node in self.config.nodes:
            # Generate node header and source files
            header_path = self._generate_node_header(node)
            source_path = self._generate_node_source(node)
            generated_files.extend([header_path, source_path])
            
            # Generate launch file if this is a ROS2 node
            if node.publishers or node.subscribers or node.services or node.parameters:
                launch_path = self._generate_launch_file(node)
                generated_files.append(launch_path)
        
        # Generate CUDA kernels
        for kernel in self.config.cuda_kernels:
            kernel_files = self._generate_cuda_kernel(kernel)
            generated_files.extend(kernel_files)
        
        # Generate package.xml if this is a ROS2 project
        package_path = self._generate_package_xml()
        if package_path:
            generated_files.append(package_path)
        
        # Generate CMakeLists.txt
        cmake_path = self._generate_cmakelists()
        generated_files.append(cmake_path)
        
        # Generate launch files for each node
        for node in self.config.nodes:
            if node.publishers or node.subscribers or node.services or node.parameters:
                launch_path = self._generate_launch_file(node)
                if launch_path:
                    generated_files.append(launch_path)
        
        return generated_files
    
    def _generate_message_includes(self, node: NodeConfig) -> List[str]:
        """Generate message includes wrapped in ENABLE_ROS2 guards or stub definitions."""
        includes: List[str] = []

        # Collect unique message/service types referenced by the node
        msg_headers: Set[str] = set()
        for pub in node.publishers:
            msg_headers.add(pub.msg_type.replace(".", "/"))
        for sub in node.subscribers:
            msg_headers.add(sub.msg_type.replace(".", "/"))
        for srv in node.services:
            msg_headers.add(srv.srv_type.replace(".", "/"))

        if msg_headers:
            # Wrap real includes in ENABLE_ROS2 so host-only builds don't see them
            includes.append('#if ENABLE_ROS2')
            for header in sorted(msg_headers):
                includes.append(f'#include "{header}.hpp"')
            includes.append('#else')
            includes.append('// Stub message/service types for non-ROS2 builds')
            includes.append('#include <memory>')

            # Generate simple struct stubs (SharedPtr aliases) so code compiles
            stubs: Set[str] = set()
            for header in msg_headers:
                parts = header.split('/')  # e.g. std_msgs/msg/Float32
                if len(parts) < 3:
                    continue
                ns, category, name = parts[0], parts[1], parts[2]
                if category == 'msg':
                    stubs.add(
                        f'namespace {ns} {{\n'
                        f'  namespace msg {{\n'
                        f'    struct {name} {{\n'
                        f'      using SharedPtr = std::shared_ptr<{name}>;\n'
                        f'      using ConstSharedPtr = std::shared_ptr<const {name}>;\n'
                        f'    }};\n'
                        f'  }}\n'
                        f'}}')
                elif category == 'srv':
                    stubs.add(
                        f'namespace {ns} {{\n'
                        f'  namespace srv {{\n'
                        f'    struct {name} {{\n'
                        f'      struct Request {{\n'
                        f'        using SharedPtr = std::shared_ptr<Request>;\n'
                        f'      }};\n'
                        f'      struct Response {{\n'
                        f'        using SharedPtr = std::shared_ptr<Response>;\n'
                        f'      }};\n'
                        f'      using Request = Request;\n'
                        f'      using Response = Response;\n'
                        f'    }};\n'
                        f'  }}\n'
                        f'}}')
            includes.extend(sorted(stubs))
            includes.append('#endif')
            includes.append('')
        return includes
    
    def _generate_node_header(self, node: NodeConfig) -> Path:
        """Generate a C++ header file for a ROS2 node using a template.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated header file
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Initialize context with common values
        context = {
            'include_guard': f'ROBODSL_{node.name.upper()}_NODE_HPP_',
            'class_name': class_name,
            'node_name': node.name,
            'base_class': 'rclcpp_lifecycle::LifecycleNode' if getattr(node, 'lifecycle', False) else 'rclcpp::Node',
            'is_lifecycle': getattr(node, 'lifecycle', False),
            'namespace': getattr(node, 'namespace', ''),
            'includes': [
                '#include <memory>',
                '#include <string>',
                '#include <vector>',
                '#include <rclcpp/rclcpp.hpp>',
                '#include <rclcpp_lifecycle/lifecycle_node.hpp>',
                '#include <rclcpp_lifecycle/lifecycle_publisher.hpp>',
                '#include <std_msgs/msg/string.hpp>',
            ],
            'ros2_includes': [],
            'cuda_includes': [],
            'publishers': [],
            'subscribers': [],
            'services': [],
            'timers': [],
            'parameters': [],
            'cuda_kernels': []
        }
        
        # Add ROS2 includes if needed
        if (hasattr(node, 'services') and node.services) or (hasattr(node, 'parameters') and node.parameters):
            context['ros2_includes'].extend([
                '#include <rclcpp/service.hpp>',
                '#include <rclcpp/parameter.hpp>'
            ])
            
        # Add CUDA includes if needed
        if hasattr(node, 'cuda_kernels') and node.cuda_kernels:
            context['cuda_includes'].extend([
                '#include <cuda_runtime.h>',
                '#include <cstdint>'
            ])
            
        # Add publishers if they exist
        if hasattr(node, 'publishers') and node.publishers:
            context['publishers'] = [{
                'name': pub.topic.lstrip('/').replace('/', '_'),
                'msg_type': pub.msg_type.replace('.', '::'),
                'qos': getattr(pub, 'qos', {})
            } for pub in node.publishers]
            
        # Add subscribers if they exist
        if hasattr(node, 'subscribers') and node.subscribers:
            context['subscribers'] = [{
                'name': sub.topic.lstrip('/').replace('/', '_'),
                'msg_type': sub.msg_type.replace('.', '::'),
                'callback_name': f"{sub.topic.lstrip('/').replace('/', '_')}_callback",
                'qos': getattr(sub, 'qos', {})
            } for sub in node.subscribers]
            
        # Add services if they exist
        if hasattr(node, 'services') and node.services:
            context['services'] = [{
                'name': srv.service.lstrip('/').replace('/', '_'),
                'srv_type': srv.srv_type.replace('.', '::'),
                'callback_name': f"{srv.service.lstrip('/').replace('/', '_')}_callback"
            } for srv in node.services]
            
        # Add timers if they exist
        if hasattr(node, 'timers') and node.timers:
            context['timers'] = [{
                'name': timer.name,
                'callback_name': getattr(timer, 'callback', f"on_timer_{timer.name}"),
                'period': timer.period,
                'autostart': getattr(timer, 'autostart', True)
            } for timer in node.timers]
            
        # Add parameters if they exist
        if hasattr(node, 'parameters') and node.parameters:
            context['parameters'] = [{
                'name': param.name,
                'type': self._map_parameter_type(param.type),
                'default_value': getattr(param, 'default', '0' if param.type in ['int', 'float', 'double'] else 'false')
            } for param in node.parameters]
            
        # Add CUDA kernels if they exist
        if hasattr(node, 'cuda_kernels') and node.cuda_kernels:
            context['cuda_kernels'] = [{
                'name': kernel.name,
                'return_type': 'void',
                'params': ', '.join([f'{p["type"]} {p["name"]}' for p in kernel.get('parameters', [])]),
                'members': [{'type': 'void*', 'name': f'{kernel["name"]}_dev_ptr'}]
            } for kernel in node.cuda_kernels]
        
        # Add message includes
        context['includes'].extend(self._generate_message_includes(node))
        
        # Render template
        template = self.env.get_template('node.hpp.jinja2')
        header_content = template.render(**context)
        
        # Create output directory if it doesn't exist
        node_path = node.name.replace('.', '/')
        header_path = self.output_dir / 'include' / f"{node_path}_node.hpp"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the generated content to file
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(header_content)
            
        return header_path

    def _generate_publisher_declarations(self, publishers: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 publishers."""
        if not publishers:
            return "    // No publishers defined\n"

        decls = []
        for pub in publishers:
            topic = pub['topic']
            msg_type = pub['msg_type'].replace('.', '::')
            var_name = f"{topic.lstrip('/').replace('/', '_')}_pub_"
            # Use SharedPtr for publisher handle
            decls.append(f"    rclcpp::Publisher<{msg_type}>::SharedPtr {var_name};")
            
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

    def _generate_parameter_callback_declarations(self, enabled: bool) -> str:
        """Generate member + prototype declarations for parameter callbacks."""
        if not enabled:
            return ""
        return (
            "    // Parameter callback\n"
            "    rclcpp::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;\n"
            "    #if ENABLE_ROS2\n"
            "    rcl_interfaces::msg::SetParametersResult on_parameter_event(const std::vector<rclcpp::Parameter> & params);\n"
            "    #else\n"
            "    void on_parameter_event(const std::vector<void*>& /*params*/) {}\n"
            "    #endif  // ENABLE_ROS2\n"
        )

    def _generate_timer_declarations(self, timers: List[Dict[str, Any]]) -> str:
        """Generate C++ member declarations for timers."""
        if not timers:
            return "    // No timers defined"
        lines: List[str] = []
        for tmr in timers:
            var_name = f"timer_{tmr['name']}_"
            lines.append(f"    rclcpp::TimerBase::SharedPtr {var_name};")
        return '\n'.join(lines)

    def _generate_action_declarations(self, actions: List[Dict[str, Any]]) -> str:
        """Generate C++ member declarations and callback prototypes for actions."""
        if not actions:
            return "    // No actions defined"

        lines: List[str] = []
        for act in actions:
            action_name = act['name']
            action_type_alias = action_name.capitalize()
            var_name = f"{action_name}_action_server_"
            lines.append(f"    // Action: {action_name}")
            lines.append(f"    rclcpp_action::Server<{action_type_alias}>::SharedPtr {var_name};")
            lines.append("    #if ENABLE_ROS2")
            lines.append(f"    rclcpp_action::GoalResponse handle_goal_{action_name}(const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const {action_type_alias}::Goal> goal);")
            lines.append(f"    rclcpp_action::CancelResponse handle_cancel_{action_name}(const std::shared_ptr<rclcpp_action::ServerGoalHandle<{action_type_alias}>> goal_handle);")
            lines.append(f"    void handle_accepted_{action_name}(const std::shared_ptr<rclcpp_action::ServerGoalHandle<{action_type_alias}>> goal_handle);")
            lines.append("    #else")
            lines.append(f"    void handle_goal_{action_name}(...) {{}}")
            lines.append(f"    void handle_cancel_{action_name}(...) {{}}")
            lines.append(f"    void handle_accepted_{action_name}(...) {{}}")
            lines.append("    #endif  // ENABLE_ROS2")
            lines.append("")
        return '\n'.join(lines)

    def _generate_cuda_kernel_declarations(self, node_name: str, kernels: List[CudaKernelConfig]) -> str:
        """Generate C++ declarations for CUDA kernels used by this node.
        
        Args:
            node_name: Name of the node these kernels belong to
            kernels: List of CUDA kernel configurations
            
        Returns:
            Formatted C++ declarations as a string
        """
        if not kernels:
            return "    // No CUDA kernels defined\n"
            
        # Filter kernels for this node
        node_kernels = [
            k for k in kernels 
            if not hasattr(k, 'node') or k.node == node_name
        ]
        
        if not node_kernels:
            return "    // No CUDA kernels defined for this node\n"
            
        # Prepare context for template
        context = {
            'node': {
                'name': node_name,
                'namespace': self.config.project_name.lower(),
                'cuda_kernels': []
            }
        }
        
        # Add kernel information to context
        for kernel in node_kernels:
            kernel_info = {
                'name': kernel.name,
                'description': getattr(kernel, 'description', ''),
                'parameters': []
            }
            
            # Add kernel parameters
            for param in getattr(kernel, 'parameters', []):
                param_info = {
                    'name': getattr(param, 'name', ''),
                    'type': getattr(param, 'type', 'void*'),
                    'description': getattr(param, 'description', '')
                }
                kernel_info['parameters'].append(param_info)
                
            context['node']['cuda_kernels'].append(kernel_info)
        
        # Render the template
        try:
            template = self.env.get_template('cuda/kernel.hpp.jinja2')
            return template.render(**context)
            
        except Exception as e:
            print(f"Error generating CUDA kernel declarations: {str(e)}")
            return "    // Error generating CUDA kernel declarations\n"
    def _generate_launch_file(self, node: NodeConfig) -> Optional[Path]:
        """Generate a launch file for a ROS2 node using Jinja2 template.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated launch file, or None if not needed
        """
        # Skip if this isn't a ROS2 node
        if not (node.publishers or node.subscribers or node.services or node.parameters or node.actions):
            return None
            
        # Prepare context for the template
        context = {
            'package_name': self.config.project_name,
            'node_name': node.name,
            'is_lifecycle': getattr(node, 'is_lifecycle', False),
            'parameters': [],
            'remappings': {},
            'env_vars': {},
            'extra_nodes': []
        }
        
        # Add namespace if specified
        if hasattr(node, 'namespace') and node.namespace:
            context['namespace'] = node.namespace
            
        # Add parameters
        if hasattr(node, 'parameters') and node.parameters:
            for name, param in node.parameters.items():
                param_dict = {name: param.default} if hasattr(param, 'default') else {name: param}
                context['parameters'].append(param_dict)
                
        # Add remappings
        if hasattr(node, 'remappings') and node.remappings:
            if isinstance(node.remappings, dict):
                context['remappings'] = dict(node.remappings)
            elif isinstance(node.remappings, list):
                for remap in node.remappings:
                    if ':' in remap:
                        src, dst = remap.split(':', 1)
                        context['remappings'][src] = dst
            
        # Add environment variables
        if hasattr(node, 'env') and node.env:
            context['env_vars'] = dict(node.env)
            
        # Handle launch configuration if available
        if hasattr(self.config, 'launch') and hasattr(self.config.launch, 'node'):
            launch_nodes = self.config.launch.node
            if isinstance(launch_nodes, list) and len(launch_nodes) > 1:
                # Add additional nodes from launch config
                for extra_node in launch_nodes[1:]:  # Skip the main node
                    extra_node_dict = {
                        'name': extra_node.name,
                        'metadata': {'package': getattr(extra_node, 'package', None)},
                        'namespace': getattr(extra_node, 'namespace', ''),
                        'parameters': [],
                        'remappings': {}
                    }
                    
                    # Add parameters for extra node
                    if hasattr(extra_node, 'parameters'):
                        if isinstance(extra_node.parameters, dict):
                            extra_node_dict['parameters'].append(extra_node.parameters)
                        elif isinstance(extra_node.parameters, list):
                            extra_node_dict['parameters'].extend(extra_node.parameters)
                    
                    # Add remappings for extra node
                    if hasattr(extra_node, 'remappings'):
                        if isinstance(extra_node.remappings, dict):
                            extra_node_dict['remappings'].update(extra_node.remappings)
                        elif isinstance(extra_node.remappings, list):
                            for remap in extra_node.remappings:
                                if ':' in remap:
                                    src, dst = remap.split(':', 1)
                                    extra_node_dict['remappings'][src] = dst
                    
                    context['extra_nodes'].append(extra_node_dict)
        
        # Render the template
        try:
            template = self.env.get_template('launch/node.launch.py.jinja2')
            launch_content = template.render(**context)
            
            # Ensure output directory exists
            parts = node.name.split('.')
            node_base_name = parts[-1]
            launch_dir = self.output_dir / 'launch' / '/'.join(parts[:-1]) if len(parts) > 1 else self.output_dir / 'launch'
            launch_dir.mkdir(parents=True, exist_ok=True)
            launch_path = launch_dir / f"{node_base_name}.launch.py"
            
            # Write the launch file
            with open(launch_path, 'w', encoding='utf-8') as f:
                f.write(launch_content)
                
            return launch_path
            
        except Exception as e:
            print(f"Error generating launch file for {node.name}: {str(e)}")
            return None

    def _generate_cmake_content(self) -> List[str]:
        """Generate the main CMakeLists.txt content.
        
        Returns:
            List of CMake commands as strings
        """
        return [
            "cmake_minimum_required(VERSION 3.8)",
            f"project({self.config.project_name})",
            "",
            "# Default to C++17",
            "set(CMAKE_CXX_STANDARD 17)",
            "set(CMAKE_CXX_STANDARD_REQUIRED ON)",
            "",
            "# Find dependencies",
            "find_package(ament_cmake REQUIRED)",
            "find_package(rclcpp REQUIRED)",
            "find_package(rclcpp_components REQUIRED)",
            "find_package(std_msgs REQUIRED)",
            "",
            "# Add include directories",
            "include_directories(",
            "    include",
            "    ${rclcpp_INCLUDE_DIRS}",
            "    ${rclcpp_components_INCLUDE_DIRS}",
            ")",
            "",
            "# Add library target",
            f"add_library({self.config.project_name}_lib",
            "    # Add source files here",
            "    # src/common.cpp",
            ")",
            "",
            "# Set C++ standard properties",
            f"target_compile_features({self.config.project_name}_lib PRIVATE cxx_std_17)",
            f"set_target_properties({self.config.project_name}_lib PROPERTIES",
            "    CXX_STANDARD 17",
            "    CXX_STANDARD_REQUIRED ON",
            "    CXX_EXTENSIONS OFF",
            ")",
            "",
            "# Link dependencies",
            f"target_link_libraries({self.config.project_name}_lib",
            "    ${rclcpp_LIBRARIES}",
            "    ${rclcpp_components_LIBRARIES}",
            "    ${std_msgs_LIBRARIES}",
            ")",
            ""
        ]

    def _generate_cmakelists(self) -> Path:
        """Generate the main CMakeLists.txt file using Jinja2 template.
        
        Returns:
            Path to the generated CMakeLists.txt file
        """
        # Prepare context for the template
        context = {
            'project_name': self.config.project_name,
            'version': '0.1.0',
            'has_cuda': any(hasattr(node, 'cuda_kernels') and node.cuda_kernels 
                          for node in self.config.nodes),
            'has_lifecycle': any(hasattr(node, 'is_lifecycle') and node.is_lifecycle 
                              for node in self.config.nodes),
            'has_qos': any(hasattr(node, 'qos_profiles') and node.qos_profiles 
                         for node in self.config.nodes),
            'nodes': []
        }
        
        # Add node-specific information
        for node in self.config.nodes:
            node_path = node.name.replace('.', '/')
            node_info = {
                'name': node.name,
                'type': 'lifecycle' if hasattr(node, 'is_lifecycle') and node.is_lifecycle else 'regular',
                'has_cuda': hasattr(node, 'cuda_kernels') and bool(node.cuda_kernels),
                'sources': [f'src/{node_path}_node.cpp']
            }
            context['nodes'].append(node_info)
        
        # Render the template
        template = self.env.get_template('cmake/CMakeLists.txt.jinja2')
        cmake_content = template.render(**context)
        
        # Ensure output directory exists
        cmake_path = self.output_dir / 'CMakeLists.txt'
        cmake_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the CMakeLists.txt file
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
            
        return cmake_path

    def _generate_package_xml(self) -> Path:
        """Generate the package.xml file using Jinja2 template.
        
        Returns:
            Path to the generated package.xml file
        """
        # Prepare context for the template
        context = {
            'project_name': self.config.project_name,
            'description': 'Auto-generated ROS 2 package',
            'maintainer_email': 'user@example.com',
            'maintainer_name': 'User',
            'license': 'Apache License 2.0',
            'has_lifecycle': any(hasattr(node, 'is_lifecycle') and node.is_lifecycle 
                              for node in self.config.nodes),
            'message_dependencies': self._collect_message_dependencies()
        }
        
        # Render the template
        template = self.env.get_template('cmake/package.xml.jinja2')
        package_content = template.render(**context)
        
        # Ensure output directory exists
        package_path = self.output_dir / 'package.xml'
        package_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the package.xml file
        with open(package_path, 'w') as f:
            f.write(package_content)
            
        return package_path
        
    def _collect_message_dependencies(self) -> Set[str]:
        """Collect all message dependencies from nodes.
        
        Returns:
            Set of package names that provide message types used in the project
        """
        message_packages = set()
        
        # Add standard ROS 2 message packages
        message_packages.update(['std_msgs'])
        
        # Check all nodes for message dependencies
        for node in self.config.nodes:
            # Check publishers
            if hasattr(node, 'publishers'):
                for pub in node.publishers:
                    if 'type' in pub and '/' in pub['type']:
                        pkg = pub['type'].split('/')[0]
                        if pkg != 'std_msgs':  # Already added
                            message_packages.add(pkg)
            
            # Check subscribers
            if hasattr(node, 'subscribers'):
                for sub in node.subscribers:
                    if 'type' in sub and '/' in sub['type']:
                        pkg = sub['type'].split('/')[0]
                        if pkg != 'std_msgs':
                            message_packages.add(pkg)
            
            # Check services
            if hasattr(node, 'services'):
                for srv in node.services:
                    if 'type' in srv and '/' in srv['type']:
                        pkg = srv['type'].split('/')[0]
                        message_packages.add(pkg)
            
            # Check actions
            if hasattr(node, 'actions'):
                for action in node.actions:
                    if 'type' in action and '/' in action['type']:
                        pkg = action['type'].split('/')[0]
                        message_packages.add(pkg)
        
        return sorted(message_packages)

    def _add_cuda_support(self, cmake_content: List[str]) -> List[str]:
        """Add CUDA support to the CMake configuration.
        
        Args:
            cmake_content: The current CMake content
            
        Returns:
            Updated CMake content with CUDA support
        """
        if not any(hasattr(node, 'cuda_kernels') and node.cuda_kernels for node in self.config.nodes):
            return cmake_content
            
        # Add CUDA language and find packages
        cuda_content = [
            "",
            "# Enable CUDA language",
            "enable_language(CUDA)",
            "find_package(CUDA REQUIRED)",
            "",
            "# Set CUDA architecture flags",
            "set(CUDA_ARCHITECTURES \"all-major\")",
            "set(CMAKE_CUDA_STANDARD 17)",
            "set(CMAKE_CUDA_STANDARD_REQUIRED ON)",
            "",
            "# Add CUDA include directories",
            "include_directories(${CUDA_INCLUDE_DIRS})",
            "",
            "# Add CUDA flags",
            "set(CMAKE_CUDA_FLAGS \"${{CMAKE_CUDA_FLAGS}} -O3 --use_fast_math -Xcompiler -fPIC\")",
            ""
        ]
        
        # Insert after the initial CMake configuration
        insert_pos = next((i for i, line in enumerate(cmake_content) 
                         if line.startswith('set(CMAKE_CXX_STANDARD')), 0) + 2
        cmake_content[insert_pos:insert_pos] = cuda_content
        
        # Add CUDA libraries to the main library
        lib_link_pos = next((i for i, line in enumerate(cmake_content) 
                           if f'target_link_libraries({self.config.project_name}_lib' in line), -1)
        if lib_link_pos != -1:
            cmake_content.insert(lib_link_pos + 1, "    ${CUDA_LIBRARIES}")
            
        return cmake_content

    def _add_ros2_dependencies(self, cmake_content: List[str]) -> List[str]:
        """Add ROS2 dependencies to the CMake configuration.
        
        Args:
            cmake_content: The current CMake content
            
        Returns:
            Updated CMake content with ROS2 dependencies
        """
        if not any(hasattr(node, 'uses_ros2') and node.uses_ros2 for node in self.config.nodes):
            return cmake_content
            
        # Add ROS2 specific dependencies
        ros2_content = [
            "",
            "# Find ROS2 packages",
            "find_package(rclcpp REQUIRED)",
            "find_package(rclcpp_components REQUIRED)",
            "find_package(std_msgs REQUIRED)",
            "",
            "# Add ROS2 include directories",
            "include_directories(",
            "    ${rclcpp_INCLUDE_DIRS}",
            "    ${rclcpp_components_INCLUDE_DIRS}",
            "    ${std_msgs_INCLUDE_DIRS}",
            ")",
            ""
        ]
        
        # Insert after the initial CMake configuration
        insert_pos = next((i for i, line in enumerate(cmake_content) 
                         if line.startswith('set(CMAKE_CXX_STANDARD')), 0) + 2
        cmake_content[insert_pos:insert_pos] = ros2_content
        
        return cmake_content
        
    def _add_node_executables(self, cmake_content: List[str]) -> List[str]:
        """Add node executables to the CMake configuration.
        
        Args:
            cmake_content: The current CMake content
            
        Returns:
            Updated CMake content with node executables
        """
        for node in self.config.nodes:
            if not hasattr(node, 'uses_ros2') or not node.uses_ros2:
                continue
                
            node_name = node.name
            cmake_content.extend([
                "",
                f"# {node_name} node",
                f"add_executable({node_name}_node",
                f"    src/nodes/{node_name}.cpp",
                ")",
                "",
                f"target_link_libraries({node_name}_node",
                f"    {self.config.project_name}_lib"
            ])
            
            # Add ROS2 libraries
            cmake_content.append("    ${rclcpp_LIBRARIES}")
            
            # Add lifecycle libraries if needed
            if hasattr(node, 'lifecycle') and node.lifecycle:
                cmake_content.extend([
                    "    ${rclcpp_lifecycle_LIBRARIES}",
                    "    ${lifecycle_msgs_LIBRARIES}"
                ])
                
            # Add QoS libraries if needed
            if hasattr(node, 'has_qos_settings') and node.has_qos_settings:
                cmake_content.extend([
                    "    ${rcl_LIBRARIES}",
                    "    ${rmw_implementation_LIBRARIES}"
                ])
                
            cmake_content.extend([
                ")",
                "",
                f"# Register {node_name} as a component",
                f"rclcpp_components_register_nodes({node_name}_node "
                f'"{self.config.project_name}::{node_name}Component")',
                "",
                f"# Install {node_name} node",
                f"install(TARGETS {node_name}_node",
                "    DESTINATION lib/${PROJECT_NAME}",
                ")"
            ])
            
        return cmake_content
        
        # Add installation rules
        cmake_content.extend([
            "# Install headers",
            "install(",
            "    DIRECTORY include/",
            "    DESTINATION include",
            "    FILES_MATCHING",
            "    PATTERN \"*.hpp\"",
            "    PATTERN \"*.h\"",
            ")",
            "",
            "# Install Python modules if they exist",
            "if(EXISTS \"${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}\")",
            "    install(DIRECTORY ${PROJECT_NAME}/",
            "        DESTINATION lib/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/${PROJECT_NAME}",
            "        PATTERN \"*.py\"",
            "        PATTERN \"__pycache__\" EXCLUDE",
            "    )",
            "endif()",
            ""
        ])
        
        # Write the CMakeLists.txt file
        cmake_path = self.output_dir / 'CMakeLists.txt'
        cmake_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cmake_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(cmake_content))
            
        return cmake_path

    def _generate_cuda_section(self) -> str:
        """Generate the CUDA-specific CMake configuration."""
        return """
# CUDA configuration
if(ENABLE_CUDA)
    add_compile_definitions(WITH_CUDA=1)
    
    # Set CUDA architecture
    set(CUDA_ARCHITECTURES "native")
    
    # CUDA compile options
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} -std=c++17 --expt-relaxed-constexpr -Xcompiler -fPIC")
    
    # Find CUDA
    find_package(CUDA REQUIRED)
    
    # CUDA library
    add_library(cuda_utils SHARED
        ${{CMAKE_CURRENT_SOURCE_DIR}}/src/cuda/cuda_utils.cu
    )
    
    # Set CUDA properties
    set_target_properties(cuda_utils PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )
    
    # Link against CUDA libraries
    target_link_libraries(cuda_utils
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    # Install CUDA library
    install(TARGETS cuda_utils
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
    )
else()
    add_compile_definitions(
        WITH_CUDA=0
        WITH_THRUST=0
    )
endif()
"""

    def _generate_ros2_section(self, has_lifecycle_nodes: bool, has_qos_settings: bool) -> str:
        """Generate the ROS2-specific CMake configuration."""
        lifecycle_deps = """
    find_package(rclcpp_lifecycle REQUIRED)
    find_package(lifecycle_msgs REQUIRED)""" if has_lifecycle_nodes else ""
        
        qos_deps = """
    find_package(rmw_implementation_cmake REQUIRED)
    find_package(rcl REQUIRED)
    find_package(rmw REQUIRED)""" if has_qos_settings else ""
        
        return f"""
# ROS2 configuration
if(ENABLE_ROS2)
    find_package(ament_cmake REQUIRED)
    find_package(rclcpp REQUIRED)
    find_package(rclcpp_components REQUIRED)
    find_package(std_msgs REQUIRED)
    {lifecycle_deps}
    {qos_deps}
    
    # Include directories
    include_directories(
        ${{CMAKE_CURRENT_SOURCE_DIR}}/include
        ${{rclcpp_INCLUDE_DIRS}}
        ${{rclcpp_components_INCLUDE_DIRS}}
        ${{rosidl_default_generators_INCLUDE_DIRS}}
        ${{rosidl_default_runtime_INCLUDE_DIRS}}
    )
    
    # Handle custom message generation
    if(EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/msg" OR 
       EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/srv" OR 
       EXISTS "${{CMAKE_CURRENT_SOURCE_DIR}}/action")
        # Find all .msg, .srv, .action files
        ament_export_dependencies(
            rclcpp_lifecycle
            lifecycle_msgs
        )
    endif()
    
    # Add QoS dependencies if needed
    if({str(has_qos_settings).upper()})
        ament_export_dependencies(
            rmw_implementation
            rcl
            rmw
        )
    endif()
    
    # Install package.xml
    install(FILES package.xml
        DESTINATION share/${{PROJECT_NAME}}
    )
    endforeach()
    
    # Install node executables with proper destination
    install(TARGETS ${{NODE_NAMES}}
        RUNTIME DESTINATION lib/${{PROJECT_NAME}}
        LIBRARY DESTINATION lib/${{PROJECT_NAME}}
        ARCHIVE DESTINATION lib/${{PROJECT_NAME}}
    )
    
    # Generate and install environment hooks
    ament_environment_hooks("${{CMAKE_CURRENT_SOURCE_DIR}}/env-hooks/99-{project_name}.dsv.in")
    
    # Generate and install package.xml
    ament_package(
        CONFIG_EXTRAS "{project_name}-extras.cmake.in"
    )
endif()
""".format(
            project_name=self.config.project_name,
            has_lifecycle='ON' if has_lifecycle_nodes else 'OFF',
            has_qos='ON' if has_qos_settings else 'OFF',
            node_names=' '.join([node.name + '_node' for node in self.config.nodes])
        )

        # Add uninstall target
        cmake_content += """
add_custom_target(uninstall
    COMMAND ${{CMAKE_COMMAND}} -P ${{CMAKE_CURRENT_BINARY_DIR}}/ament_cmake_uninstall.cmake
)
"""
        # Create output directory if it doesn't exist
        cmake_path = self.output_dir / 'CMakeLists.txt'
        cmake_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format the main CMake content with double braces escaped
        cmake_content = cmake_content.format(
            project_name=self.config.project_name,
            has_lifecycle='ON' if has_lifecycle_nodes else 'OFF',
            has_qos='ON' if has_qos_settings else 'OFF',
            has_lifecycle_int=1 if has_lifecycle_nodes else 0,
            has_qos_int=1 if has_qos_settings else 0,
            has_ns_int=1 if has_namespaces else 0
        )

        # Add ROS2 dependencies if enabled
        if uses_ros2:
            ros2_content = """
if(ENABLE_ROS2)
    ament_target_dependencies(${{PROJECT_NAME}}_lib
        rclcpp
        rclcpp_components
        std_msgs
        # Add other ROS2 dependencies here
    )
    
    # Add lifecycle dependencies if needed
    if({has_lifecycle})
        ament_target_dependencies(${{PROJECT_NAME}}_lib
            rclcpp_lifecycle
            lifecycle_msgs
        )
    endif()
    
    # Add QoS dependencies if needed
    if({has_qos})
        ament_target_dependencies(${{PROJECT_NAME}}_lib
            rcl
            rmw_implementation
        )
    endif()
    
    # Add component registration
    rclcpp_components_register_nodes(${{PROJECT_NAME}}_lib "robodsl::Node")
endif()
""".format(
                has_lifecycle='ON' if has_lifecycle_nodes else 'OFF',
                has_qos='ON' if has_qos_settings else 'OFF'
            )
            cmake_content += ros2_content

        # Add node executables
        for node in self.config.nodes:
            node_name = node.name
            node_namespace = getattr(node, 'namespace', '')
            is_lifecycle = getattr(node, 'is_lifecycle_node', False)
            
            node_content = f"""
# {node_name} node
add_executable({node_name}_node
    src/{node_name}_node.cpp
)

# Set node properties
set_target_properties({node_name}_node PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add include directories
target_include_directories({node_name}_node PRIVATE
    ${{CMAKE_CURRENT_SOURCE_DIR}}/include
    ${{CMAKE_CURRENT_SOURCE_DIR}}/include/robodsl
)

# Link against main library
target_link_libraries({node_name}_node
    ${{PROJECT_NAME}}_lib
)
"""
            cmake_content += node_content
            
            # Add ROS2 specific configurations
            if uses_ros2:
                ros2_node_content = f"""
# Add ROS2 dependencies if enabled
if(ENABLE_ROS2)
    target_link_libraries({node_name}_node
        rclcpp::rclcpp
        rclcpp_components::component
    )
    
    # Add lifecycle dependencies if needed
    if("{{'ON' if is_lifecycle else 'OFF'}}")
        target_link_libraries({node_name}_node
            rclcpp_lifecycle::rclcpp_lifecycle
        )
    endif()
    
    # Add component registration
    rclcpp_components_register_nodes({node_name}_node "robodsl::{node_name}")
endif()
"""
                cmake_content += ros2_node_content

        # Write the CMakeLists.txt file
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
            
        return cmake_path
        
    def _generate_cuda_kernel(self, kernel: CudaKernelConfig) -> List[Path]:
        """Generate CUDA kernel implementation files using Jinja2 templates.
        
        Args:
            kernel: The CUDA kernel configuration
            
        Returns:
            List of paths to generated files (header and source)
        """
        generated_files = []
        
        # Create output directories
        include_dir = self.output_dir / 'include' / self.config.project_name.lower() / 'cuda'
        src_dir = self.output_dir / 'src' / 'cuda'
        
        include_dir.mkdir(parents=True, exist_ok=True)
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare context for templates
        context = {
            'include_guard': f"{self.config.project_name.upper()}_{kernel.name.upper()}_KERNEL_HPP",
            'include_path': f"{kernel.name}_kernel.hpp",
            'namespace': self.config.project_name.lower(),
            'kernel_name': kernel.name,
            'cuda_enabled': True,
            'input_type': 'float',  # Default, could be made configurable
            'output_type': 'float',  # Default, could be made configurable
            'param_type': 'float',   # Default, could be made configurable
            'block_size': getattr(kernel, 'block_size', (256, 1, 1))[0] if hasattr(kernel, 'block_size') else 256,
            'use_thrust': getattr(kernel, 'use_thrust', False),
            'includes': getattr(kernel, 'includes', []),
            'code': getattr(kernel, 'code', '// TODO: Implement kernel logic'),
            'members': [
                {'name': 'input', 'type': 'float'},
                {'name': 'output', 'type': 'float'},
                {'name': 'parameters', 'type': 'float'}
            ]
        }
        
        # Add kernel parameters to context with proper formatting
        kernel_parameters = []
        for p in getattr(kernel, 'parameters', []):
            # Build the parameter type with const and pointer qualifiers
            param_type = p.type
            if getattr(p, 'is_const', False):
                param_type = f"const {param_type}"
            if getattr(p, 'is_pointer', False):
                param_type = f"{param_type}*"
            
            param_info = {
                'name': p.name,
                'type': param_type,
                'description': getattr(p, 'description', ''),
                'is_const': getattr(p, 'is_const', False),
                'is_pointer': getattr(p, 'is_pointer', False),
                'size_expr': getattr(p, 'size_expr', '')
            }
            kernel_parameters.append(param_info)
        context['kernel_parameters'] = kernel_parameters
        
        try:
            # Generate header file using the correct template
            header_template = self.env.get_template('cuda/kernel.cuh.jinja2')
            header_content = header_template.render(**context)
            
            header_path = include_dir / f"{kernel.name}_kernel.hpp"
            with open(header_path, 'w', encoding='utf-8') as f:
                f.write(header_content)
            generated_files.append(header_path)
            
            # Generate source file
            source_template = self.env.get_template('cuda/kernel.cu.jinja2')
            source_content = source_template.render(**context)
            
            source_path = src_dir / f"{kernel.name}_kernel.cu"
            with open(source_path, 'w', encoding='utf-8') as f:
                f.write(source_content)
            generated_files.append(source_path)
            
            return generated_files
            
        except Exception as e:
            print(f"Error generating CUDA kernel {kernel.name}: {str(e)}")
            return []
        
    def _validate_qos_config(self, qos_cfg: Optional[QoSConfig], context: str = '') -> Optional[QoSConfig]:
        """Validate a QoS configuration object.
        
        Args:
            qos_cfg: The QoS configuration object or None
            context: Context string for error messages
            
        Returns:
            The validated QoS configuration object or None
            
        Raises:
            ValueError: If the QoS configuration is invalid
        """
        if qos_cfg is None:
            return None
            
        if not isinstance(qos_cfg, QoSConfig):
            raise ValueError(f"Invalid QoS configuration in {context}, expected QoSConfig object")
                
        # Validate enum values
        if qos_cfg.reliability is not None and qos_cfg.reliability not in {'reliable', 'best_effort'}:
            raise ValueError(f"Invalid reliability value in {context}, must be 'reliable' or 'best_effort'")
            
        if qos_cfg.durability is not None and qos_cfg.durability not in {'volatile', 'transient_local'}:
            raise ValueError(f"Invalid durability value in {context}, must be 'volatile' or 'transient_local'")
            
        if qos_cfg.history is not None and qos_cfg.history not in {'keep_last', 'keep_all'}:
            raise ValueError(f"Invalid history value in {context}, must be 'keep_last' or 'keep_all'")
            
        if qos_cfg.liveliness is not None and qos_cfg.liveliness not in {'automatic', 'manual_by_topic'}:
            raise ValueError(f"Invalid liveliness value in {context}, must be 'automatic' or 'manual_by_topic'")
            
        # Validate numeric values
        numeric_fields = {
            'depth': qos_cfg.depth,
            'deadline': qos_cfg.deadline,
            'lifespan': qos_cfg.lifespan,
            'liveliness_lease_duration': qos_cfg.liveliness_lease_duration
        }
        
        for field, value in numeric_fields.items():
            if value is not None and not isinstance(value, (int, float)):
                raise ValueError(f"Invalid {field} value in {context}, must be a number")
                
        return qos_cfg

    def _generate_qos_config(self, qos_cfg: Optional[QoSConfig], var_name: str) -> str:
        """Generate C++ code to configure a QoS object.
        
        Args:
            qos_cfg: The QoS configuration object or None
            var_name: Name of the QoS variable to configure
            
        Returns:
            C++ code to configure the QoS object
        """
        if qos_cfg is None:
            return ''
            
        code = []
        
        # Handle depth (queue size)
        depth = qos_cfg.depth if qos_cfg.depth is not None else 10
        code.append(f'rclcpp::QoS {var_name}({depth});')
        
        # Handle reliability
        if qos_cfg.reliability == 'best_effort':
            code.append(f'{var_name}.reliability(rclcpp::ReliabilityPolicy::BestEffort);')
        elif qos_cfg.reliability == 'reliable':
            code.append(f'{var_name}.reliability(rclcpp::ReliabilityPolicy::Reliable);')
            
        # Handle durability
        if qos_cfg.durability == 'volatile':
            code.append(f'{var_name}.durability(rclcpp::DurabilityPolicy::Volatile);')
        elif qos_cfg.durability == 'transient_local':
            code.append(f'{var_name}.durability(rclcpp::DurabilityPolicy::TransientLocal);')
            
        # Handle history
        if qos_cfg.history == 'keep_last':
            code.append(f'{var_name}.history(rclcpp::HistoryPolicy::KeepLast);')
        elif qos_cfg.history == 'keep_all':
            code.append(f'{var_name}.history(rclcpp::HistoryPolicy::KeepAll);')
            
        # Handle deadline
        if qos_cfg.deadline is not None:
            code.append(f'{var_name}.deadline(rclcpp::Duration({qos_cfg.deadline}ms));')
            
        # Handle lifespan
        if qos_cfg.lifespan is not None:
            code.append(f'{var_name}.lifespan(rclcpp::Duration({qos_cfg.lifespan}ms));')
            
        # Handle liveliness
        if qos_cfg.liveliness == 'automatic':
            code.append(f'{var_name}.liveliness(rclcpp::LivelinessPolicy::Automatic);')
        elif qos_cfg.liveliness == 'manual_by_topic':
            code.append(f'{var_name}.liveliness(rclcpp::LivelinessPolicy::ManualByTopic);')
            
        # Handle liveliness lease duration
        if qos_cfg.liveliness_lease_duration is not None:
            code.append(f'{var_name}.liveliness_lease_duration(rclcpp::Duration({qos_cfg.liveliness_lease_duration}ms));')
        
        return '\n    '.join(code)

    def _generate_node_source(self, node: NodeConfig) -> Path:
        """Generate a C++ source file for a ROS2 node using a Jinja2 template.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated source file
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Prepare the template context
        context = {
            'include_path': f'robodsl/{node.name}_node.hpp',
            'namespace': getattr(node, 'namespace', ''),
            'class_name': class_name,
            'base_class': 'rclcpp_lifecycle::LifecycleNode' if node.lifecycle else 'rclcpp::Node',
            'node_name': node.name,
            'is_lifecycle': node.lifecycle,
            'parameters': [],
            'publishers': [],
            'subscribers': [],
            'services': [],
            'timers': [],
            'cuda_kernels': []
        }
        
        # Add parameters to context
        for name, type_info in node.parameters.items():
            context['parameters'].append({
                'name': name,
                'type': self._map_parameter_type(type_info),
                'default_value': '0'  # Default value can be customized
            })
        
        # Add publishers to context
        for pub in node.publishers:
            pub_config = {
                'name': pub.topic.lstrip('/').replace('/', '_'),
                'msg_type': pub.msg_type.replace('.', '::'),
                'topic': pub.topic,
                'qos': {}
            }
            if hasattr(pub, 'qos') and pub.qos:
                pub_config['qos'] = {
                    'depth': pub.qos.depth if hasattr(pub.qos, 'depth') else 10,
                    'reliability': pub.qos.reliability if hasattr(pub.qos, 'reliability') else None,
                    'durability': pub.qos.durability if hasattr(pub.qos, 'durability') else None
                }
            context['publishers'].append(pub_config)
        
        # Add subscribers to context
        for sub in node.subscribers:
            sub_config = {
                'name': sub.topic.lstrip('/').replace('/', '_'),
                'msg_type': sub.msg_type.replace('.', '::'),
                'topic': sub.topic,
                'callback_name': f"{sub.topic.lstrip('/').replace('/', '_')}_callback",
                'qos': {}
            }
            if hasattr(sub, 'qos') and sub.qos:
                sub_config['qos'] = {
                    'depth': sub.qos.depth if hasattr(sub.qos, 'depth') else 10,
                    'reliability': sub.qos.reliability if hasattr(sub.qos, 'reliability') else None,
                    'durability': sub.qos.durability if hasattr(sub.qos, 'durability') else None
                }
            context['subscribers'].append(sub_config)
            
        # Add services to context
        for srv in node.services:
            context['services'].append({
                'name': srv.service.lstrip('/').replace('/', '_'),
                'srv_type': srv.srv_type.replace('.', '::'),
                'service': srv.service,
                'callback_name': f"{srv.service.lstrip('/').replace('/', '_')}_callback"
            })
            
        # Add timers to context
        for tmr in node.timers:
            context['timers'].append({
                'name': tmr.name,
                'period': tmr.period_ms / 1000.0,  # Convert ms to seconds
                'callback_name': f"{tmr.name}_callback",
                'autostart': getattr(tmr, 'autostart', True)
            })
            
        # Add CUDA kernels to context if any
        if hasattr(node, 'cuda_kernels'):
            for kernel in node.cuda_kernels:
                context['cuda_kernels'].append({
                    'name': kernel.name,
                    'return_type': kernel.return_type,
                    'params': kernel.params,
                    'members': [{'name': m.name, 'type': m.type} for m in kernel.members]
                })
        
        # Load and render the template
        env = Environment(
            loader=PackageLoader('robodsl', 'templates/cpp'),
            autoescape=select_autoescape(['cpp', 'hpp', 'jinja2']),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters
        env.filters['to_snake_case'] = self._to_snake_case
        
        template = env.get_template('node.cpp.jinja2')
        source_content = template.render(**context)
        
        # Create output directory if it doesn't exist
        source_path = self.output_dir / 'src' / f"{node.name}_node.cpp"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the generated content to file
        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(source_content)
        
        return source_path

    def _map_parameter_type(self, type_info: str) -> str:
        """Map parameter type from DSL to C++ type.
        
        Args:
            type_info: Type information from DSL
            
        Returns:
            Corresponding C++ type
        """
        type_map = {
            'int': 'int',
            'float': 'double',
            'bool': 'bool',
            'string': 'std::string',
            'list': 'std::vector<double>',
            'dict': 'std::map<std::string, std::string>',
            'int[]': 'std::vector<int>',
            'float[]': 'std::vector<float>',
            'double[]': 'std::vector<double>',
            'bool[]': 'std::vector<bool>',
            'string[]': 'std::vector<std::string>',
        }
        return type_map.get(type_info, 'auto')  # Default to auto if type not recognized

