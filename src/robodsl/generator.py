"""
Code generation for RoboDSL.

This module handles the generation of C++ and CUDA source files from the parsed DSL configuration.
"""

import jinja2
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

from .parser import RoboDSLConfig, NodeConfig, CudaKernelConfig, QoSConfig

# Set up Jinja2 environment with template directories
template_dirs = [
    Path(__file__).parent / 'templates/cpp',
    Path(__file__).parent / 'templates/py',
    Path(__file__).parent / 'templates/cmake',
    Path(__file__).parent / 'templates/launch',
    Path(__file__).parent / 'templates',  # For backward compatibility
]

# Filter out non-existent directories
template_loaders = [
    jinja2.FileSystemLoader(str(d)) for d in template_dirs if d.exists()
]

if template_loaders:
    env = jinja2.Environment(
        loader=jinja2.ChoiceLoader(template_loaders),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True
    )
    env.filters['indent'] = lambda text, n: '\n'.join(' ' * n + line if line.strip() else line 
                                                     for line in text.split('\n'))


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

        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / 'templates'
        self.env = Environment(loader=FileSystemLoader(template_dir), trim_blocks=True, lstrip_blocks=True)
        self.env.filters['qos_config'] = self._generate_qos_config
        self.env.filters['map_param_type'] = self._map_parameter_type
        
    def generate(self) -> List[Path]:
        """Generate all source files from the DSL configuration.
        
        Returns:
            List of Path objects for all generated files
        """
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include' / self.config.project_name).mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'src').mkdir(exist_ok=True)
        (self.output_dir / 'launch').mkdir(exist_ok=True)
        (self.output_dir / 'cuda').mkdir(exist_ok=True)
        
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
            kernel_path = self._generate_cuda_kernel(kernel)
            generated_files.append(kernel_path)
        
        # Generate package.xml if this is a ROS2 project
        package_path = self._generate_package_xml()
        if package_path:
            generated_files.append(package_path)
        
        # Generate CMakeLists.txt
        cmake_path = self._generate_cmakelists()
        generated_files.append(cmake_path)
        

        
        return generated_files
    
    def _generate_method_implementations(self, class_name: str, methods: List[Any]) -> List[str]:
        """Generate C++ method implementations from method configurations.
        
        Args:
            class_name: Name of the C++ class
            methods: List of method configurations
            
        Returns:
            List of C++ method implementations as strings
        """
        implementations = []
        for method in methods:
            # Format parameters with types and names
            params = []
            param_names = []
            for param in method.parameters:
                # Split on whitespace and take the last part as the parameter name
                parts = param.split()
                param_type = ' '.join(parts[:-1])
                param_name = parts[-1]
                params.append(f"{param_type} {param_name}")
                param_names.append(param_name)
            
            # Create method implementation
            implementation = f"{method.return_type} {class_name}::{method.name}({', '.join(params)})"
            implementation += f" {{\n{method.implementation}\n}}\n"
            implementations.append(implementation)
        
        return implementations

    def _generate_method_declarations(self, methods: List[Any]) -> List[str]:
        """Generate C++ method declarations from method configurations.
        
        Args:
            methods: List of method configurations
            
        Returns:
            List of C++ method declarations as strings
        """
        declarations = []
        for method in methods:
            # Format parameters with types and names
            params = []
            for param in method.parameters:
                # Split on whitespace and take the last part as the parameter name
                parts = param.split()
                param_type = ' '.join(parts[:-1])
                param_name = parts[-1]
                params.append(f"{param_type} {param_name}")
            
            # Create method declaration
            declaration = f"    virtual {method.return_type} {method.name}({', '.join(params)});"
            declarations.append(declaration)
        
        return declarations

    def _generate_message_includes(self, node: NodeConfig) -> List[str]:
        """Generate message includes wrapped in ENABLE_ROS2 guards or stub definitions."""
        includes: List[str] = []

        # Collect unique message/service types referenced by the node
        msg_headers: Set[str] = set()
        for pub in node.publishers:
            msg_headers.add(pub["msg_type"].replace(".", "/"))
        for sub in node.subscribers:
            msg_headers.add(sub["msg_type"].replace(".", "/"))
        for srv in node.services:
            msg_headers.add(srv["srv_type"].replace(".", "/"))

        if msg_headers:
            # Wrap real includes in ENABLE_ROS2 so host-only builds don’t see them
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
            'base_class': 'rclcpp_lifecycle::LifecycleNode' if node.lifecycle.enabled else 'rclcpp::Node',
            'is_lifecycle': node.lifecycle.enabled,
            'namespace': node.namespace,
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
            'methods': node.methods,  # Pass methods to the template
        }

        # Render template
        template = self.env.get_template('cpp/node.hpp.jinja2')
        header_content = template.render(**context)

        # Create output directory if it doesn't exist
        header_path = self.output_dir / 'include' / self.config.project_name / f"{node.name}.hpp"
        header_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the generated content to file
        with open(header_path, 'w', encoding='utf-8') as f:
            f.write(header_content)

        return header_path

    def _generate_node_source(self, node: NodeConfig) -> Path:
        """Generate a C++ source file for a ROS2 node using a template."""
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))

        context = {
            'class_name': class_name,
            'node_name': node.name,
            'project_name': self.config.project_name,
            'node': node,
            'is_lifecycle': node.lifecycle.enabled,
            'methods': node.methods,
            'cuda_kernels': self.config.cuda_kernels,
        }

        template = self.env.get_template('cpp/node.cpp.jinja2')
        source_content = template.render(**context)

        source_path = self.output_dir / 'src' / f"{node.name}.cpp"
        source_path.parent.mkdir(parents=True, exist_ok=True)

        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(source_content)

        return source_path

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
            var_name = f"timer_{tmr.name}_"
            lines.append(f"    rclcpp::TimerBase::SharedPtr {var_name};")
        return '\n'.join(lines)

    def _generate_action_declarations(self, actions: List[Dict[str, Any]]) -> str:
        """Generate C++ member declarations and callback prototypes for actions."""
        if not actions:
            return "    // No actions defined"

        lines: List[str] = []
        for act in actions:
            action_name = act.name
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
        """Generate C++ declarations for CUDA kernels used by this node."""
        if not kernels:
            return ""

        node_kernels = [k for k in kernels if not hasattr(k, 'node') or k.node == node_name]
        if not node_kernels:
            return ""

        declarations = ["#if ENABLE_CUDA", "// CUDA kernel forward declarations"]
        for kernel in node_kernels:
            params = [f"{p.type} {p.name}" for p in kernel.parameters]
            param_str = ", ".join(params)
            declarations.append(f"void {kernel.name}_launch({param_str});")
        declarations.append("#endif // ENABLE_CUDA")
        return '\n'.join(declarations)

    def _generate_cuda_kernel(self, kernel: CudaKernelConfig) -> Path:
        """Generate a .cu source file for a single CUDA kernel."""
        context = {
            'kernel': kernel,
            'project_name': self.config.project_name,
        }
        template = self.env.get_template('cuda/kernel.cu.jinja2')
        content = template.render(**context)
        
        output_path = self.output_dir / 'cuda' / f"{kernel.name}.cu"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return output_path
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
            for param in node.parameters:
                context['parameters'].append({param.name: param.default})
                
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
            launch_dir = self.output_dir / 'launch'
            launch_dir.mkdir(parents=True, exist_ok=True)
            launch_path = launch_dir / f"{node.name}.launch.py"

            # Write the launch file
            with open(launch_path, 'w', encoding='utf-8') as f:
                f.write(launch_content)

            return launch_path

        except Exception as e:
            print(f"Error generating launch file for {node.name}: {str(e)}")
            return None

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
            'nodes': [],
            'message_dependencies': self._collect_message_dependencies()
        }

        # Add node-specific information
        for node in self.config.nodes:
            node_info = {
                'name': node.name,
                'type': 'lifecycle' if hasattr(node, 'is_lifecycle') and node.is_lifecycle else 'regular',
                'has_cuda': hasattr(node, 'cuda_kernels') and bool(node.cuda_kernels),
                'sources': [f'src/{node.name}.cpp']
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
        """Collect all message dependencies from nodes."""
        message_packages: Set[str] = set()
        for node in self.config.nodes:
            for pub in node.publishers:
                if hasattr(pub, 'msg_type') and '/' in pub.msg_type:
                    message_packages.add(pub.msg_type.split('/')[0])
            for sub in node.subscribers:
                if hasattr(sub, 'msg_type') and '/' in sub.msg_type:
                    message_packages.add(sub.msg_type.split('/')[0])
            for srv in node.services:
                if hasattr(srv, 'srv_type') and '/' in srv.srv_type:
                    message_packages.add(srv.srv_type.split('/')[0])
            for action in node.actions:
                if hasattr(action, 'action_type') and '/' in action.action_type:
                    message_packages.add(action.action_type.split('/')[0])
        return message_packages

    def _generate_cuda_kernel(self, kernel: CudaKernelConfig) -> List[Path]:
        """Generate CUDA kernel implementation files using Jinja2 templates.
        
        Args:
            kernel: The CUDA kernel configuration
            
        Returns:
            List of paths to generated files (header and source)
        """
        generated_files = []
        
        # Create output directories
        include_dir = Path(self.output_dir) / 'include' / self.config.project_name / 'cuda'
        src_dir = self.output_dir / 'src' / 'cuda'
        
        include_dir.mkdir(parents=True, exist_ok=True)
        src_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare context for templates
        context = {
            'node': {
                'name': getattr(kernel, 'node', 'unnamed'),
                'namespace': self.config.project_name,
                'cuda_kernels': [{
                    'name': kernel.name,
                    'description': getattr(kernel, 'description', ''),
                    'parameters': [{
                        'name': p.name,
                        'type': p.type,
                        'description': getattr(p, 'description', ''),
                        'is_const': getattr(p, 'is_const', False),
                        'is_pointer': getattr(p, 'is_pointer', False),
                        'size_expr': getattr(p, 'size_expr', '')
                    } for p in getattr(kernel, 'parameters', [])]
                }]
            },
            'kernel': {
                'name': kernel.name,
                'use_thrust': getattr(kernel, 'use_thrust', False),
                'includes': getattr(kernel, 'includes', []),
                'code': getattr(kernel, 'code', '// TODO: Implement kernel logic')
            },
            'inputs': [{
                'name': p.name,
                'type': p.type
            } for p in getattr(kernel, 'inputs', [])],
            'outputs': [{
                'name': p.name,
                'type': p.type
            } for p in getattr(kernel, 'outputs', [])]
        }
        
        try:
            # Generate header file
            header_template = self.env.get_template('cuda/kernel.cuh.jinja2')
            header_content = header_template.render(**context)
            
            header_path = include_dir / f"{kernel.name}_kernels.cuh"
            with open(header_path, 'w', encoding='utf-8') as f:
                f.write(header_content)
            generated_files.append(header_path)
            
            # Generate source file
            source_template = self.env.get_template('cuda/kernel.cu.jinja2')
            source_content = source_template.render(**context)
            
            source_path = src_dir / f"{kernel.name}_kernels.cu"
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
        """Generate a C++ source file for a ROS2 node using a Jinja2 template."""
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))

        context = {
            'project_name': self.config.project_name,
            'class_name': class_name,
            'node_name': node.name,
            'namespace': node.namespace,
            'is_lifecycle': node.lifecycle.enabled,
            'publishers': node.publishers,
            'subscribers': node.subscribers,
            'services': node.services,
            'actions': node.actions,
            'parameters': node.parameters,
            'timers': node.timers,
            'parameter_callbacks': node.parameter_callbacks,
            'remaps': getattr(node, 'remap', []),
            'methods': node.methods,
        }

        template = self.env.get_template('cpp/node.cpp.jinja2')
        source_code = template.render(**context)

        source_path = self.output_dir / 'src' / f"{node.name}.cpp"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, 'w', encoding='utf-8') as f:
            f.write(source_code)

        return source_path

    def _map_parameter_type(self, type_info: str) -> str:
        """Map parameter type from DSL to C++ type.

        Args:
            type_info: Type information from DSL (e.g., 'string', 'int', 'double[]')

        Returns:
            Corresponding C++ type
        """
        type_map = {
            'string': 'std::string',
            'int': 'int',
            'double': 'double',
            'bool': 'bool',
            'string[]': 'std::vector<std::string>',
            'int[]': 'std::vector<int>',
            'double[]': 'std::vector<double>',
            'bool[]': 'std::vector<bool>'
        }
        return type_map.get(type_info, 'std::string')
