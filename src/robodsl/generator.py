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
        has_actions = bool(node.actions)
        has_timers = bool(node.timers)
        use_lifecycle = node.lifecycle

        if uses_ros2:
            includes.insert(0, '#if ENABLE_ROS2')
            includes.insert(1, '#include "rclcpp/rclcpp.hpp"')
            if use_lifecycle:
                includes.insert(2, '#include "rclcpp_lifecycle/lifecycle_node.hpp"')
            if has_actions:
                includes.append('#include "rclcpp_action/rclcpp_action.hpp"')
            if has_param_cb:
                includes.append('#include "rcl_interfaces/msg/set_parameters_result.hpp"')
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
        param_cb_decl = self._generate_parameter_callback_declarations(node.parameter_callbacks)
        timers_decl = self._generate_timer_declarations(node.timers)
        actions_decl = self._generate_action_declarations(node.actions)
        cuda_kernels = self._generate_cuda_kernel_declarations(node.name, self.config.cuda_kernels)
        
        # Generate the class declaration with conditional ROS2 support
        class_declaration = []
        
        if uses_ros2:
            base_class = 'rclcpp::LifecycleNode' if use_lifecycle else 'rclcpp::Node'
            class_declaration.append(f'class {class_name} : public {base_class} {{')
            class_declaration.append('public:')
                        # Build NodeOptions with namespace/remap if provided
            ns_option = f'.namespace_("{node.namespace}")' if node.namespace else ''
            remap_chain = ''
            if node.remap:
                for frm, to in node.remap.items():
                    remap_chain += f'.append_remap_rule("{frm}:={to}")'
            node_options_expr = f'rclcpp::NodeOptions(){ns_option}{remap_chain}'
            class_declaration.append(f'    {class_name}() : {base_class}("{node.name}", {node_options_expr}) {{}}')
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
       
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Generate include guard
        include_guard = f"ROBODSL_{node.name.upper()}_NODE_HPP_"
        
        # Check if this node uses ROS2 features
        uses_ros2 = bool(node.publishers or node.subscribers or node.services or node.parameters)
        
        # Generate base includes
        header_content = "#pragma once\n\n"
        
        # System includes
        header_content += "#include <memory>\n"
        header_content += "#include <string>\n"
        header_content += "#include <vector>\n"
        header_content += "#include <functional>\n\n"
        
        # ROS2 includes if needed
        if node.publishers or node.subscribers or node.services or node.parameters:
            header_content += "#if ENABLE_ROS2\n"
            header_content += "#include \"rclcpp/rclcpp.hpp\"\n"
            header_content += "#include \"std_msgs/msg/float32_multi_array.hpp\"\n"
            header_content += "#include \"std_msgs/msg/header.hpp\"\n"
            header_content += "#else\n"
            header_content += "// ROS2 stubs for non-ROS2 builds\n"
            header_content += "#include <cstdint>\n"
            header_content += "#include <string>\n"
            header_content += "#include <vector>\n"
            header_content += "#include <chrono>\n"
            header_content += "#include <memory>\n"
            header_content += "#include <map>\n"
            header_content += "#include <functional>\n\n"
            header_content += "// ROS2 stubs\n"
            header_content += "namespace rclcpp {\n"
            header_content += "class Node : public std::enable_shared_from_this<Node> {\n"
            header_content += "public:\n"
            header_content += "  using SharedPtr = std::shared_ptr<Node>;\n"
            header_content += "  using WeakPtr = std::weak_ptr<Node>;\n"
            header_content += "  using ConstSharedPtr = const std::shared_ptr<const Node>;\n"
            header_content += "  using ConstWeakPtr = const std::weak_ptr<const Node>;\n"
            header_content += "  using SharedPtrConst = std::shared_ptr<const Node>;\n"
            header_content += "  using WeakPtrConst = std::weak_ptr<const Node>;\n\n"
            header_content += "  virtual ~Node() = default;\n"
            header_content += "  static SharedPtr make_shared(const std::string &name) { return std::make_shared<Node>(); }\n"
            header_content += "  std::string get_name() const { return \"stub_node\"; }\n"
            header_content += "};\n\n"
            header_content += "class Parameter {\n"
            header_content += "public:\n"
            header_content += "  Parameter() = default;\n"
            header_content += "  template<typename T>\n"
            header_content += "  T get_value(const std::string &name, const T &default_value) const { return default_value; }\n"
            header_content += "};\n"
            header_content += "}  // namespace rclcpp\n\n"
            header_content += "// std_msgs stubs\n"
            header_content += "namespace std_msgs {\n"
            header_content += "namespace msg {\n"
            header_content += "struct Header {\n"
            header_content += "  struct Stamp {\n"
            header_content += "    int32_t sec = 0;\n"
            header_content += "    uint32_t nanosec = 0;\n"
            header_content += "  };\n"
            header_content += "  Stamp stamp;\n"
            header_content += "  std::string frame_id;\n"
            header_content += "};\n\n"
            header_content += "template<typename T>\n"
            header_content += "class Float32MultiArray_ {\n"
            header_content += "public:\n"
            header_content += "  using SharedPtr = std::shared_ptr<Float32MultiArray_<T>>;\n"
            header_content += "  using ConstSharedPtr = std::shared_ptr<const Float32MultiArray_<T>>;\n"
            header_content += "  Header header;\n"
            header_content += "  std::vector<float> data;\n"
            header_content += "};\n"
            header_content += "using Float32MultiArray = Float32MultiArray_<void>;\n"
            header_content += "}  // namespace msg\n"
            header_content += "}  // namespace std_msgs\n"
            header_content += "#endif  // ENABLE_ROS2\n\n"
        
        # Add class definition
        header_content += f"namespace robodsl {{\n\n"
        header_content += f"class {node.name.capitalize()}Node {{\n"
        # Add public section
        header_content += "public:\n"
        header_content += f"  explicit {node.name.capitalize()}Node();\n"
        header_content += f"  ~{node.name.capitalize()}Node() = default;\n\n"
        
        # Add private methods and members
        header_content += "private:\n"
        
        # Add ROS2 node if needed
        if node.publishers or node.subscribers or node.services or node.parameters:
            header_content += "  #if ENABLE_ROS2\n"
            header_content += "  rclcpp::Node::SharedPtr node_ = nullptr;\n"
            header_content += "  #else\n"
            header_content += "  // ROS2 stubs for non-ROS2 builds\n"
            header_content += "  std::shared_ptr<void> node_ = nullptr;  // Placeholder for ROS2 node\n"
            header_content += "  #endif  // ENABLE_ROS2\n\n"
        # Timer and action member declarations (inside class)
        header_content += timers_decl + "\n\n" + actions_decl + "\n" + param_cb_decl + "\n"
        
        header_content += f"}};  // class {node.name.capitalize()}Node\n\n"
        header_content += "}}  // namespace robodsl\n"
        

    """
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
        """Generate C++ declarations for CUDA kernels used by this node."""
        if not kernels:
            return "    // No CUDA kernels defined\n"
            
        decls = ["    // CUDA Kernels"]
        for kernel in kernels:
            # Skip if kernel has a node attribute that doesn't match
            if hasattr(kernel, 'node') and kernel.node != node_name:
                continue
                
            # Add kernel declaration
            decls.append(f"    // CUDA kernel: {kernel.name}")
            decls.append(f"    // Add kernel declarations here")
                
        return '\n'.join(decls)
        
    def _generate_launch_file(self, node: NodeConfig) -> Optional[Path]:
        """Generate a launch file for a ROS2 node.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated launch file, or None if not needed
        """
        # Check if this node uses ROS2 features
        uses_ros2 = bool(node.publishers or node.subscribers or node.services or node.parameters or node.actions)
        
        # Only generate launch file for ROS2 nodes
        if not uses_ros2:
            return None
            
        # Get launch configuration if available
        launch_config = getattr(self.config, 'launch', None)
        
        # Generate launch file content
        package_name = self.config.project_name
        exec_name = f"{node.name}_node"
        node_name = node.name
        
        # Default node configuration
        node_config = {
            'package': package_name,
            'executable': exec_name,
            'name': node_name,
            'output': 'screen',
            'parameters': [{'use_sim_time': False}],
            'remappings': [],
            'namespace': ''
        }
        
        # Apply launch configuration if available
        if launch_config and hasattr(launch_config, 'node'):
            launch_node = launch_config.node[0] if isinstance(launch_config.node, list) else launch_config.node
            
            # Update node configuration from launch config
            if hasattr(launch_node, 'package'):
                node_config['package'] = launch_node.package
            if hasattr(launch_node, 'executable'):
                node_config['executable'] = launch_node.executable
            if hasattr(launch_node, 'name'):
                node_config['name'] = launch_node.name
            if hasattr(launch_node, 'output'):
                node_config['output'] = launch_node.output
            if hasattr(launch_node, 'namespace'):
                node_config['namespace'] = launch_node.namespace
            
            # Handle parameters
            if hasattr(launch_node, 'parameters'):
                if isinstance(launch_node.parameters, list):
                    node_config['parameters'] = launch_node.parameters
            
            # Handle remappings
            if hasattr(launch_node, 'remappings'):
                if isinstance(launch_node.remappings, list):
                    node_config['remappings'] = [
                        f"{src}:={dst}" for src, dst in 
                        (remap.split(':', 1) for remap in launch_node.remappings)
                    ]
            
            # Handle environment variables
            env_vars = {}
            if hasattr(launch_node, 'env'):
                if isinstance(launch_node.env, dict):
                    env_vars.update(launch_node.env)
            if env_vars:
                node_config['env'] = env_vars
        
        # Add node parameters if they exist
        if node.parameters:
            params = {}
            for param_name, param_info in node.parameters.items():
                if isinstance(param_info, dict) and 'default' in param_info:
                    params[param_name] = param_info['default']
                else:
                    params[param_name] = param_info
            
            if 'parameters' not in node_config or not isinstance(node_config['parameters'], list):
                node_config['parameters'] = []
            
            # Add parameters as a dictionary
            node_config['parameters'].append(params)
        
        # Generate launch file content
        launch_content = """# Generated by RoboDSL - DO NOT EDIT
"""
        
        # Add imports
        launch_content += """from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, LifecycleNode
from ament_index_python.packages import get_package_share_directory
import os

"""
        
        # Generate launch description function
        launch_content += "def generate_launch_description():\n"
        
        # Add launch arguments
        launch_content += "    # Launch arguments\n"
        launch_content += "    use_sim_time = LaunchConfiguration('use_sim_time', default='false')\n"
        launch_content += "    use_rviz = LaunchConfiguration('use_rviz', default='false')\n\n"
        
        # Add node configuration
        launch_content += "    # Node configuration\n"
        
        # Handle lifecycle nodes
        is_lifecycle = getattr(node, 'lifecycle', False)
        node_type = 'LifecycleNode' if is_lifecycle else 'Node'
        
        # Start node configuration
        launch_content += f"    {node_name}_node = {node_type}(\n"
        
        # Add basic node configuration
        launch_content += f"        package='{node_config['package']}',\n"
        launch_content += f"        executable='{node_config['executable']}',\n"
        launch_content += f"        name='{node_config['name']}',\n"
        
        # Add namespace if specified
        if node_config.get('namespace'):
            launch_content += f"        namespace='{node_config['namespace']}',\n"
        
        # Add output configuration
        launch_content += f"        output='{node_config['output']}',\n"
        
        # Add parameters
        if node_config.get('parameters'):
            params_str = "        parameters=[\n"
            for param in node_config['parameters']:
                if isinstance(param, dict):
                    params_str += "            {\n"
                    for k, v in param.items():
                        # Handle different parameter types
                        if isinstance(v, str):
                            v_str = f"'{v}'"
                        elif isinstance(v, bool):
                            v_str = 'true' if v else 'false'
                        else:
                            v_str = str(v)
                        params_str += f"                '{k}': {v_str},\n"
                    params_str += "            },\n"
                else:
                    params_str += f"            '{param}',\n"
            params_str += "        ],\n"
            launch_content += params_str
        
        # Add remappings
        if node_config.get('remappings'):
            remaps_str = "        remappings=[\n"
            for remap in node_config['remappings']:
                src, dst = remap.split(':', 1)
                remaps_str += f"            ('{src}', '{dst}'),\n"
            remaps_str += "        ],\n"
            launch_content += remaps_str
        
        # Add environment variables
        if node_config.get('env'):
            env_str = "        env=[\n"
            for k, v in node_config['env'].items():
                env_str += f"            {{'{k}': '{v}'}},\n"
            env_str += "        ],\n"
            launch_content += env_str
        
        # Close node configuration
        launch_content += "    )\n\n"
        
        # Add additional nodes (like RViz) if specified in launch config
        additional_nodes = []
        if launch_config and hasattr(launch_config, 'node') and len(launch_config.node) > 1:
            for i, extra_node in enumerate(launch_config.node[1:], 1):
                node_var = f"extra_node_{i}"
                additional_nodes.append(node_var)
                
                launch_content += f"    # Additional node: {getattr(extra_node, 'name', 'unnamed')}\n"
                launch_content += f"    {node_var} = Node(\n"
                
                # Add basic node configuration
                if hasattr(extra_node, 'package'):
                    launch_content += f"        package='{extra_node.package}',\n"
                if hasattr(extra_node, 'executable'):
                    launch_content += f"        executable='{extra_node.executable}',\n"
                if hasattr(extra_node, 'name'):
                    launch_content += f"        name='{extra_node.name}',\n"
                if hasattr(extra_node, 'namespace'):
                    launch_content += f"        namespace='{extra_node.namespace}',\n"
                if hasattr(extra_node, 'output'):
                    launch_content += f"        output='{extra_node.output}',\n"
                
                # Handle parameters
                if hasattr(extra_node, 'parameters') and extra_node.parameters:
                    params_str = "        parameters=[\n"
                    for param in extra_node.parameters:
                        if isinstance(param, dict):
                            params_str += "            {\n"
                            for k, v in param.items():
                                if isinstance(v, str):
                                    v_str = f'"{v}"'
                                elif isinstance(v, bool):
                                    v_str = 'true' if v else 'false'
                                else:
                                    v_str = str(v)
                                params_str += f"                '{k}': {v_str},\n"
                            params_str += "            },\n"
                        else:
                            params_str += f"            '{param}',\n"
                    params_str += "        ],\n"
                    launch_content += params_str
                
                # Close the Node constructor
                launch_content += "    )\n\n"
        
        # Create launch description and add nodes
        launch_content += "    # Create the launch description and populate\n"
        launch_content += "    ld = LaunchDescription()\n\n"
        
        # Add the main node
        launch_content += f"    # Add the nodes to the launch description\n"
        launch_content += f"    ld.add_action({node_name}_node)\n"
        
        # Add any additional nodes
        for node_var in additional_nodes:
            launch_content += f"    ld.add_action({node_var})\n"
        
        # Return the launch description
        launch_content += "\n    return ld\n"
        
        # Write the launch file
        launch_path = self.output_dir / 'launch' / f"{node.name}.launch.py"
        launch_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(launch_path, 'w') as f:
            f.write(launch_content)
            
        return launch_path

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
        """Generate the main CMakeLists.txt file.
        
        Returns:
            Path to the generated CMakeLists.txt file
        """
        cmake_path = self.output_dir / 'CMakeLists.txt'
        
        # Generate base CMake content
        cmake_content = self._generate_cmake_content()
        
        # Add CUDA support if needed
        cmake_content = self._add_cuda_support(cmake_content)
        
        # Add ROS2 dependencies if needed
        cmake_content = self._add_ros2_dependencies(cmake_content)
        
        # Add node executables and component registration
        cmake_content = self._add_node_executables(cmake_content)
        
        # Add installation rules for include directories
        cmake_content.extend([
            "",
            "# Install include directories",
            "install(",
            "    DIRECTORY include/",
            "    DESTINATION include",
            "    FILES_MATCHING PATTERN \"*.hpp\"",
            ")",
            "",
            "# Install launch files",
            "install(",
            "    DIRECTORY launch/",
            "    DESTINATION share/${PROJECT_NAME}/launch",
            "    PATTERN \"*.launch.py\"",
            ")",
            "",
            "# Export dependencies",
            "ament_export_dependencies(",
            "    rclcpp",
            "    rclcpp_components",
            "    std_msgs",
            ")",
            "",
            "# Finalize package",
            "ament_package()"
        ])
        
        # Write the CMakeLists.txt file
        with open(cmake_path, 'w') as f:
            f.write('\n'.join(cmake_content) + '\n')
            
        return cmake_path

    def _generate_package_xml(self) -> Path:
        """Generate the package.xml file.
        
        Returns:
            Path to the generated package.xml file
        """
        package_path = self.output_dir / 'package.xml'
        
        # Start with basic package information
        content = [
            '<?xml version="1.0"?>',
            '<?xml-model href="http://download.ros.org/schema/package_format3.xsd" \
              schematypens="http://www.w3.org/2001/XMLSchema\" version="1.0"?>',
            f'<package format="3">',
            f'  <name>{self.config.project_name}</name>',
            '  <version>0.1.0</version>',
            '  <description>Auto-generated ROS 2 package</description>',
            '  <maintainer email="user@example.com">User</maintainer>',
            '  <license>Apache License 2.0</license>',
            '',
            '  <buildtool_depend>ament_cmake</buildtool_depend>',
            '  <buildtool_depend>rclcpp_components</buildtool_depend>',
            '',
            '  <depend>rclcpp</depend>',
            '  <depend>std_msgs</depend>',
            '',
            '  <test_depend>ament_lint_auto</test_depend>',
            '  <test_depend>ament_lint_common</test_depend>',
            '',
            '  <export>',
            '    <build_type>ament_cmake</build_type>',
            '  </export>',
            '</package>'
        ]
        
        with open(package_path, 'w') as f:
            f.write('\n'.join(content) + '\n')
            
        return package_path

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
            "set(CMAKE_CUDA_FLAGS \"${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -fPIC\")",
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
        """Generate CUDA kernel implementation files.
        
        Args:
            kernel: The CUDA kernel configuration
            
        Returns:
            List of paths to generated files (header and source)
        """
        generated_files = []
        
        # Create output directory for CUDA kernels
        output_dir = self.output_dir / 'include' / 'robodsl' / 'cuda'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate header file
        header_guard = f"ROBODSL_{kernel.name.upper()}_KERNEL_HPP"
        
        # Generate includes
        includes = [
            '#include <cuda_runtime.h>',
            '#include <cstdint>',
            ''
        ]
        
        if kernel.use_thrust:
            includes.extend([
                '#include <thrust/device_vector.h>',
                '#include <thrust/host_vector.h>',
                '#include <thrust/functional.h>',
                ''
            ])
        
        # Add custom includes
        for include in kernel.includes:
            includes.append(f'#include "{include}"')
        includes.append('')
        
        # Generate parameter declarations
        param_decls = []
        for param in kernel.parameters:
            param_type = param.type
            param_name = param.name
            param_mod = ''
            
            # Handle const and pointer qualifiers
            if param.is_const:
                param_mod += 'const '
            
            if param.is_pointer:
                param_mod += '*' if not param_mod else ' *'
            
            # Build parameter declaration
            decl = f"{param_type} {param_mod}{param_name}"
            
            # Add size expression for dynamic arrays
            if param.size_expr and param.is_pointer:
                decl += f"[{param.size_expr}]"
                
            param_decls.append(decl)
        
        # Generate function declaration
        kernel_decl = f"__global__ void {kernel.name}_kernel("
        kernel_decl += ', '.join(param_decls) + ')'
        
        # Generate header content
        header_content = f"""// Generated by RoboDSL - DO NOT EDIT

// CUDA Kernel: {kernel.name}

#ifndef {header_guard}
#define {header_guard}

{"\n".join(includes)}

namespace robodsl {{

/**
 * @brief CUDA kernel for {kernel.name}
 * 
 * @param grid_size Grid size for kernel launch
 * @param block_size Block size for kernel launch
 * @param shared_mem_bytes Shared memory size in bytes
 * @param args Arguments to pass to the kernel
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_{kernel.name}_kernel(
    dim3 grid_size,
    dim3 block_size,
    unsigned int shared_mem_bytes,
    {', '.join([f"{p.type} {p.name}" for p in kernel.parameters])});

}} // namespace robodsl

#endif // {header_guard}
"""
        
        # Write header file
        header_path = output_dir / f"{kernel.name}_kernel.cuh"
        with open(header_path, 'w') as f:
            f.write(header_content)
        generated_files.append(header_path)
        
        # Prepare kernel parameters with proper pointer types for outputs
        kernel_params = []
        for param in kernel.inputs:
            kernel_params.append(f"{param['type']} {param['name']}")
        for param in kernel.outputs:
            kernel_params.append(f"{param['type']}* {param['name']}")
            
        # Generate source file
        source_content = f"""// Generated by RoboDSL - DO NOT EDIT

#include "robodsl/cuda/{kernel.name}_kernel.cuh"

namespace robodsl {{

// Kernel implementation
__global__ void {kernel.name}_kernel(
    {', '.join(kernel_params)})
{{
    // TODO: Implement CUDA kernel
    // Example: int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
{kernel.code}
}}

// Kernel launcher
cudaError_t launch_{kernel.name}_kernel(
    dim3 grid_size,
    dim3 block_size,
    unsigned int shared_mem_bytes,
    {', '.join([f"{p['type']} {p['name']}" for p in kernel.inputs + kernel.outputs])})
{{
    // Launch the kernel
    {kernel.name}_kernel<<<grid_size, block_size, shared_mem_bytes>>>(
        {', '.join([p['name'] for p in kernel.inputs + kernel.outputs])});
        
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {{
        return err;
    }}
    
    // Synchronize to check for kernel execution errors
    return cudaDeviceSynchronize();
}}

}} // namespace robodsl
"""
        
        # Write source file
        source_path = output_dir.parent.parent / 'src' / 'cuda' / f"{kernel.name}_kernel.cu"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, 'w') as f:
            f.write(source_content)
        generated_files.append(source_path)
        
        return generated_files
        
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
        """Generate a C++ source file for a ROS2 node.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated source file
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Check if this node uses ROS2 features
        uses_ros2 = bool(node.publishers or node.subscribers or node.services or node.parameters)
        
        # Flags for advanced features
        has_timers = bool(node.timers)
        has_actions = bool(node.actions)
        use_lifecycle = node.lifecycle
        has_qos_settings = any(
            entity.qos is not None
            for entity in node.publishers + node.subscribers + node.services + node.actions
        )

        # Initialize includes
        includes = [
            f'#include "robodsl/{node.name}_node.hpp"',
            '#include <memory>',
            '#include <iostream>',
            ''
        ]
        
        # Add message includes with ROS2 conditional compilation
        if uses_ros2:
            msg_includes = set()
            for pub in node.publishers:
                msg_includes.add(f'#include "{pub.msg_type.replace(".", "/")}.hpp"')
            for sub in node.subscribers:
                msg_includes.add(f'#include "{sub.msg_type.replace(".", "/")}.hpp"')
            for srv in node.services:
                msg_includes.add(f'#include "{srv.srv_type.replace(".", "/")}.hpp"')
            
            if msg_includes:
                includes.append('#if ENABLE_ROS2')
                includes.extend(sorted(msg_includes))
                includes.append('#else')
                includes.append('// ROS2 message includes stubbed out for non-ROS2 builds')
                includes.append('#endif')
                includes.append('')
        
        # Initialize constructor
        if uses_ros2:
            base_ctor = 'LifecycleNode' if use_lifecycle else 'Node'
            constructor = f'''{class_name}::{class_name}() : {base_ctor}("{node.name}")
{{
    // Initialize parameters
'''
        else:
            constructor = f'''{class_name}::{class_name}()
{{
    // Initialize parameters
'''
        
        # Add parameter declarations
        if node.parameters:
            if uses_ros2:
                constructor += '    // Declare parameters\n'
                for name, type_info in node.parameters.items():
                    constructor += f'    this->declare_parameter("{name}", /* default value */);\n'
            else:
                constructor += '    // Parameters (stubbed out for non-ROS2 builds)\n'
                for name, type_info in node.parameters.items():
                    constructor += f'    // {name}_ = /* default value */;\n'
        
        # Register parameter callback
        if uses_ros2 and has_param_cb:
            constructor += '\n    // Register parameter change callback\n'
            constructor += f'    param_cb_handle_ = this->add_on_set_parameters_callback(\n        std::bind(&{class_name}::on_parameter_event, this, std::placeholders::_1));\n'

        # Create timers
        if uses_ros2 and has_timers:
            constructor += '\n    // Initialize timers\n'
            for tmr in node.timers:
                var_name = f"timer_{tmr['name']}_"
                period = tmr['period_ms']
                qos = self._validate_qos_config(tmr.get('qos'), f"timer {tmr['name']}")
                if qos:
                    qos_var = f"{var_name}_qos"
                    constructor += '    {\n'
                    constructor += f'      {self._generate_qos_config(qos, qos_var)}\n'
                    constructor += f'      {var_name} = this->create_wall_timer(std::chrono::milliseconds({period}), {qos_var}, [this]() {{ /* TODO: {tmr["name"]} callback */ }});\n'
                    constructor += '    }\n'
                else:
                    constructor += f'    {var_name} = this->create_wall_timer(std::chrono::milliseconds({period}), [this]() {{ /* TODO: {tmr["name"]} callback */ }});\n'

        # Create action servers
        if uses_ros2 and has_actions:
            constructor += '\n    // Initialize action servers\n'
            for act in node.actions:
                action_type = act.action
                action_name = act.name
                execute_cb = act.execute_callback
                goal_cb = act.goal_callback if hasattr(act, 'goal_callback') else 'handle_goal'
                cancel_cb = act.cancel_callback if hasattr(act, 'cancel_callback') else 'handle_cancel'
                qos = self._validate_qos_config(act.qos if hasattr(act, 'qos') else None, f"action {action_name}")
                
                # Add action message include
                self._add_action_include(action_type)
                
                # Generate action server code with QoS if specified
                if qos:
                    qos_var = f"{self._to_snake_case(action_name)}_qos"
                    qos_code = self._generate_qos_config(qos, qos_var)
                    constructor += '    {\n'
                    constructor += f'      {qos_code}\n'
                    
                    constructor += (
                        f'      rcl_action_server_options_t {self._to_snake_case(action_name)}_server_options = '
                        f'rcl_action_server_get_default_options();\n'
                        f'      {self._to_snake_case(action_name)}_server_options.result_timeout.nanoseconds = RCL_MS_TO_NS(1000);\n'
                        f'      {self._to_snake_case(action_name)}_server_options.goal_service_qos = {qos_var}.get_rmw_qos_profile();\n'
                        f'      {self._to_snake_case(action_name)}_server_options.cancel_service_qos = {qos_var}.get_rmw_qos_profile();\n'
                        f'      {self._to_snake_case(action_name)}_server_options.result_service_qos = {qos_var}.get_rmw_qos_profile();\n'
                        f'      {self._to_snake_case(action_name)}_server_options.cancel_subscription_qos = {qos_var}.get_rmw_qos_profile();\n'
                        f'      {self._to_snake_case(action_name)}_server_options.goal_event_qos = {qos_var}.get_rmw_qos_profile();\n'
                        f'      {self._to_snake_case(action_name)}_server_options.result_timeout_qos = {qos_var}.get_rmw_qos_profile();\n'
                        f'      {self._to_snake_case(action_name)}_ = '
                        f'rclcpp_action::create_server<{action_type}>('
                        f'this, "{action_name}", '
                        f'std::bind(&{class_name}::{execute_cb}, this, std::placeholders::_1), '
                        f'std::bind(&{class_name}::{goal_cb}, this, std::placeholders::_1), '
                        f'std::bind(&{class_name}::{cancel_cb}, this, std::placeholders::_1), '
                        f'{self._to_snake_case(action_name)}_server_options);\n'
                    )
                    constructor += '    }\n'
                else:
                    constructor += (
                        f'    {self._to_snake_case(action_name)}_ = '
                        f'rclcpp_action::create_server<{action_type}>('
                        f'this, "{action_name}", '
                        f'std::bind(&{class_name}::{execute_cb}, this, std::placeholders::_1), '
                        f'std::bind(&{class_name}::{goal_cb}, this, std::placeholders::_1), '
                        f'std::bind(&{class_name}::{cancel_cb}, this, std::placeholders::_1));\n'
                    )

        # Add publishers, subscribers, services with ROS2 guards
        if uses_ros2:
            if node.publishers:
                constructor += '\n    // Initialize publishers\n'
                for pub in node.publishers:
                    msg_type = pub.message
                    topic = pub.topic
                    qos = self._validate_qos_config(pub.qos, f"publisher {topic}")
                    
                    # Add message include
                    self._add_message_include(msg_type)
                    
                    # Generate QoS code if needed
                    qos_var = f"{self._to_snake_case(topic)}_qos"
                    qos_code = self._generate_qos_config(qos, qos_var) if qos else ''
                    
                    # Generate publisher code
                    publisher_var = f"{self._to_snake_case(topic)}_pub_"
                    publisher_code = f"{qos_var} if ({qos_var}.get_rmw_qos_profile().depth > 0) else 10" if qos else '10'
                    
                    constructor += f"    rclcpp::Publisher<{msg_type}>::{SharedPtr} {publisher_var};\n"
                    
                    if qos_code:
                        constructor += qos_code
                        
                    constructor += (
                        f"    {publisher_var} = this->create_publisher<{msg_type}>("
                        f"\"{topic}\", {publisher_code});\n"
                    )

            if node.subscribers:
                constructor += '\n    // Initialize subscribers\n'
                for sub in node.subscribers:
                    msg_type = sub.message
                    topic = sub.topic
                    callback = sub.callback
                    qos = self._validate_qos_config(sub.qos, f"subscriber {topic}")
                    
                    # Add message include
                    self._add_message_include(msg_type)
                    
                    # Generate QoS code if needed
                    qos_var = f"{self._to_snake_case(topic)}_qos"
                    qos_code = self._generate_qos_config(qos, qos_var) if qos else ''
                    
                    # Generate subscriber code
                    if qos_code:
                        constructor += qos_code
                        
                    constructor += (
                        f"    auto {self._to_snake_case(topic)}_sub_ = "
                        f"this->create_subscription<{msg_type}>("
                        f"\"{topic}\", "
                        f"{qos_var if qos else '10'}, "
                        f"std::bind(&{class_name}::{callback}, this, std::placeholders::_1));\n"
                    )

            if node.services:
                constructor += '\n    // Initialize services\n'
                for svc in node.services:
                    srv_type = svc.service
                    service_name = svc.name
                    callback = svc.callback
                    qos = self._validate_qos_config(svc.qos, f"service {service_name}")
                    
                    # Add service message include
                    self._add_service_include(srv_type)
                    
                    # Generate service code with QoS if specified
                    if qos:
                        qos_var = f"{self._to_snake_case(service_name)}_qos"
                        qos_code = self._generate_qos_config(qos, qos_var)
                        constructor += qos_code
                        
                        constructor += (
                            f"    rclcpp::ServiceOptions {self._to_snake_case(service_name)}_opts;\n"
                            f"    {self._to_snake_case(service_name)}_opts.qos({qos_var});\n"
                            f"    {self._to_snake_case(service_name)}_opts.callback_group(cb_group_);\n"
                            f"    auto {self._to_snake_case(service_name)}_ = "
                            f"this->create_service<{srv_type}>("
                            f"\"{service_name}\", "
                            f"std::bind(&{class_name}::{callback}, this, std::placeholders::_1, std::placeholders::_2), "
                            f"{self._to_snake_case(service_name)}_opts);\n"
                        )
                    else:
                        constructor += (
                            f"    auto {self._to_snake_case(service_name)}_ = "
                            f"this->create_service<{srv_type}>("
                            f"\"{service_name}\", "
                            f"std::bind(&{class_name}::{callback}, this, std::placeholders::_1, std::placeholders::_2), "
                            f"rmw_qos_profile_services_default, cb_group_.get());\n"
                        )

        constructor += '}'
        
        # Generate callback methods
        callbacks = []
        
        # Add callbacks with proper ROS2 message types and guards
        for sub in node.subscribers:
            topic = sub['topic']
            callback_name = f"{topic.lstrip('/').replace('/', '_')}_callback"
            msg_type = sub['msg_type'].replace('.', '::')
            
            if uses_ros2:
                callback = f'''#if ENABLE_ROS2
void {class_name}::{callback_name}(const {msg_type}::SharedPtr msg) const
{{
    // Process incoming message
    RCLCPP_DEBUG(this->get_logger(), "Received message on topic %s", "{topic}");
    
    // TODO: Implement message processing
    (void)msg;  // Avoid unused parameter warning
}}
#else
void {class_name}::{callback_name}(const void* /*msg*/) const
{{
    // Stub implementation for non-ROS2 builds
    std::cout << "[{node.name}] Received message on topic: {topic}" << std::endl;
    // TODO: Implement non-ROS2 message processing
}}
#endif'''
                callbacks.append(callback)
            else:
                callbacks.append(f'// Callback for {topic} stubbed out for non-ROS2 builds')
        
        # Generate service callbacks
        for srv in node.services:
            service_name = srv['service']
            srv_type = srv['srv_type'].replace('.', '::')
            callback_name = f"{service_name.lstrip('/').replace('/', '_')}_callback"
            
            if uses_ros2:
                callback = f'''#if ENABLE_ROS2
void {class_name}::{callback_name}(
    const std::shared_ptr<{srv_type}::Request> request,
    std::shared_ptr<{srv_type}::Response> response) const
{{
    // Process service request
    RCLCPP_DEBUG(this->get_logger(), "Received service call on %s", "{service_name}");
    
    // TODO: Implement service logic
    (void)request;  // Avoid unused parameter warning
    (void)response; // Avoid unused parameter warning
}}
#else
void {class_name}::{callback_name}(
    const void* /*request*/, void* /*response*/) const
{{
    // Stub implementation for non-ROS2 builds
    std::cout << "[{node.name}] Received service call: {service_name}" << std::endl;
    // TODO: Implement non-ROS2 service logic
}}
#endif'''
                callbacks.append(callback)
            else:
                callbacks.append(f'// Service {service_name} stubbed out for non-ROS2 builds')
        
        # Build the source content
        source_lines = [
            '// Generated by RoboDSL - DO NOT EDIT',
            '',
            f'#include "robodsl/{node.name}_node.hpp"',
            '#include <memory>',
            '#include <iostream>',
            ''
        ]
        
        # Add includes
        source_lines.extend(includes)
        
        # Add namespace and class implementation
        source_lines.extend([
            'namespace robodsl {',
            '',
            f'{constructor}',
            '',
            f'{class_name}::~{class_name}()',
            '{',
            '    // Cleanup resources if needed',
            '}',
            '',
            '// Implement interface methods',
            f'void {class_name}::initialize()',
            '{',
            '    // TODO: Initialize the node',
            '    #if !ENABLE_ROS2',
            f'    std::cout << "[{node.name}] Initializing (non-ROS2 mode)" << std::endl;',
            '    #endif',
            '}',
            '',
            f'void {class_name}::update()',
            '{',
            '    // TODO: Update the node state',
            '    #if !ENABLE_ROS2',
            '    // Non-ROS2 update logic here',
            '    #endif',
            '}',
            '',
            f'void {class_name}::cleanup()',
            '{',
            '    // TODO: Cleanup resources',
            '    #if !ENABLE_ROS2',
            f'    std::cout << "[{node.name}] Cleaning up (non-ROS2 mode)" << std::endl;',
            '    #endif',
            '}',
            ''
        ])
        
        # Add callbacks
        source_lines.extend(callbacks)
        
        # Generate action callbacks
        for act in node.actions:
            action_name = act['name']
            act_type = action_name.capitalize()
            if uses_ros2:
                callback_code = f'''#if ENABLE_ROS2
rclcpp_action::GoalResponse {class_name}::handle_goal_{action_name}(const rclcpp_action::GoalUUID & /*uuid*/, std::shared_ptr<const {act_type}::Goal> /*goal*/)
{{
    RCLCPP_INFO(this->get_logger(), "Received goal request for action {action_name}");
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}}

rclcpp_action::CancelResponse {class_name}::handle_cancel_{action_name}(const std::shared_ptr<rclcpp_action::ServerGoalHandle<{act_type}>> /*goal_handle*/)
{{
    RCLCPP_INFO(this->get_logger(), "Received cancel request for action {action_name}");
    return rclcpp_action::CancelResponse::ACCEPT;
}}

void {class_name}::handle_accepted_{action_name}(const std::shared_ptr<rclcpp_action::ServerGoalHandle<{act_type}>> goal_handle)
{{
    RCLCPP_INFO(this->get_logger(), "Goal accepted for action {action_name}");
    // Example CUDA off-load in detached thread
#if ENABLE_CUDA && defined(__CUDACC__)
    std::thread([this, goal_handle]() {{
        // TODO: copy goal to device, allocate buffers
        dim3 grid(1); dim3 block(256);
        // exampleKernel<<<grid, block>>>();
        cudaDeviceSynchronize();

        auto result = std::make_shared<{act_type}::Result>();
        // TODO: fill result fields
        goal_handle->succeed(result);
    }}).detach();
#else
    (void)goal_handle;
#endif
    (void)goal_handle;
    // TODO: start execution
}}
#else
void {class_name}::handle_goal_{action_name}(... ) {{}}
void {class_name}::handle_cancel_{action_name}(... ) {{}}
void {class_name}::handle_accepted_{action_name}(... ) {{}}
#endif'''
                source_lines.append(callback_code)
            else:
                callbacks.append(f'// Action callbacks for {action_name} stubbed for non-ROS2 builds')

        # Generate parameter callback definition
        if node.parameter_callbacks:
            if uses_ros2:
                param_cb_code = f'''#if ENABLE_ROS2
rcl_interfaces::msg::SetParametersResult {class_name}::on_parameter_event(const std::vector<rclcpp::Parameter> & params)
{{
    (void)params; // TODO: Inspect parameters and set result accordingly
    rcl_interfaces::msg::SetParametersResult result;
    result.successful = true;
    result.reason = "Parameters accepted";
    return result;
}}
#else
void {class_name}::on_parameter_event(const std::vector<void*>& /*params*/) {{}}
#endif'''
                source_lines.append(param_cb_code)
            else:
                source_lines.append(f'// Parameter callback stubbed for non-ROS2 builds')

        # Close namespace and add ROS2 registration
        source_lines.extend([
            '}  // namespace robodsl',
            ''
        ])
        
        if uses_ros2:
            source_lines.extend([
                '#if ENABLE_ROS2',
                f'RCLCPP_COMPONENTS_REGISTER_NODE(robodsl::{class_name})',
                '#endif',
                ''
            ])
        
        source_content = '\n'.join(source_lines)
        
        # Create output directory if it doesn't exist
        source_path = self.output_dir / 'src' / f"{node.name}_node.cpp"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the source file
        with open(source_path, 'w') as f:
            f.write(source_content)
            
        # Update the has_qos_settings flag for CMake generation
        node.has_qos_settings = has_qos_settings
            
        return source_path
    
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

