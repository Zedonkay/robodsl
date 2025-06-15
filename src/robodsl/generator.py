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
    
    def _generate_node_header(self, node: NodeConfig) -> str:
        """Generate a C++ header file for a ROS2 node.
        
        Args:
            node: The node configuration
            
        Returns:
            The generated header file content
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
        
        header_content += f"}};  // class {node.name.capitalize()}Node\n\n"
        header_content += "}}  // namespace robodsl\n"
        
        return header_content
    
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
        uses_ros2 = bool(node.publishers or node.subscribers or node.services or node.parameters)
        
        # Only generate launch file for ROS2 nodes
        if not uses_ros2:
            return None
            
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Generate launch file content
        package_name = self.config.project_name
        exec_name = f"{node.name}_node"
        node_name = node.name
        
        launch_content = f"""# Generated by RoboDSL - DO NOT EDIT

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{package_name}',
            executable='{exec_name}',
            name='{node_name}',
            output='screen',
            parameters=[
                # Add parameters here if needed
            ]
        )
    ])""".format(
            package_name=package_name,
            exec_name=exec_name,
            node_name=node_name
        )
        
        # Create output directory if it doesn't exist
        launch_path = self.output_dir / 'launch' / f"{node.name}.launch.py"
        launch_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the launch file
        with open(launch_path, 'w') as f:
            f.write(launch_content)
            
        return launch_path
        
    def _generate_package_xml(self) -> Optional[Path]:
        """Generate the package.xml file for ROS2 packages.
        
        Returns:
            Path to the generated package.xml file, or None if not needed
        """
        # Only generate package.xml if this is a ROS2 project
        uses_ros2 = any(
            node.publishers or node.subscribers or node.services or node.parameters
            for node in self.config.nodes
        )
        
        if not uses_ros2:
            return None
            
        # Generate package.xml content
        package_xml = f'''<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:noNamespaceSchemaLocation="http://download.ros.org/schema/package_format3.xsd">
<package format="3">
  <name>{self.config.project_name}</name>
  <version>0.1.0</version>
  <description>Generated by RoboDSL</description>
  <maintainer email="user@example.com">User</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>
  <buildtool_depend>rclcpp</buildtool_depend>
  <buildtool_depend>rclcpp_components</buildtool_depend>
  
  <depend>rclcpp</depend>
  <depend>rclcpp_components</depend>
  <depend>std_msgs</depend>
  
  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
'''
        
        # Create output directory if it doesn't exist
        package_path = self.output_dir / 'package.xml'
        package_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the package.xml file
        with open(package_path, 'w') as f:
            f.write(package_xml)
            
        return package_path
        
    def _generate_cmakelists(self) -> Path:
        """Generate the CMakeLists.txt file for the project.
        
        Returns:
            Path to the generated CMakeLists.txt file
        """
        # Check if this project uses ROS2 features
        uses_ros2 = any(
            node.publishers or node.subscribers or node.services or node.parameters
            for node in self.config.nodes
        )
        
        # Check if this project has CUDA kernels
        has_cuda_kernels = bool(self.config.cuda_kernels)
        
        # Generate CMake content
        cmake_content = f"""# Generated by RoboDSL - DO NOT EDIT

cmake_minimum_required(VERSION 3.10)
project({self.config.project_name} VERSION 0.1.0)

# Default to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options for optional features
option(ENABLE_ROS2 "Enable ROS2 support" {'ON' if uses_ros2 else 'OFF'})
option(ENABLE_CUDA "Enable CUDA support" {'ON' if has_cuda_kernels else 'OFF'})

# Find required packages
find_package(Threads REQUIRED)

# Find ROS2 packages if enabled
if(ENABLE_ROS2)
    # Find ament_cmake if not already found
    if(NOT ament_cmake_FOUND)
        find_package(ament_cmake REQUIRED)
    endif()
    
    # Find ROS2 packages
    find_package(rclcpp REQUIRED)
    find_package(rclcpp_components REQUIRED)
    find_package(std_msgs REQUIRED)
    
    # Find message/service dependencies
    # TODO: Add any additional ROS2 message/service dependencies here
    
    # Include directories
    include_directories(
        ${{CMAKE_CURRENT_SOURCE_DIR}}/include
        ${{rclcpp_INCLUDE_DIRS}}
        ${{rclcpp_components_INCLUDE_DIRS}}
    )
    
    # Add compile definitions
    add_compile_definitions(ENABLE_ROS2=1)
else()
    # Include directories for non-ROS2 build
    include_directories(
        ${{CMAKE_CURRENT_SOURCE_DIR}}/include
    )
    
    # Add compile definitions
    add_compile_definitions(ENABLE_ROS2=0)
    
    # Add stubs for ROS2 types if needed
    # TODO: Add any necessary stubs for ROS2 types
endif()

# Handle CUDA if enabled
if(ENABLE_CUDA)
    enable_language(CUDA)
    
    # Set CUDA architecture
    set(CUDA_ARCHITECTURES "native")
    
    # CUDA compile options
    set(CUDA_NVCC_FLAGS "${{CUDA_NVCC_FLAGS}} -std=c++17 --expt-relaxed-constexpr")
    
    # Find CUDA
    find_package(CUDA REQUIRED)
    
    # Include CUDA directories
    include_directories(
        ${{CUDA_INCLUDE_DIRS}}
    )
    
    # Add CUDA compile definitions
    add_compile_definitions(WITH_CUDA=1)
    
    # Find Thrust if needed
    if(EXISTS "${{CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}}/thrust")
        add_compile_definitions(WITH_THRUST=1)
    else()
        add_compile_definitions(WITH_THRUST=0)
    endif()
else()
    # Add stubs for CUDA if needed
    add_compile_definitions(WITH_CUDA=0)
    add_compile_definitions(WITH_THRUST=0)
    
    # TODO: Add any necessary stubs for CUDA types/functions
endif()

# Add library
add_library(${{PROJECT_NAME}}_lib
    # Add source files here
    # src/example.cpp
)

target_link_libraries(${{PROJECT_NAME}}_lib
    ${{CMAKE_THREAD_LIBS_INIT}}
)

if(ENABLE_ROS2)
    ament_target_dependencies(${{PROJECT_NAME}}_lib
        rclcpp
        rclcpp_components
        std_msgs
        # Add other ROS2 dependencies here
    )
endif()

# Add CUDA kernel library if needed
if(ENABLE_CUDA AND ENABLE_CUDA_SUPPORT AND TARGET cuda_kernels)
    add_dependencies(${{PROJECT_NAME}}_lib cuda_kernels)
    target_link_libraries(${{PROJECT_NAME}}_lib cuda_kernels)
endif()

# Add executables for each node
"""

        # Add executable for each node
        for node in self.config.nodes:
            node_name = node.name
            exec_name = f"{node_name}_node"
            
            cmake_content += f"""
# {node_name} node
target_sources(${{PROJECT_NAME}}_lib PRIVATE
    src/{node_name}_node.cpp
)

# Add include directory for this node
target_include_directories(${{PROJECT_NAME}}_lib PRIVATE
    ${{CMAKE_CURRENT_SOURCE_DIR}}/include/robodsl
)

# Create executable
add_executable({exec_name} src/{node_name}_node.cpp)
target_link_libraries({exec_name} ${{PROJECT_NAME}}_lib)

# Install targets
install(TARGETS {exec_name}
    RUNTIME DESTINATION lib/${{PROJECT_NAME}}
    LIBRARY DESTINATION lib/${{PROJECT_NAME}}
    ARCHIVE DESTINATION lib/${{PROJECT_NAME}}
)"""

        # Add ROS2-specific install rules if needed
        if uses_ros2:
            cmake_content += """
# Install Python modules for the package
install(DIRECTORY
    launch
    DESTINATION share/${PROJECT_NAME}/
)

# Export dependencies
if(ENABLE_ROS2)
    ament_export_dependencies(
        rclcpp
        rclcpp_components
        std_msgs
        # Add other ROS2 dependencies here
    )
    
    ament_export_include_directories(include)
    ament_export_libraries(${PROJECT_NAME}_lib)
    
    ament_package()
endif()
"""
        else:
            cmake_content += """
# Install include files
install(DIRECTORY
    include/
    DESTINATION include
)
"""

        # Create output directory if it doesn't exist
        cmake_path = self.output_dir / 'CMakeLists.txt'
        cmake_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        
        # Generate input/output parameter declarations
        input_params = []
        for param in kernel.inputs:
            param_type = param['type']
            param_name = param['name']
            input_params.append(f"{param_type} {param_name}")
        
        output_params = []
        for param in kernel.outputs:
            param_type = param['type']
            param_name = param['name']
            output_params.append(f"{param_type}* {param_name}")
        
        # Generate function declaration
        kernel_decl = f"__global__ void {kernel.name}_kernel("
        kernel_decl += ', '.join(input_params + output_params) + ')'
        
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
    {', '.join([f"{p['type']} {p['name']}" for p in kernel.inputs + kernel.outputs])});

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
                msg_includes.add(f'#include "{pub["msg_type"].replace(".", "/")}.hpp"')
            for sub in node.subscribers:
                msg_includes.add(f'#include "{sub["msg_type"].replace(".", "/")}.hpp"')
            for srv in node.services:
                msg_includes.add(f'#include "{srv["srv_type"].replace(".", "/")}.hpp"')
            
            if msg_includes:
                includes.append('#if ENABLE_ROS2')
                includes.extend(sorted(msg_includes))
                includes.append('#else')
                includes.append('// ROS2 message includes stubbed out for non-ROS2 builds')
                includes.append('#endif')
                includes.append('')
        
        # Initialize constructor
        if uses_ros2:
            constructor = f'''{class_name}::{class_name}() : Node("{node.name}")
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
        
        # Add publishers, subscribers, and services with ROS2 guards
        if uses_ros2:
            if node.publishers:
                constructor += '\n    // Initialize publishers\n'
                for pub in node.publishers:
                    topic = pub['topic']
                    msg_type = pub['msg_type'].replace('.', '::')
                    var_name = f"{topic.lstrip('/').replace('/', '_')}_pub_"
                    
                    decl = f'{var_name} = this->create_publisher<{msg_type}>("{topic}", 10);'
                    constructor += f'    {decl}\n'
            
            if node.subscribers:
                constructor += '\n    // Initialize subscribers\n'
                for sub in node.subscribers:
                    topic = sub['topic']
                    msg_type = sub['msg_type'].replace('.', '::')
                    callback_name = f"{topic.lstrip('/').replace('/', '_')}_callback"
                    
                    decl = f'this->create_subscription<{msg_type}>(' \
                          f'"{topic}", 10, ' \
                          f'std::bind(&{class_name}::{callback_name}, this, std::placeholders::_1));'
                    constructor += f'    {decl}\n'
            
            if node.services:
                constructor += '\n    // Initialize services\n'
                for srv in node.services:
                    service_name = srv['service']
                    srv_type = srv['srv_type'].replace('.', '::')
                    callback_name = f"{service_name.lstrip('/').replace('/', '_')}_callback"
                    
                    decl = f'{service_name.lstrip("/").replace("/", "_")}_srv_ = ' \
                          f'this->create_service<{srv_type}>(' \
                          f'"{service_name}", ' \
                          f'std::bind(&{class_name}::{callback_name}, this, ' \
                          f'std::placeholders::_1, std::placeholders::_2));'
                    constructor += f'    {decl}\n'
        
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
