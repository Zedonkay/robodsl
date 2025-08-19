"""C++ Node Generator for RoboDSL.

This generator creates C++ header (.hpp) and source (.cpp) files for ROS2 nodes,
including virtual methods for CUDA kernels defined within the node.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST, NodeNode, KernelNode


class CppNodeGenerator(BaseGenerator):
    """Generates C++ node files with CUDA kernel virtual methods."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate C++ node files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        self.ast = ast  # Store AST for use in _prepare_node_context
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include').mkdir(parents=True, exist_ok=True)
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
        
        # Use the same subdirectory structure as source files
        subdir = self._get_node_subdirectory(node)
        
        try:
            content = self.render_template('node.hpp.jinja2', context)
            header_path = self.get_output_path('include', subdir, f'{node.name}_node.hpp')
            return self.write_file(header_path, content)
        except Exception as e:
            print(f"Template error for node {node.name}: {e}")
            # Fallback to simple header
            content = f"// Generated header for {node.name}\n"
            header_path = self.get_output_path('include', subdir, f'{node.name}_node.hpp')
            return self.write_file(header_path, content)
    
    def _generate_node_source(self, node: NodeNode) -> Path:
        """Generate a C++ source file for a ROS2 node."""
        context = self._prepare_node_context(node)
        
        # Determine subdirectory structure based on node name and type
        # Organize nodes into subdirectories for better structure
        subdir = self._get_node_subdirectory(node)
        
        try:
            content = self.render_template('node.cpp.jinja2', context)
            source_path = self.get_output_path('src', subdir, f'{node.name}_node.cpp')
            return self.write_file(source_path, content)
        except Exception as e:
            print(f"Template error for node {node.name}: {e}")
            # Fallback to simple source
            content = f"// Generated source for {node.name}\n"
            source_path = self.get_output_path('src', subdir, f'{node.name}_node.cpp')
            return self.write_file(source_path, content)
    
    def _get_node_subdirectory(self, node: NodeNode) -> str:
        """Determine the appropriate subdirectory for a node based on its name and content."""
        # For subnodes with dots, use the existing logic
        if '.' in node.name:
            parts = node.name.split('.')
            if len(parts) > 1:
                return '/'.join(parts[:-1])
        
        # For regular nodes, organize by type/function
        node_name = node.name.lower()
        
        # Main/control nodes
        if 'main' in node_name:
            return 'nodes/main'
        # Perception/vision nodes
        elif 'perception' in node_name or 'vision' in node_name or 'camera' in node_name:
            return 'nodes/perception'
        # Navigation/movement nodes
        elif 'navigation' in node_name or 'movement' in node_name or 'drive' in node_name:
            return 'nodes/navigation'
        # Safety/monitoring nodes
        elif 'safety' in node_name or 'monitor' in node_name or 'emergency' in node_name:
            return 'nodes/safety'
        # CPP nodes (from package definitions)
        elif 'cpp' in node_name or node_name.startswith('robot_'):
            return 'nodes'
        # Default to nodes directory
        else:
            return 'nodes'
    
    def _camel_to_snake(self, name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re
        # Insert underscore before capital letters (except the first)
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        # Insert underscore before capital letters that follow lowercase
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
    
    def _ros_type_to_cpp(self, ros_type: str) -> str:
        """Convert ROS message type to C++ type."""
        if ros_type.startswith('std_msgs/msg/'):
            return f"std_msgs::msg::{ros_type.split('/')[-1]}"
        elif ros_type.startswith('sensor_msgs/msg/'):
            return f"sensor_msgs::msg::{ros_type.split('/')[-1]}"
        elif ros_type.startswith('geometry_msgs/msg/'):
            return f"geometry_msgs::msg::{ros_type.split('/')[-1]}"
        elif ros_type.startswith('nav_msgs/msg/'):
            return f"nav_msgs::msg::{ros_type.split('/')[-1]}"
        elif ros_type.startswith('std_srvs/srv/'):
            return f"std_srvs::srv::{ros_type.split('/')[-1]}"
        else:
            return ros_type
    
    def _ros_type_to_include(self, ros_type: str) -> str:
        """Convert ROS message type to include statement."""
        parts = ros_type.split('/')
        if len(parts) == 3:
            package, msg_or_srv, type_name = parts
            # Convert the type name from CamelCase to snake_case
            snake_case_name = self._camel_to_snake(type_name)
            return f"#include <{package}/{msg_or_srv}/{snake_case_name}.hpp>"
        else:
            return f"// Custom message: {ros_type}"
    
    def _prepare_node_context(self, node: NodeNode) -> Dict[str, Any]:
        """Prepare context for node template rendering."""
        # Determine if this is a lifecycle node
        is_lifecycle = getattr(node, 'lifecycle', False) or 'lifecycle' in node.name.lower()
        
        # Determine base class
        if is_lifecycle:
            base_class = "rclcpp_lifecycle::LifecycleNode"
        else:
            base_class = "rclcpp::Node"
        
        # Class name
        class_name = f"{node.name.capitalize()}Node"
        
        # Include guard
        include_guard = f"{node.name.upper()}_NODE_HPP"
        
        # Include path for source file
        subdir = self._get_node_subdirectory(node)
        if subdir:
            if subdir.startswith('nodes/'):
                include_path = f"{subdir}/{node.name}_node.hpp"
            elif subdir == 'nodes':
                include_path = f"nodes/{node.name}_node.hpp"
            else:
                include_path = f"nodes/{subdir}/{node.name}_node.hpp"
        else:
            include_path = f"nodes/{node.name}_node.hpp"
        
        # Namespace
        namespace = getattr(node, 'namespace', None)
        if namespace:
            if hasattr(namespace, 'namespace'):  # NamespaceNode object
                namespace_str = namespace.namespace
                if namespace_str.startswith('/'):
                    namespace = namespace_str[1:].replace('/', '_')
                else:
                    namespace = namespace_str
            elif hasattr(namespace, 'startswith'):  # String object
                if namespace.startswith('/'):
                    namespace = namespace[1:].replace('/', '_')
            else:
                # Fallback
                namespace_str = str(namespace)
                if namespace_str.startswith('/'):
                    namespace = namespace_str[1:].replace('/', '_')
                else:
                    namespace = namespace_str
        
        # Collect all message types
        all_msg_types = set()
        
        # Publishers
        publishers = []
        for pub in getattr(node, 'publishers', []):
            msg_type = self._ros_type_to_cpp(pub.msg_type)
            all_msg_types.add(pub.msg_type)
            # Extract name from topic (last part after /)
            pub_name = pub.topic.split('/')[-1] if '/' in pub.topic else pub.topic
            publishers.append({
                'name': pub_name,
                'topic': pub.topic,
                'msg_type': msg_type,
                'qos': getattr(pub, 'qos', {})
            })
        
        # Subscribers
        subscribers = []
        for sub in getattr(node, 'subscribers', []):
            msg_type = self._ros_type_to_cpp(sub.msg_type)
            all_msg_types.add(sub.msg_type)
            # Extract name from topic (last part after /)
            sub_name = sub.topic.split('/')[-1] if '/' in sub.topic else sub.topic
            callback_name = f"on_{sub_name}"
            subscribers.append({
                'name': sub_name,
                'topic': sub.topic,
                'msg_type': msg_type,
                'callback_name': callback_name,
                'qos': getattr(sub, 'qos', {})
            })
        
        # Services
        services = []
        for srv in getattr(node, 'services', []):
            srv_type = self._ros_type_to_cpp(srv.srv_type)
            all_msg_types.add(srv.srv_type)
            callback_name = f"on_{srv.service.split("/")[-1] if "/" in srv.service else srv.service}"
            services.append({
                'name': srv.service.split("/")[-1] if "/" in srv.service else srv.service,
                'service': srv.service,
                'srv_type': srv_type,
                'callback_name': callback_name
            })
        
        # Actions
        actions = []
        for action in getattr(node, 'actions', []):
            action_type = self._ros_type_to_cpp(action.action_type)
            all_msg_types.add(action.action_type)
            actions.append({
                'name': action.name,
                'topic': action.name,
                'action_type': action_type
            })
        
        # Timers
        timers = []
        for timer in getattr(node, 'timers', []):
            callback_name = f"on_{timer.name}"
            timers.append({
                'name': timer.name,
                'period': timer.period,
                'callback_name': callback_name
            })
        
        # Parameters
        parameters = []
        for param in getattr(node, 'parameters', []):
            # Convert parameter type to template type
            template_type = self._convert_param_type_to_template(param.type)
            # Extract the actual value from ValueNode
            default_value = None
            if hasattr(param, 'value') and param.value is not None:
                if hasattr(param.value, 'value'):  # ValueNode object
                    default_value = param.value.value
                else:
                    default_value = param.value
            # Format the value for C++ code
            formatted_value = self._format_parameter_value(default_value, param.type)
            parameters.append({
                'name': param.name,
                'type': param.type,
                'template_type': template_type,
                'default_value': formatted_value
            })
        
        # CUDA kernels
        cuda_kernels = []
        for kernel in getattr(node, 'cuda_kernels', []):
            kernel_context = self._prepare_kernel_context(kernel)
            cuda_kernels.append(kernel_context)
        
        # C++ methods
        cpp_methods = []
        for method in getattr(node, 'methods', []):
            cpp_methods.append({
                'name': method.name,
                'inputs': method.inputs,
                'outputs': method.outputs,
                'code': getattr(method, 'code', '')
            })
        
        # Global C++ code
        global_cpp_code = getattr(self.ast, 'global_cpp_code', [])
        
        # Check if OpenCV is needed
        opencv_needed = any(
            'cv_bridge' in str(code) or 'cv::' in str(code) or 'opencv' in str(code)
            for code in global_cpp_code
        )
        
        # Generate includes
        includes = []
        for msg_type in sorted(all_msg_types):
            include = self._ros_type_to_include(msg_type)
            if not include.startswith('//'):
                includes.append(include)
        
        # Additional ROS2 includes
        ros2_includes = []
        if actions:
            ros2_includes.append("#include <rclcpp_action/rclcpp_action.hpp>")
        if services:
            ros2_includes.append("#include <rclcpp/rclcpp.hpp>")
        
        # Custom types
        custom_types = []
        if any('DetectionResult' in str(code) for code in global_cpp_code):
            custom_types.append({
                'name': 'DetectionResult',
                'fields': [
                    {'type': 'uint32_t', 'name': 'id'},
                    {'type': 'std::string', 'name': 'label'},
                    {'type': 'float64', 'name': 'confidence'},
                    {'type': 'float64', 'name': 'bbox_x'},
                    {'type': 'float64', 'name': 'bbox_y'},
                    {'type': 'float64', 'name': 'bbox_width'},
                    {'type': 'float64', 'name': 'bbox_height'}
                ]
            })
        
        if cuda_kernels:
            custom_types.append({
                'name': 'CudaParams',
                'fields': [
                    {'type': 'int', 'name': 'device_id'},
                    {'type': 'bool', 'name': 'enable_processing'}
                ]
            })
        
        return {
            'node_name': node.name,
            'class_name': class_name,
            'base_class': base_class,
            'include_guard': include_guard,
            'include_path': include_path,
            'namespace': namespace,
            'is_lifecycle': is_lifecycle,
            'publishers': publishers,
            'subscribers': subscribers,
            'services': services,
            'actions': actions,
            'timers': timers,
            'parameters': parameters,
            'cuda_kernels': cuda_kernels,
            'cpp_methods': cpp_methods,
            'global_cpp_code': global_cpp_code,
            'includes': includes,
            'ros2_includes': ros2_includes,
            'custom_types': custom_types,
            'opencv_needed': opencv_needed
        }
    
    def _convert_param_type_to_template(self, param_type: str) -> str:
        """Convert parameter type to ROS2 template type."""
        if param_type == 'int':
            return 'int'
        elif param_type == 'double':
            return 'double'
        elif param_type == 'bool':
            return 'bool'
        elif param_type == 'string':
            return 'std::string'
        elif param_type.startswith('std::vector<int>'):
            return 'std::vector<int>'
        elif param_type.startswith('std::vector<double>'):
            return 'std::vector<double>'
        elif param_type.startswith('std::vector<std::string>'):
            return 'std::vector<std::string>'
        elif param_type.startswith('std::map'):
            # Convert map to string for ROS2 compatibility
            return 'std::string'
        else:
            return 'std::string'  # Default fallback
    
    def _format_parameter_value(self, value, param_type: str) -> str:
        """Format parameter value for C++ code."""
        if value is None:
            return "0" if param_type == 'int' else "0.0" if param_type == 'double' else "false" if param_type == 'bool' else '""'
        
        if param_type == 'string':
            if isinstance(value, str):
                return f'"{value}"'
            else:
                return f'"{str(value)}"'
        elif param_type == 'bool':
            if isinstance(value, bool):
                return str(value).lower()
            else:
                return str(bool(value)).lower()
        elif param_type == 'int':
            return str(int(value))
        elif param_type == 'double':
            return str(float(value))
        else:
            # For complex types like lists and dicts, convert to string
            return f'"{str(value)}"'
    
    def _prepare_kernel_context(self, kernel: KernelNode) -> Dict[str, Any]:
        """Prepare context for CUDA kernel."""
        # Generate struct name
        struct_name = f"{kernel.name.capitalize()}Parameters"
        
        # Generate parameters
        parameters = []
        for param in getattr(kernel, 'parameters', []):
            parameters.append({
                'type': param.type,
                'name': param.name
            })
        
        # Generate member variables
        members = []
        for param in getattr(kernel, 'parameters', []):
            if param.type.endswith('*'):
                members.append({
                    'type': param.type,
                    'name': param.name
                })
        
        return {
            'name': kernel.name,
            'struct_name': struct_name,
            'parameters': parameters,
            'members': members
        } 