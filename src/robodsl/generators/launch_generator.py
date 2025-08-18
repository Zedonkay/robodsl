"""Launch Generator for RoboDSL.

This generator creates ROS2 launch files for the generated nodes.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST


class LaunchGenerator(BaseGenerator):
    """Generates ROS2 launch files."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate launch files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Create launch directory
        launch_dir = self.output_dir / 'launch'
        launch_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate main launch file
        main_launch_path = self._generate_main_launch(ast)
        generated_files.append(main_launch_path)
        
        # Generate individual node launch files
        for node in ast.nodes:
            node_launch_path = self._generate_node_launch(node)
            generated_files.append(node_launch_path)
        
        return generated_files
    
    def _generate_main_launch(self, ast: RoboDSLAST) -> Path:
        """Generate the main launch file that launches all nodes."""
        context = self._prepare_main_launch_context(ast)
        
        try:
            content = self.render_template('launch/main_launch.py.jinja2', context)
            launch_path = self.output_dir / 'launch' / 'main_launch.py'
            return self.write_file(launch_path, content)
        except Exception as e:
            print(f"Template error for main launch: {e}")
            # Fallback to simple launch
            content = self._generate_fallback_main_launch(ast)
            launch_path = self.output_dir / 'launch' / 'main_launch.py'
            return self.write_file(launch_path, content)
    
    def _generate_node_launch(self, node) -> Path:
        """Generate a launch file for a single node."""
        context = self._prepare_node_launch_context(node)
        
        try:
            content = self.render_template('launch/node_launch.py.jinja2', context)
            launch_path = self.output_dir / 'launch' / f'{node.name}_launch.py'
            return self.write_file(launch_path, content)
        except Exception as e:
            print(f"Template error for node {node.name} launch: {e}")
            # Fallback to simple launch
            content = self._generate_fallback_node_launch(node)
            launch_path = self.output_dir / 'launch' / f'{node.name}_launch.py'
            return self.write_file(launch_path, content)
    
    def _prepare_main_launch_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for main launch template."""
        nodes = []
        
        for node in ast.nodes:
            node_info = self._prepare_node_launch_info(node)
            nodes.append(node_info)
        
        return {
            'nodes': nodes,
            'package_name': 'robodsl_package'
        }
    
    def _prepare_node_launch_context(self, node) -> Dict[str, Any]:
        """Prepare context for node launch template."""
        node_info = self._prepare_node_launch_info(node)
        
        return {
            'node': node_info,
            'package_name': 'robodsl_package'
        }
    
    def _prepare_node_launch_info(self, node) -> Dict[str, Any]:
        """Prepare node information for launch file generation."""
        # Get node parameters
        parameters = []
        for param in getattr(node, 'parameters', []):
            param_value = self._format_parameter_value(param)
            parameters.append({
                'name': param.name,
                'value': param_value
            })
        
        # Add common parameters
        parameters.append({
            'name': 'use_sim_time',
            'value': 'LaunchConfiguration("use_sim_time")'
        })
        
        return {
            'name': node.name,
            'executable': f'{node.name}_node',
            'namespace': f'/{node.name}',
            'parameters': parameters
        }
    
    def _format_parameter_value(self, param) -> str:
        """Format parameter value for launch file."""
        param_type = getattr(param, 'type', 'string')
        default_value = getattr(param, 'default_value', None)
        
        if param_type == 'string':
            if default_value is None:
                return '""'
            return f'"{default_value}"'
        elif param_type == 'int':
            if default_value is None:
                return '0'
            return str(default_value)
        elif param_type == 'double':
            if default_value is None:
                return '0.0'
            return str(float(default_value))
        elif param_type == 'bool':
            if default_value is None:
                return 'False'
            return 'True' if default_value else 'False'
        elif param_type.startswith('std::vector<int>'):
            if default_value is None:
                return '[]'
            if isinstance(default_value, list):
                return str(default_value)
            return '[]'
        elif param_type.startswith('std::vector<double>'):
            if default_value is None:
                return '[]'
            if isinstance(default_value, list):
                return str([float(x) for x in default_value])
            return '[]'
        elif param_type.startswith('std::vector<std::string>'):
            if default_value is None:
                return '[]'
            if isinstance(default_value, list):
                return str([f'"{x}"' for x in default_value])
            return '[]'
        elif param_type.startswith('std::map'):
            if default_value is None:
                return '{}'
            if isinstance(default_value, dict):
                formatted_dict = {}
                for k, v in default_value.items():
                    if isinstance(v, str):
                        formatted_dict[k] = f'"{v}"'
                    else:
                        formatted_dict[k] = v
                return str(formatted_dict)
            return '{}'
        else:
            # Default to string
            if default_value is None:
                return '""'
            return f'"{str(default_value)}"'
    
    def _generate_fallback_main_launch(self, ast: RoboDSLAST) -> str:
        """Generate a fallback main launch file."""
        nodes_content = []
        
        for node in ast.nodes:
            node_content = f"""    # {node.name} node
    {node.name}_node = Node(
        package='robodsl_package',
        executable='{node.name}_node',
        name='{node.name}_node',
        namespace='/{node.name}',
        output='screen',
        parameters=[
            {{'use_sim_time': LaunchConfiguration('use_sim_time')}},
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )
    nodes.append({node.name}_node)"""
            nodes_content.append(node_content)
        
        return f"""#!/usr/bin/env python3
\"\"\"
Launch all nodes from RoboDSL specification

This file was auto-generated by RoboDSL.
\"\"\"

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    launch_arguments = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level for all nodes'
        ),
    ]
    
    # Create nodes
    nodes = []
{chr(10).join(nodes_content)}

    return LaunchDescription(launch_arguments + nodes)"""
    
    def _generate_fallback_node_launch(self, node) -> str:
        """Generate a fallback node launch file."""
        return f"""#!/usr/bin/env python3
\"\"\"
Launch {node.name} node

This file was auto-generated by RoboDSL.
\"\"\"

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    # Declare launch arguments
    launch_arguments = [
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time'
        ),
        DeclareLaunchArgument(
            'log_level',
            default_value='info',
            description='Log level for the node'
        ),
    ]
    
    # Create node
    {node.name}_node = Node(
        package='robodsl_package',
        executable='{node.name}_node',
        name='{node.name}_node',
        namespace='/{node.name}',
        output='screen',
        parameters=[
            {{'use_sim_time': LaunchConfiguration('use_sim_time')}},
        ],
        arguments=['--ros-args', '--log-level', LaunchConfiguration('log_level')],
    )

    return LaunchDescription(launch_arguments + [{node.name}_node])""" 