"""Launch Generator for RoboDSL.

This generator creates ROS2 launch files for running nodes.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..ast import RoboDSLAST, NodeNode


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
        (self.output_dir / 'launch').mkdir(parents=True, exist_ok=True)
        
        # Generate main launch file
        main_launch_path = self._generate_main_launch(ast)
        generated_files.append(main_launch_path)
        
        # Generate individual node launch files
        for node in ast.nodes:
            node_launch_path = self._generate_node_launch(node)
            generated_files.append(node_launch_path)
        
        return generated_files
    
    def _generate_main_launch(self, ast: RoboDSLAST) -> Path:
        """Generate the main launch file that starts all nodes."""
        context = self._prepare_main_launch_context(ast)
        
        try:
            content = self.render_template('main_launch.py.jinja2', context)
            launch_path = self.get_output_path('launch', 'main_launch.py')
            return self.write_file(launch_path, content)
        except Exception as e:
            print(f"Template error for main launch: {e}")
            # Fallback to simple launch file
            content = self._generate_fallback_main_launch(ast)
            launch_path = self.get_output_path('launch', 'main_launch.py')
            return self.write_file(launch_path, content)
    
    def _generate_node_launch(self, node: NodeNode) -> Path:
        """Generate a launch file for a single node."""
        context = self._prepare_node_launch_context(node)
        
        try:
            content = self.render_template('node.launch.py.jinja2', context)
            launch_path = self.get_output_path('launch', f'{node.name}_launch.py')
            return self.write_file(launch_path, content)
        except Exception as e:
            print(f"Template error for node {node.name} launch: {e}")
            # Fallback to simple node launch
            content = self._generate_fallback_node_launch(node)
            launch_path = self.get_output_path('launch', f'{node.name}_launch.py')
            return self.write_file(launch_path, content)
    
    def _prepare_main_launch_context(self, ast: RoboDSLAST) -> Dict[str, Any]:
        """Prepare context for main launch template rendering."""
        # Determine package name
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        # Prepare node configurations
        nodes = []
        for node in ast.nodes:
            node_config = {
                'name': node.name,
                'executable': f'{node.name}_node',
                'package': package_name,
                'namespace': f'/{node.name}',
                'parameters': []
            }
            
            # Add parameters if any
            for param in node.content.parameters:
                node_config['parameters'].append({
                    'name': param.name,
                    'value': param.default_value or 'null'
                })
            
            nodes.append(node_config)
        
        return {
            'package_name': package_name,
            'nodes': nodes,
            'description': 'Launch all nodes from RoboDSL specification'
        }
    
    def _prepare_node_launch_context(self, node: NodeNode) -> Dict[str, Any]:
        """Prepare context for node launch template rendering."""
        # Determine package name
        package_name = getattr(node, 'package_name', 'robodsl_package')
        
        # Prepare parameters
        parameters = []
        for param in node.content.parameters:
            parameters.append({
                'name': param.name,
                'value': param.default_value or 'null'
            })
        
        # Determine if this is a lifecycle node
        is_lifecycle = node.content.lifecycle is not None
        
        return {
            'node_name': node.name,
            'executable': f'{node.name}_node',
            'package': package_name,
            'namespace': f'/{node.name}',
            'parameters': parameters,
            'is_lifecycle': is_lifecycle,
            'description': f'Launch {node.name} node'
        }
    
    def _generate_fallback_main_launch(self, ast: RoboDSLAST) -> str:
        """Generate a fallback main launch file if template fails."""
        package_name = getattr(ast, 'package_name', 'robodsl_package')
        
        content = """#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
"""
        
        # Add each node
        for node in ast.nodes:
            content += f"""        Node(
            package='{package_name}',
            executable='{node.name}_node',
            name='{node.name}_node',
            namespace='/{node.name}',
"""
            
            # Add parameters if any
            if node.content.parameters:
                content += "            parameters=[\n"
                for param in node.content.parameters:
                    value = param.default_value or 'null'
                    content += f"                {{'{param.name}': {value}}},\n"
                content += "            ],\n"
            
            content += "        ),\n"
        
        content += "    ])\n"
        
        return content
    
    def _generate_fallback_node_launch(self, node: NodeNode) -> str:
        """Generate a fallback node launch file if template fails."""
        package_name = getattr(node, 'package_name', 'robodsl_package')
        
        content = f"""#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{package_name}',
            executable='{node.name}_node',
            name='{node.name}_node',
            namespace='/{node.name}',
"""
        
        # Add parameters if any
        if node.content.parameters:
            content += "            parameters=[\n"
            for param in node.content.parameters:
                value = param.default_value or 'null'
                content += f"                {{'{param.name}': {value}}},\n"
            content += "            ],\n"
        
        content += """        ),
    ])
"""
        
        return content 