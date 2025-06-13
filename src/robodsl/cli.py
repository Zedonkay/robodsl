"""Command-line interface for RoboDSL."""

import os
import sys
import click
from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
from datetime import datetime

def create_robodsl_config(project_path: Path, node_name: str, publishers: List[Dict[str, str]] = None,
                       subscribers: List[Dict[str, str]] = None) -> None:
    """Create or update a RoboDSL configuration file for a node.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node
        publishers: List of publisher configurations
        subscribers: List of subscriber configurations
    """
    config_file = project_path / f"{node_name}.robodsl"
    
    # Read existing config if it exists
    config_lines = []
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_lines = f.readlines()
    
    # Find or create node section
    node_section = f"node {node_name}"
    node_start = -1
    node_end = -1
    
    for i, line in enumerate(config_lines):
        if line.strip().startswith('node '):
            if node_start == -1:  # First node found
                node_start = i
            current_node = line.split('{')[0].strip().split(' ')[1]
            if current_node == node_name:
                node_start = i
                # Find the end of this node
                brace_count = 0
                for j in range(i, len(config_lines)):
                    brace_count += config_lines[j].count('{')
                    brace_count -= config_lines[j].count('}')
                    if brace_count == 0 and j > i:
                        node_end = j + 1
                        break
                break
    
    # Create or update node section
    new_node_lines = []
    if node_start == -1:  # Node doesn't exist, add it
        new_node_lines.append(f"node {node_name} {{")
    else:  # Node exists, update it
        new_node_lines = config_lines[node_start:node_end]
    
    # Add publishers
    for pub in (publishers or []):
        pub_line = f"    publisher {pub['topic']} {pub['msg_type']}"
        if not any(pub_line in line for line in new_node_lines):
            new_node_lines.insert(-1, f"{pub_line}\n")
    
    # Add subscribers
    for sub in (subscribers or []):
        sub_line = f"    subscriber {sub['topic']} {sub['msg_type']}"
        if not any(sub_line in line for line in new_node_lines):
            new_node_lines.insert(-1, f"{sub_line}\n")
    
    # Close node if we created it
    if node_start == -1:
        new_node_lines.append("}\n")
    
    # Update config file
    if node_start == -1:  # New node, append to file
        with open(config_file, 'a') as f:
            f.write('\n'.join(new_node_lines) + '\n')
    else:  # Update existing node
        updated_lines = config_lines[:node_start] + new_node_lines + config_lines[node_end:]
        with open(config_file, 'w') as f:
            f.writelines(updated_lines)

def create_node_files(project_path: Path, node_name: str, language: str = 'cpp') -> None:
    """Create the necessary files for a new node.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node
        language: Programming language to use ('cpp' or 'python')
    """
    # Create src directory if it doesn't exist
    src_dir = project_path / 'src'
    src_dir.mkdir(exist_ok=True)
    
    # Create include directory for C++
    if language == 'cpp':
        include_dir = project_path / 'include' / node_name
        include_dir.mkdir(parents=True, exist_ok=True)
    
    # Create launch directory
    launch_dir = project_path / 'launch'
    launch_dir.mkdir(exist_ok=True)
    
    # Create config directory
    config_dir = project_path / 'config'
    config_dir.mkdir(exist_ok=True)
    
    # Create node file
    if language == 'cpp':
        # Create header file
        header_content = f"""#ifndef {node_name.upper()}_NODE_H
#define {node_name.upper()}_NODE_H

#include <rclcpp/rclcpp.hpp>

class {node_name.capitalize()}Node : public rclcpp::Node {{
public:
    {node_name.capitalize()}Node();

private:
    // Member variables and callbacks go here
}};

#endif  // {node_name.upper()}_NODE_H
"""
        with open(include_dir / f"{node_name}_node.h", 'w') as f:
            f.write(header_content)
        
        # Create source file
        source_content = f"""#include "{node_name}/{node_name}_node.h"

{node_name.capitalize()}Node::{node_name.capitalize()}Node()
    : Node("{node_name}")
{{
    // Node initialization code here
    RCLCPP_INFO(this->get_logger(), "{node_name} node has been started");
}}
"""
        with open(src_dir / f"{node_name}_node.cpp", 'w') as f:
            f.write(source_content)
    
    elif language == 'python':
        # Create Python node file
        python_content = f"""#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class {node_name.capitalize()}Node(Node):
    def __init__(self):
        super().__init__('{node_name}')
        self.get_logger().info('{node_name} node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = {node_name.capitalize()}Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
        node_file = src_dir / f"{node_name}_node.py"
        with open(node_file, 'w') as f:
            f.write(python_content)
        
        # Make the file executable
        node_file.chmod(0o755)
    
    # Create launch file
    launch_content = f"""from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{project_path.name}',
            executable='{node_name}_node',
            name='{node_name}',
            output='screen',
            parameters=[
                # Add parameters here
            ]
        )
    ])
"""
    with open(launch_dir / f"{node_name}.launch.py", 'w') as f:
        f.write(launch_content)
    
    # Create config file
    config_content = """# Node configuration parameters
{
    # Add parameters here
}
"""
    with open(config_dir / f"{node_name}.yaml", 'w') as f:
        f.write(config_content)

@click.group()
@click.version_option()
def main() -> None:
    """RoboDSL - A DSL for GPU-accelerated robotics applications with ROS2 and CUDA."""
    pass

@main.command()
@click.argument('project_name')
@click.option('--template', '-t', default='basic', help='Template to use for the project')
@click.option('--output-dir', '-o', default='.', help='Directory to create the project in')
def init(project_name: str, template: str, output_dir: str) -> None:
    """Initialize a new RoboDSL project."""
    project_path = Path(output_dir) / project_name
    
    try:
        project_path.mkdir(parents=True, exist_ok=False)
        click.echo(f"Created project directory: {project_path}")
        
        # Create basic project structure
        (project_path / 'src').mkdir()
        (project_path / 'include').mkdir()
        (project_path / 'launch').mkdir()
        (project_path / 'config').mkdir()
        (project_path / 'robodsl').mkdir()
        
        # Create a basic robodsl file
        (project_path / f"{project_name}.robodsl").write_text(
            f"# {project_name} RoboDSL configuration\n\n"
            "node main_node {{\n    # Define your ROS2 node configuration here\n    # Example:\n    # publisher /camera/image_raw sensor_msgs/msg/Image\n    # subscriber /cmd_vel geometry_msgs/msg/Twist\n}}\n\n"
            "cuda_kernels {{\n    # Define your CUDA kernels here\n    # Example:\n    # kernel process_image {{\n    #     input: Image\n    #     output: Image\n    #     block_size: (16, 16, 1)\n    # }}\n}}"
        )
        
        click.echo(f"Initialized RoboDSL project in {project_path}")
        click.echo(f"Edit {project_name}.robodsl to define your application")
        
    except FileExistsError:
        click.echo(f"Error: Directory {project_path} already exists", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@main.command()
@click.argument('project_dir', default='.')
def build(project_dir: str) -> None:
    """Build the RoboDSL project."""
    project_dir = Path(project_dir).resolve()
    click.echo(f"Building project in {project_dir}...")
    
    # TODO: Implement build logic
    click.echo("Build command not yet implemented")

@main.command()
@click.argument('node_name')
@click.option('--publisher', '-p', 'publishers', multiple=True, type=(str, str), 
              help='Add a publisher with format: TOPIC MESSAGE_TYPE')
@click.option('--subscriber', '-s', 'subscribers', multiple=True, type=(str, str),
              help='Add a subscriber with format: TOPIC MESSAGE_TYPE')
@click.option('--language', '-l', type=click.Choice(['cpp', 'python'], case_sensitive=False),
              default='cpp', help='Programming language for the node')
@click.option('--project-dir', '-d', default='.',
              help='Path to the project directory (default: current directory)')
def add_node(node_name: str, publishers: List[tuple], subscribers: List[tuple], 
            language: str, project_dir: str) -> None:
    """Add a new node to an existing project.
    
    Args:
        node_name: Name of the node to create
        publishers: List of (topic, message_type) tuples for publishers
        subscribers: List of (topic, message_type) tuples for subscribers
        language: Programming language to use ('cpp' or 'python')
        project_dir: Path to the project directory
    """
    project_path = Path(project_dir).resolve()
    
    # Validate project directory
    if not project_path.exists():
        click.echo(f"Error: Directory '{project_path}' does not exist", err=True)
        sys.exit(1)
    
    # Convert publishers/subscribers to dict format
    pub_configs = [{'topic': p[0], 'msg_type': p[1]} for p in publishers]
    sub_configs = [{'topic': s[0], 'msg_type': s[1]} for s in subscribers]
    
    try:
        # Create or update the RoboDSL config
        create_robodsl_config(project_path, node_name, pub_configs, sub_configs)
        
        # Create node files
        create_node_files(project_path, node_name, language)
        
        click.echo(f"Successfully created node '{node_name}' in {project_path}")
        click.echo(f"\nNext steps:")
        click.echo(f"1. Edit {node_name}.robodsl to configure your node")
        click.echo(f"2. Implement your node logic in src/{node_name}_node.{'cpp' if language == 'cpp' else 'py'}")
        click.echo(f"3. Build and run your node")
        
    except Exception as e:
        click.echo(f"Error: Failed to create node: {str(e)}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
