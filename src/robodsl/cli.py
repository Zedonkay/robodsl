"""Command-line interface for RoboDSL."""

import os
import sys
import click
from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
from datetime import datetime

def get_node_paths(project_path: Path, node_name: str) -> tuple[Path, str]:
    """Get the file path and node name, handling subnodes.
    
    Args:
        project_path: Base project directory
        node_name: Node name, potentially with dots for subnodes
        
    Returns:
        Tuple of (config_file_path, node_name_without_namespace)
    """
    # Split node name into components
    parts = node_name.split('.')
    node_base_name = parts[-1]
    
    # Create subdirectories if they don't exist
    if len(parts) > 1:
        node_dir = project_path / 'robodsl' / 'nodes' / '/'.join(parts[:-1])
        node_dir.mkdir(parents=True, exist_ok=True)
        config_file = node_dir / f"{node_base_name}.robodsl"
        return config_file, node_base_name
    
    return project_path / f"{node_name}.robodsl", node_name


def create_robodsl_config(project_path: Path, node_name: str, publishers: List[Dict[str, str]] = None,
                       subscribers: List[Dict[str, str]] = None) -> None:
    """Create or update a RoboDSL configuration file for a node.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node (can contain dots for subnodes)
        publishers: List of publisher configurations
        subscribers: List of subscriber configurations
    """
    config_file, node_base_name = get_node_paths(project_path, node_name)
    
    # Read existing config if it exists
    config_lines = []
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_lines = f.readlines()
    
    # Find or create node section - use base node name only
    node_section = f"node {node_base_name}"
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
        new_node_lines.append(f"node {node_base_name} {{\n")
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
        # Create parent directories if they don't exist
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the config file
        with open(config_file, 'w') as f:
            f.write(''.join(config_lines + new_node_lines) + '\n')
    else:  # Update existing node
        updated_lines = config_lines[:node_start] + new_node_lines + config_lines[node_end:]
        with open(config_file, 'w') as f:
            f.writelines(updated_lines)

def get_node_file_paths(project_path: Path, node_name: str, language: str) -> tuple[Path, Path]:
    """Get the source and include file paths for a node, handling subnodes.
    
    Args:
        project_path: Base project directory
        node_name: Node name (can contain dots for subnodes)
        language: Programming language ('python' or 'cpp')
        
    Returns:
        Tuple of (source_file_path, include_file_path or None if Python)
    """
    parts = node_name.split('.')
    node_base_name = parts[-1]
    
    # For Python
    if language == 'python':
        if len(parts) > 1:
            # For subnodes, create a package-like structure
            src_dir = project_path / 'src' / '/'.join(parts[:-1])
            src_dir.mkdir(parents=True, exist_ok=True)
            src_file = src_dir / f"{node_base_name}_node.py"
            return src_file, None
        return project_path / 'src' / f"{node_name}_node.py", None
    
    # For C++
    src_dir = project_path / 'src' / '/'.join(parts[:-1]) if len(parts) > 1 else project_path / 'src'
    include_dir = project_path / 'include' / '/'.join(parts[:-1]) if len(parts) > 1 else project_path / 'include'
    
    src_dir.mkdir(parents=True, exist_ok=True)
    include_dir.mkdir(parents=True, exist_ok=True)
    
    src_file = src_dir / f"{node_base_name}_node.cpp"
    header_file = include_dir / f"{node_base_name}_node.hpp"
    
    return src_file, header_file


def create_node_files(project_path: Path, node_name: str, language: str = 'python',
                    publishers: List[Dict[str, str]] = None,
                    subscribers: List[Dict[str, str]] = None) -> None:
    """Create node source files.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node (can contain dots for subnodes)
        language: Programming language ('python' or 'cpp')
        publishers: List of publisher configurations
        subscribers: List of subscriber configurations
    """
    # Get base node name (without namespace)
    node_base_name = node_name.split('.')[-1]
    
    # Get file paths
    if language == 'python':
        node_file, _ = get_node_file_paths(project_path, node_name, language)
        
        # Create parent directories if they don't exist
        node_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate Python imports based on node name parts
        module_parts = node_name.split('.')
        imports = []
        if len(module_parts) > 1:
            imports.append(f"from {' import '.join(module_parts[:-1])} import {module_parts[-2]}")
        
        with open(node_file, 'w') as f:
            f.write(f"""#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

{''.join(imports)}

class {node_base_name.capitalize()}Node(Node):
    def __init__(self):
        super().__init__('{node_name}')
        self.get_logger().info('{node_name} node started')
        self.get_logger().info('{node_name} node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = {node_name.capitalize()}Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
""")
        
        # Make the file executable
        if os.name != 'nt':  # Only on Unix-like systems
            os.chmod(node_file, 0o755)
            
    elif language == 'cpp':
        node_file, include_file = get_node_file_paths(project_path, node_name, language)
        
        # Create parent directories if they don't exist
        node_file.parent.mkdir(parents=True, exist_ok=True)
        include_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate include guard
        include_guard = f"{'_'.join(node_name.split('.')).upper()}_NODE_H_"
        
        # Write header file
        with open(include_file, 'w') as f:
            f.write(f"""#ifndef {include_guard}
#define {include_guard}

#include <rclcpp/rclcpp.hpp>

class {node_base_name.capitalize()}Node : public rclcpp::Node {{
public:
    {node_base_name.capitalize()}Node();

private:
    // Add your members and callbacks here
}};

#endif // {include_guard}
""")
        
        # Write source file
        with open(node_file, 'w') as f:
            f.write(f"""#include "{node_name.replace('.', '/')}_node.hpp"

{node_base_name.capitalize()}Node::{node_base_name.capitalize()}Node()
: Node("{node_name}")
{{
    RCLCPP_INFO(this->get_logger(), "{node_name} node started");
}}

int main(int argc, char * argv[])
{{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<{node_base_name.capitalize()}Node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}}""")

def create_launch_file(project_path: Path, node_name: str, language: str = 'python') -> None:
    """Create a launch file for the node.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node (can contain dots for subnodes)
        language: Programming language ('python' or 'cpp')
    """
    parts = node_name.split('.')
    node_base_name = parts[-1]
    
    # Create launch directory if it doesn't exist
    launch_dir = project_path / 'launch' / '/'.join(parts[:-1]) if len(parts) > 1 else project_path / 'launch'
    launch_dir.mkdir(parents=True, exist_ok=True)
    
    # For Python, the package is the first part of the node name or project name
    package_name = parts[0] if len(parts) > 1 else project_path.name
    
    # For C++, the executable is just the base name
    executable_name = node_base_name + ('_node' if language == 'python' else '')
    
    launch_file = launch_dir / f"{node_base_name}.launch.py"
    
    with open(launch_file, 'w') as f:
        f.write(f"""from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{package_name}',
            executable='{executable_name}',
            name='{node_name}',
            output='screen',
            emulate_tty=True,
        )
    ])
""")

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
@click.option('--publisher', '-p', multiple=True, nargs=2,
              help='Add a publisher with format: TOPIC MESSAGE_TYPE')
@click.option('--subscriber', '-s', multiple=True, nargs=2,
              help='Add a subscriber with format: TOPIC MESSAGE_TYPE')
@click.option('--language', '-l', type=click.Choice(['cpp', 'python'], case_sensitive=False),
              default='python', help='Programming language for the node')
@click.option('--project-dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True),
              default=Path.cwd(), help='Project directory (default: current directory)')
def add_node(node_name: str, publisher: List[tuple], subscriber: List[tuple], 
            language: str, project_dir: Path) -> None:
    """Add a new node to an existing project.
    
    Node names can use dot notation for subdirectories, e.g., 'sensors.camera'.
    This will create the appropriate directory structure.
    """
    project_path = project_dir
    node_base_name = node_name.split('.')[-1]
    
    # Validate node name
    if not node_name.replace('.', '_').replace('-', '_').isidentifier():
        raise click.BadParameter(
            f"Invalid node name: '{node_name}'. "
            "Node names must be valid Python/C++ identifiers"
        )
    
    # Create necessary directories
    src_dir = project_path / 'src'
    include_dir = project_path / 'include'
    launch_dir = project_path / 'launch'
    config_dir = project_path / 'config'
    
    # Create node files
    create_node_files(project_path, node_name, language, 
                     [{'topic': p[0], 'msg_type': p[1]} for p in publisher],
                     [{'topic': s[0], 'msg_type': s[1]} for s in subscriber])
    
    # Create launch file
    create_launch_file(project_path, node_name, language)
    
    # Create or update RoboDSL config
    create_robodsl_config(
        project_path,
        node_name,
        [{'topic': p[0], 'msg_type': p[1]} for p in publisher],
        [{'topic': s[0], 'msg_type': s[1]} for s in subscriber]
    )
    
    try:
        click.echo(f"Node '{node_name}' added successfully!")
        click.echo(f"\nNext steps:")
        click.echo(f"1. Edit the node implementation in: {src_dir}/" + "/".join(node_name.split('.')) + f"_node.{'py' if language == 'python' else 'cpp'}")
        click.echo(f"2. Update the configuration in: {project_path}/robodsl/nodes/{'/'.join(node_name.split('.'))}.robodsl")
        click.echo(f"3. Launch the node with: ros2 launch {project_path.name} {node_name}.launch.py")
    except Exception as e:
        click.echo(f"Error: Failed to create node: {str(e)}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
