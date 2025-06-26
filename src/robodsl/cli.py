"""Command-line interface for RoboDSL."""

import os
import sys
import click
from pathlib import Path
from typing import Optional, List, Dict

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
    src_dir = project_path / 'src'
    include_dir = project_path / 'include'
    
    # Create source and header file paths
    if len(parts) > 1:
        # For subnodes, create a nested directory structure
        src_dir = src_dir / '/'.join(parts[:-1])
        src_file = src_dir / f"{node_base_name}_node.cpp"
        
        # Create nested include directories
        node_include_dir = include_dir / '/'.join(parts[:-1])
        node_include_dir.mkdir(parents=True, exist_ok=True)
        header_file = node_include_dir / f"{node_base_name}_node.hpp"
    else:
        # For top-level nodes
        src_file = src_dir / f"{node_base_name}_node.cpp"
        node_include_dir = include_dir
        header_file = node_include_dir / f"{node_base_name}_node.hpp"
    
    return src_file, header_file


def _snake_to_camel(snake_str: str) -> str:
    """Convert a snake_case string to CamelCase.
    
    Args:
        snake_str: The snake_case string to convert
        
    Returns:
        The input string converted to CamelCase
    """
    # Handle empty or None input
    if not snake_str:
        return snake_str
        
    # Split on underscores and capitalize each word
    components = snake_str.split('_')
    # Capitalize the first letter of each component and join them
    return ''.join(x.capitalize() for x in components if x)


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

def node_main(args=None):
    rclpy.init(args=args)
    node = {node_name.capitalize()}Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    node_main()
""")
        
        # Make the file executable
        if os.name != 'nt':  # Only on Unix-like systems
            os.chmod(node_file, 0o755)
            
    elif language == 'cpp':
        node_file, include_file = get_node_file_paths(project_path, node_name, language)
        
        # Create parent directories if they don't exist
        node_file.parent.mkdir(parents=True, exist_ok=True)
        include_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate include guard and class name
        include_guard = f"{'_'.join(node_name.split('.')).upper()}_NODE_H_"
        class_name = f"{_snake_to_camel(node_base_name)}Node"
        
        # Write header file
        with open(include_file, 'w') as f:
            f.write(f"""#ifndef {include_guard}
#define {include_guard}

#include <rclcpp/rclcpp.hpp>

class {class_name} : public rclcpp::Node {{
public:
    {class_name}();

private:
    // Add your members and callbacks here
}};

#endif // {include_guard}
""")
        
        # Write source file
        with open(node_file, 'w') as f:
            f.write(f"""#include "{node_name.replace('.', '/')}_node.hpp"

{class_name}::{class_name}()
: Node("{node_name}")
{{
    RCLCPP_INFO(this->get_logger(), "{node_name} node started");
}}

int main(int argc, char * argv[])
{{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<{class_name}>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}}""")

def _create_launch_file_impl(project_path: Path, node_name: str, language: str = 'python') -> None:
    """Internal implementation for creating a launch file for the node.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node (can contain dots for subnodes)
        language: Programming language ('python' or 'cpp')
    """
    parts = node_name.split('.')
    node_base_name = parts[-1]
    
    # Create launch directory if it doesn't exist
    launch_dir = project_path / 'launch' / '/'.join(parts[:-1]) if len(parts) > 1 else project_path / 'launch'
    launch_dir = Path(launch_dir)  # Ensure it's a Path object
    launch_dir.mkdir(parents=True, exist_ok=True)
    
    # For Python, the package is the first part of the node name or project name
    package_name = parts[0] if len(parts) > 1 else str(project_path.name)
    
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
@click.pass_context
def main(ctx):
    """RoboDSL - A DSL for GPU-accelerated robotics applications with ROS2 and CUDA."""
    # Enable debug output if environment variable is set
    debug = os.environ.get('ROBODSL_DEBUG', '').lower() in ('1', 'true', 'yes')
    if debug:
        import sys
        print(f"DEBUG: Command: {' '.join(sys.argv)}", file=sys.stderr)
        print(f"DEBUG: Context: {ctx}", file=sys.stderr)
        print(f"DEBUG: Params: {ctx.params}", file=sys.stderr)

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
        
        # Create a comprehensive robodsl file
        (project_path / f"{project_name}.robodsl").write_text(
            f"# {project_name} RoboDSL Configuration\n\n"
            "# Project configuration\n"
            f'project_name: {project_name}\n\n'
            "# Global includes (will be added to all nodes)\n"
            "#include <rclcpp/rclcpp.hpp>\n"
            "#include <std_msgs/msg/string.hpp>\n"
            "#include <sensor_msgs/msg/image.hpp>\n\n"
            "# Main node configuration\n"
            "node main_node {\n"
            "    # Node namespace (optional)\n"
            f"    namespace: /{project_name}\n\n"
            "    # Enable lifecycle (default: false)\n"
            "    # lifecycle: true\n\n"
            "    # Enable parameter callbacks (default: false)\n"
            "    # parameter_callbacks: true\n\n"
            "    # Topic remapping (optional)\n"
            "    # remap /source_topic /target_topic\n\n"
            "    # Parameters with different types\n"
            "    parameter int count: 0\n"
            "    parameter double rate: 10.0\n"
            f"    parameter string name: \"{project_name}\"\n"
            "    parameter bool enabled: true\n\n"
            "    # Publisher with QoS settings\n"
            "    publisher /chatter std_msgs/msg/String {\n"
            "        qos: {\n"
            "            reliability: reliable\n"
            "            history: keep_last\n"
            "            depth: 10\n"
            "        }\n"
            "        queue_size: 10\n"
            "    }\n\n"
            "    # Subscriber with QoS settings\n"
            "    subscriber /chatter std_msgs/msg/String {\n"
            "        qos: {\n"
            "            reliability: best_effort\n"
            "            history: keep_last\n"
            "            depth: 10\n"
            "        }\n"
            "        queue_size: 10\n"
            "    }\n\n"
            "    # Timer example (1.0 second period)\n"
            "    timer my_timer 1.0 on_timer_callback\n"
            "}\n\n"
            "# CUDA Kernels section\n"
            "cuda_kernels {\n"
            "    # Example vector addition kernel\n"
            "    kernel vector_add {\n"
            "        # Input parameters\n"
            "        input float* a\n"
            "        input float* b\n"
            "        output float* c\n"
            "        input int size\n\n"
            "        # Kernel configuration\n"
            "        block_size = (256, 1, 1)\n\n"
            "        # Include additional headers\n"
            "        include <cuda_runtime.h>\n"
            "        include <device_launch_parameters.h>\n\n"
            "        # Kernel code\n"
            "        code {\n"
            "            __global__ void vector_add(const float* a, const float* b, float* c, int size) {\n"
            "                int i = blockIdx.x * blockDim.x + threadIdx.x;\n"
            "                if (i < size) {\n"
            "                    c[i] = a[i] + b[i];\n"
            "                }\n"
            "            }\n"
            "        }\n"
            "    }\n"
            "}\n\n"
            "# For more examples and documentation, see: examples/comprehensive_example.robodsl\n"
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
@click.option('--language', '-l', type=click.Choice(['cpp', 'python'], case_sensitive=False),
              default='cpp', help='Programming language for the node')
@click.option('--project-dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path, exists=True),
              default='.', help='Project directory (default: current directory)')
def create_launch_file(node_name: str, language: str, project_dir: Path) -> None:
    """Create a launch file for a node.
    
    Args:
        node_name: Name of the node (can contain dots for subnodes)
        language: Programming language of the node ('python' or 'cpp')
        project_dir: Path to the project directory
    """
    project_path = Path(project_dir).resolve()
    try:
        click.echo(f"Creating launch file for node '{node_name}' in {project_path}...")
        _create_launch_file_impl(project_path, node_name, language)
        
        # Get the launch file path for the success message
        parts = node_name.split('.')
        node_base_name = parts[-1]
        launch_dir = project_path / 'launch' / '/'.join(parts[:-1]) if len(parts) > 1 else project_path / 'launch'
        launch_dir = Path(launch_dir)  # Ensure it's a Path object
        launch_file = launch_dir / f"{node_base_name}.launch.py"
        
        click.echo(f"Created launch file: {launch_file.relative_to(project_path)}")
        click.echo("\nTo use this launch file, run:")
        click.echo(f"  ros2 launch {project_path.name} {node_base_name}.launch.py")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@main.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              default=None, help='Output directory for generated files')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
def generate(input_file: Path, output_dir: Optional[Path], force: bool) -> None:
    """
    Generate code from a RoboDSL file.
    
    This command processes a .robodsl file and generates the corresponding
    CUDA/ROS2 source files, headers, and build configuration.
    """
    try:
        from robodsl.parser import parse_robodsl
        from robodsl.generator import CodeGenerator
        
        # Set default output directory if not specified
        if output_dir is None:
            output_dir = input_file.parent
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Processing {input_file}...")
        
        # Parse the input file
        config = parse_robodsl(input_file.read_text())
        
        # Generate code
        generator = CodeGenerator(config, output_dir=output_dir)
        generated_files = generator.generate()
        
        click.echo(f"Generated {len(generated_files)} files in {output_dir}:")
        for file_path in generated_files:
            if isinstance(file_path, (str, Path)):
                file_path = Path(file_path)
                click.echo(f"  - {file_path.relative_to(output_dir) if file_path.is_absolute() else file_path}")
            else:
                click.echo(f"  - {file_path}")
            
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_exc()
        sys.exit(1)

@main.command(name='add-node')
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
    # Debug output
    debug = os.environ.get('ROBODSL_DEBUG', '').lower() in ('1', 'true', 'yes')
    if debug:
        import sys
        print(f"DEBUG: add_node called with:", file=sys.stderr)
        print(f"  node_name: {node_name}", file=sys.stderr)
        print(f"  language: {language}", file=sys.stderr)
        print(f"  project_dir: {project_dir} (type: {type(project_dir)})", file=sys.stderr)
        print(f"  publisher: {publisher}", file=sys.stderr)
        print(f"  subscriber: {subscriber}", file=sys.stderr)
        print(f"  ctx.params: {ctx.params}", file=sys.stderr)
    """Add a new node to an existing project.
    
    Node names can use dot notation for subdirectories, e.g., 'sensors.camera'.
    This will create the appropriate directory structure.
    """
    print(f"DEBUG: add_node called with node_name={node_name}, language={language}, project_dir={project_dir}")
    print(f"DEBUG: publisher={publisher}, subscriber={subscriber}")
    
    try:
        project_path = Path(project_dir).resolve()
        node_base_name = node_name.split('.')[-1]
        
        # Validate node name
        if not node_name.replace('.', '_').replace('-', '_').isidentifier():
            raise click.BadParameter(
                f"Invalid node name: '{node_name}'. "
                "Node names must be valid Python/C++ identifiers"
            )
            
        # Validate project directory exists
        if not project_path.exists():
            click.echo(f"Error: Directory '{project_dir}' does not exist", err=True)
            sys.exit(1)
            
        print(f"DEBUG: Node base name: {node_base_name}")
        print(f"DEBUG: Project path: {project_path} (type: {type(project_path)})")
        
        # Create necessary directories
        src_dir = project_path / 'src'
        include_dir = project_path / 'include'
        launch_dir = project_path / 'launch'
        config_dir = project_path / 'config'
        
        print(f"DEBUG: Creating directories: src={src_dir}, include={include_dir}, launch={launch_dir}, config={config_dir}")
        
        # Create include directory structure for C++
        if language == 'cpp':
            node_parts = node_name.split('.')
            if len(node_parts) > 1:
                include_node_dir = include_dir / '/'.join(node_parts[:-1])
                print(f"DEBUG: Creating C++ include directory: {include_node_dir}")
                include_node_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure all required directories exist
        for directory in [src_dir, include_dir, launch_dir, config_dir]:
            print(f"DEBUG: Creating directory: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create default config YAML
        config_file = config_dir / f"{node_base_name}.yaml"
        print(f"DEBUG: Checking for config file: {config_file}")
        if not config_file.exists():
            print(f"DEBUG: Creating config file: {config_file}")
            with open(str(config_file), 'w') as f:  # Explicitly convert to string for open()
                f.write(f"# Configuration for {node_name} node\n")
                f.write("# Add your node parameters here\n")
                f.write(f"{node_base_name}:\n")
                f.write("  ros__parameters:\n")
                f.write("    param1: value1\n")
        
        # Create node files
        print(f"DEBUG: Creating node files for {node_name}")
        create_node_files(project_path, node_name, language, 
                         [{'topic': p[0], 'msg_type': p[1]} for p in publisher],
                         [{'topic': s[0], 'msg_type': s[1]} for s in subscriber])
        
        # Create launch file
        print(f"DEBUG: Creating launch file for {node_name}")
        create_launch_file(node_name, language, str(project_path))  # Convert to string for click
        
        # Create or update RoboDSL config
        print(f"DEBUG: Creating RoboDSL config for {node_name}")
        create_robodsl_config(
            project_path,
            node_name,
            [{'topic': p[0], 'msg_type': p[1]} for p in publisher],
            [{'topic': s[0], 'msg_type': s[1]} for s in subscriber]
        )
        
        click.echo(f"Node '{node_name}' added successfully!")
        click.echo(f"\nNext steps:")
        click.echo(
            f"1. Edit the node implementation in: {src_dir}/" +
            "/".join(node_name.split('.')) +
            f"_node.{'py' if language == 'python' else 'cpp'}"
        )
        click.echo(f"2. Update the configuration in: {project_path}/robodsl/nodes{'/' + '/'.join(node_name.split('.')[:-1]) if '.' in node_name else ''}/{node_name.split('.')[-1]}.robodsl")
        click.echo(f"3. Launch the node with: ros2 launch {project_path.name} {node_name.split('.')[-1]}.launch.py")
    except Exception as e:
        print(f"DEBUG: Exception in add_node: {e}", file=sys.stderr)
        click.echo(f"Error: Failed to create node: {str(e)}", err=True)
        sys.exit(1)
        
    # Create necessary directories
    src_dir = project_path / 'src'
    include_dir = project_path / 'include'
    launch_dir = project_path / 'launch'
    config_dir = project_path / 'config'
    
    # Create include directory structure for C++
    if language == 'cpp':
        node_parts = node_name.split('.')
        if len(node_parts) > 1:
            include_node_dir = include_dir / '/'.join(node_parts)
            include_node_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure all required directories exist
    for directory in [src_dir, include_dir, launch_dir, config_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create default config YAML
    config_file = config_dir / f"{node_base_name}.yaml"
    if not config_file.exists():
        with open(config_file, 'w') as f:
            f.write(f"# Configuration for {node_name} node\n")
            f.write("# Add your node parameters here\n")
            f.write(f"{node_base_name}:\n")
            f.write("  ros__parameters:\n")
            f.write("    param1: value1\n")
    
    # Create node files
    create_node_files(project_path, node_name, language, 
                     [{'topic': p[0], 'msg_type': p[1]} for p in publisher],
                     [{'topic': s[0], 'msg_type': s[1]} for s in subscriber])
    
    # Create launch file
    create_launch_file(node_name, language, project_dir)
    
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
        click.echo(
            f"1. Edit the node implementation in: {src_dir}/" +
            "/".join(node_name.split('.')) +
            f"_node.{'py' if language == 'python' else 'cpp'}"
        )
        click.echo(f"2. Update the configuration in: {project_path}/robodsl/nodes/{'/'.join(node_name.split('.'))}.robodsl")
        click.echo(f"3. Launch the node with: ros2 launch {project_path.name} {node_name}.launch.py")
    except Exception as e:
        click.echo(f"Error: Failed to create node: {str(e)}", err=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
