"""Command-line interface for RoboDSL."""

import os
import sys
import click
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

def create_robodsl_node_file(project_path: Path, node_name: str, template: str = "basic") -> Path:
    """Create a RoboDSL node definition file.
    
    Args:
        project_path: Path to the project directory
        node_name: Name of the node (can contain dots for subnodes)
        template: Template type ('basic', 'publisher', 'subscriber', 'cuda', 'full', 'data_structures')
        
    Returns:
        Path to the created RoboDSL file
    """
    # Split node name into components for directory structure
    parts = node_name.split('.')
    node_base_name = parts[-1]
    
    # Create flat structure - subnodes are CLI-only for organization
    # All robodsl files go in the same directory, but we create nested dirs for organization
    if len(parts) > 1:
        # Create nested directory structure for organization
        node_dir = project_path / 'robodsl' / 'nodes' / '/'.join(parts[:-1])
        node_dir.mkdir(parents=True, exist_ok=True)
        config_file = node_dir / f"{node_base_name}.robodsl"
    else:
        config_file = project_path / f"{node_name}.robodsl"
    
    # Create parent directories if they don't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate different templates based on type
    if template == "basic":
        content = f"""// {node_name} RoboDSL Configuration

// Node definition
node {node_base_name} {{
    // Node namespace
    namespace: /{node_name.replace('.', '/')}
    
    // Parameters
    parameter int count = 0
    parameter double rate = 10.0
    parameter string name = "{node_name}"
    parameter bool enabled = true
    
    // Timer for periodic processing
    timer main_timer: 1.0 {{
        callback: "on_timer_callback"
    }}
    
    // C++ method for timer callback
    method on_timer_callback {{
        input: rclcpp::Time current_time
        code {{
            RCLCPP_INFO(this->get_logger(), "Timer callback executed");
        }}
    }}
}}
"""
    elif template == "publisher":
        content = f"""// {node_name} RoboDSL Configuration - Publisher Node

// Node definition
node {node_base_name} {{
    // Node namespace
    namespace: /{node_name.replace('.', '/')}
    
    // Parameters
    parameter double publish_rate = 10.0
    parameter string message = "Hello from {node_name}"
    
    // Publisher
    publisher /chatter: "std_msgs/msg/String" {{
        qos {{
            reliability: 1
            history: 1
            depth: 10
        }}
        queue_size: 10
    }}
    
    // Timer for publishing
    timer publish_timer: 1.0 / publish_rate {{
        callback: on_publish_timer
    }}
    
    // C++ method for timer callback
    method on_publish_timer {{
        input: rclcpp::Time current_time
        code {{
            auto message = std_msgs::msg::String();
            message.data = this->get_parameter("message").as_string();
            chatter_pub_->publish(message);
        }}
    }}
}}
"""
    elif template == "subscriber":
        content = f"""// {node_name} RoboDSL Configuration - Subscriber Node

// Node definition
node {node_base_name} {{
    // Node namespace
    namespace: /{node_name.replace('.', '/')}
    
    // Parameters
    parameter bool verbose = true
    
    // Subscriber
    subscriber /chatter: "std_msgs/msg/String" {{
        qos {{
            reliability: 0
            history: 1
            depth: 10
        }}
        queue_size: 10
    }}
    
    // C++ method for message callback
    method on_message_received {{
        input: const std_msgs::msg::String::SharedPtr msg
        code {{
            if (this->get_parameter("verbose").as_bool()) {{
                RCLCPP_INFO(this->get_logger(), "Received: %s", msg->data.c_str());
            }}
        }}
    }}
}}
"""
    elif template == "cuda":
        content = f"""// {node_name} RoboDSL Configuration - CUDA Node

// Node definition
node {node_base_name} {{
    // Node namespace
    namespace: /{node_name.replace('.', '/')}
    
    // Parameters
    parameter int data_size = 1024
    parameter double process_rate = 30.0
    
    // Subscriber for input data
    subscriber /input_data: "std_msgs/msg/Float32MultiArray" {{
        qos {{
            reliability: 1
            history: 1
            depth: 10
        }}
    }}
    
    // Publisher for processed data
    publisher /output_data: "std_msgs/msg/Float32MultiArray" {{
        qos {{
            reliability: 1
            history: 1
            depth: 10
        }}
    }}
    
    // Timer for processing
    timer process_timer: 1.0 / process_rate {{
        callback: "on_process_timer"
    }}
    
    // C++ method for processing
    method on_process_timer {{
        input: rclcpp::Time current_time
        code {{
            // Process data using CUDA kernel
            int size = this->get_parameter("data_size").as_int();
            process_data_cuda(size);
        }}
    }}
    
    // C++ method for message callback
    method on_input_received {{
        input: const std_msgs::msg::Float32MultiArray::SharedPtr msg
        code {{
            // Store input data for processing
            input_data_ = msg->data;
        }}
    }}
}}

// CUDA Kernels
cuda_kernels {{
    kernel process_data {{
        input: float* input_data, int size
        output: float* output_data
        
        block_size: (256, 1, 1)
        
        include <cuda_runtime.h>
        include <device_launch_parameters.h>
        
        code {{
            __global__ void process_data_kernel(const float* input, float* output, int size) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < size) {{
                    // Example processing: multiply by 2
                    output[i] = input[i] * 2.0f;
                }}
            }}
        }}
    }}
}}
"""
    elif template == "full":
        content = f"""// {node_name} RoboDSL Configuration - Full Featured Node

// Node definition
node {node_base_name} {{
    // Node namespace
    namespace: /{node_name.replace('.', '/')}
    
    // Lifecycle configuration
    lifecycle {{
        enabled: true
        autostart: true
        cleanup_on_shutdown: true
    }}
    
    // Parameter callbacks
    parameter_callbacks: true
    
    // Parameters
    parameter int count = 0
    parameter double rate = 10.0
    parameter string name = "{node_name}"
    parameter bool enabled = true
    parameter int queue_size = 10
    
    // Topic remapping
    remap /source_topic: /target_topic
    
    // Publishers
    publisher /chatter: "std_msgs/msg/String" {{
        qos {{
            reliability: 1
            history: 1
            depth: 10
        }}
        queue_size: 10
    }}
    
    publisher /status: "std_msgs/msg/String" {{
        qos {{
            reliability: 1
            history: 1
            depth: 5
        }}
    }}
    
    // Subscribers
    subscriber /chatter: "std_msgs/msg/String" {{
        qos {{
            reliability: 0
            history: 1
            depth: 10
        }}
        queue_size: 10
    }}
    
    // Services
    service /get_status: "std_srvs/srv/Trigger"
    
    // Service clients
    client /other_service: "std_srvs/srv/Trigger"
    
    // Actions
    action /long_task: "example_interfaces/action/Fibonacci"
    
    // Timers
    timer main_timer: 1.0 {{
        callback: on_main_timer
    }}
    
    timer status_timer: 5.0 {{
        callback: on_status_timer
    }}
    
    // C++ methods
    method on_main_timer {{
        input: rclcpp::Time current_time
        code {{
            auto message = std_msgs::msg::String();
            message.data = "Timer tick from " + this->get_parameter("name").as_string();
            chatter_pub_->publish(message);
        }}
    }}
    
    method on_status_timer {{
        input: rclcpp::Time current_time
        code {{
            auto status_msg = std_msgs::msg::String();
            status_msg.data = "Status: Running";
            status_pub_->publish(status_msg);
        }}
    }}
    
    method on_message_received {{
        input: const std_msgs::msg::String::SharedPtr msg
        code {{
            RCLCPP_INFO(this->get_logger(), "Received: %s", msg->data.c_str());
        }}
    }}
    
    method on_service_request {{
        input: const std_srvs::srv::Trigger::Request::SharedPtr request
        output: std_srvs::srv::Trigger::Response::SharedPtr response
        code {{
            response->success = true;
            response->message = "Service called successfully";
        }}
    }}
}}
"""
    elif template == "data_structures":
        content = f"""// {node_name} RoboDSL Configuration - Data Structures Example

// Type definitions
typedef std::vector<float> FloatVector;
using Point3D = geometry_msgs::msg::Point;

// Enums
enum class SensorType {{
    CAMERA,
    LIDAR,
    IMU,
    GPS
}};

enum ProcessingMode {{
    FAST = 0,
    ACCURATE = 1,
    BALANCED = 2
}};

// Structs
struct SensorConfig {{
    std::string name;
    SensorType type;
    double frequency;
    bool enabled;
    
    method initialize {{
        input: const std::string& config_path
        code {{
            // Initialize sensor configuration
            RCLCPP_INFO(this->get_logger(), "Initializing sensor: %s", name.c_str());
        }}
    }}
}};

struct ProcessingPipeline {{
    std::vector<std::string> stages;
    ProcessingMode mode;
    double timeout;
    
    method add_stage {{
        input: const std::string& stage_name
        code {{
            stages.push_back(stage_name);
        }}
    }}
    
    method get_stage_count {{
        output: size_t count
        code {{
            count = stages.size();
        }}
    }}
}};

// Classes
class DataProcessor {{
public:
    DataProcessor() {{
        // Constructor
    }}
    
private:
    FloatVector buffer;
    ProcessingPipeline pipeline;
    
    method process_data {{
        input: const FloatVector& input_data
        output: FloatVector& output_data
        code {{
            // Process the input data
            output_data = input_data;
            for (auto& value : output_data) {{
                value *= 2.0f;
            }}
        }}
    }}
    
    method setup_pipeline {{
        input: ProcessingMode mode
        code {{
            pipeline.mode = mode;
            pipeline.add_stage("preprocess");
            pipeline.add_stage("compute");
            pipeline.add_stage("postprocess");
        }}
    }}
}};

// Node definition
node {node_base_name} {{
    // Node namespace
    namespace: /{node_name.replace('.', '/')}
    
    // Parameters
    parameter string sensor_name = "default_sensor"
    parameter int processing_mode = 1
    parameter double frequency = 30.0
    
    // Publishers
    publisher /processed_data: "std_msgs/msg/Float32MultiArray" {{
        qos {{
            reliability: 1
            history: 1
            depth: 10
        }}
    }}
    
    // Subscribers
    subscriber /raw_data: "std_msgs/msg/Float32MultiArray" {{
        qos {{
            reliability: 0
            history: 1
            depth: 10
        }}
    }}
    
    // Timer
    timer process_timer: 1.0 / frequency {{
        callback: on_process_timer
    }}
    
    // C++ methods
    method on_process_timer {{
        input: rclcpp::Time current_time
        code {{
            // Process data using our custom structures
            DataProcessor processor;
            processor.setup_pipeline(static_cast<ProcessingMode>(this->get_parameter("processing_mode").as_int()));
            
            // Process and publish data
            FloatVector input_data = {{1.0f, 2.0f, 3.0f}};
            FloatVector output_data;
            processor.process_data(input_data, output_data);
            
            // Publish processed data
            auto msg = std_msgs::msg::Float32MultiArray();
            msg.data = output_data;
            processed_data_pub_->publish(msg);
        }}
    }}
    
    method on_raw_data_received {{
        input: const std_msgs::msg::Float32MultiArray::SharedPtr msg
        code {{
            // Handle incoming raw data
            RCLCPP_INFO(this->get_logger(), "Received %zu data points", msg->data.size());
        }}
    }}
}}
"""
    else:
        raise ValueError(f"Unknown template: {template}")
    
    # Write the file
    with open(config_file, 'w') as f:
        f.write(content)
    
    return config_file

def create_project_structure(project_path: Path) -> None:
    """Create the standard RoboDSL project structure.
    
    Args:
        project_path: Path to the project directory
    """
    # Create main directories
    directories = [
        'src',
        'include', 
        'launch',
        'config',
        'robodsl',
        'robodsl/nodes',
        'build',
        'docs'
    ]
    
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)

class DependencyChecker:
    """Check for required dependencies and provide installation instructions."""
    
    def __init__(self):
        self.missing_deps = []
        self.install_instructions = {
            'onnxruntime': {
                'ubuntu': 'sudo apt-get install libonnxruntime-dev',
                'macos': 'brew install onnxruntime',
                'pip': 'pip install onnxruntime',
                'conda': 'conda install -c conda-forge onnxruntime'
            },
            'opencv': {
                'ubuntu': 'sudo apt-get install libopencv-dev',
                'macos': 'brew install opencv',
                'pip': 'pip install opencv-python',
                'conda': 'conda install -c conda-forge opencv'
            },
            'tensorrt': {
                'ubuntu': 'Download from NVIDIA website: https://developer.nvidia.com/tensorrt',
                'macos': 'Download from NVIDIA website: https://developer.nvidia.com/tensorrt',
                'pip': 'pip install tensorrt',
                'conda': 'conda install -c nvidia tensorrt'
            },
            'cuda': {
                'ubuntu': 'sudo apt-get install nvidia-cuda-toolkit',
                'macos': 'Download from NVIDIA website: https://developer.nvidia.com/cuda-downloads',
                'pip': 'pip install nvidia-cuda-runtime-cu12',
                'conda': 'conda install -c nvidia cuda'
            },
            'ros2': {
                'ubuntu': 'sudo apt-get install ros-humble-desktop',
                'macos': 'brew install ros2',
                'pip': 'pip install ros2',
                'conda': 'conda install -c conda-forge ros2'
            }
        }
    
    def check_dependency(self, name: str, test_commands: list) -> bool:
        """Check if a dependency is available."""
        for cmd in test_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        self.missing_deps.append(name)
        return False
    
    def check_all_dependencies(self) -> bool:
        """Check all required dependencies."""
        deps = {
            'onnxruntime': [
                ['pkg-config', '--exists', 'onnxruntime'],
                ['python', '-c', 'import onnxruntime; print(onnxruntime.__version__)']
            ],
            'opencv': [
                ['pkg-config', '--exists', 'opencv4'],
                ['pkg-config', '--exists', 'opencv'],
                ['python', '-c', 'import cv2; print(cv2.__version__)']
            ],
            'tensorrt': [
                ['pkg-config', '--exists', 'tensorrt'],
                ['python', '-c', 'import tensorrt; print(tensorrt.__version__)']
            ],
            'cuda': [
                ['nvcc', '--version'],
                ['python', '-c', 'import torch; print(torch.version.cuda)']
            ],
            'ros2': [
                ['ros2', '--version'],
                ['python', '-c', 'import rclpy; print(rclpy.__version__)']
            ]
        }
        
        all_available = True
        for dep_name, commands in deps.items():
            if not self.check_dependency(dep_name, commands):
                all_available = False
        
        return all_available
    
    def get_install_instructions(self) -> str:
        """Get installation instructions for missing dependencies."""
        if not self.missing_deps:
            return ""
        
        instructions = "\nMissing dependencies detected:\n"
        for dep in self.missing_deps:
            instructions += f"\n{dep.upper()}:\n"
            if dep in self.install_instructions:
                for method, cmd in self.install_instructions[dep].items():
                    instructions += f"  {method}: {cmd}\n"
            else:
                instructions += f"  Please install {dep} manually\n"
        
        instructions += "\nAfter installing dependencies, rerun the command.\n"
        return instructions

@click.group()
@click.version_option()
def main() -> None:
    """RoboDSL - A DSL for GPU-accelerated robotics applications with ROS2 and CUDA."""
    pass

@main.command()
@click.argument('project_name')
@click.option('--template', '-t', default='basic', 
              type=click.Choice(['basic', 'publisher', 'subscriber', 'cuda', 'full', 'data_structures']),
              help='Template to use for the project')
@click.option('--output-dir', '-o', default='.', help='Directory to create the project in')
def init(project_name: str, template: str, output_dir: str) -> None:
    """Initialize a new RoboDSL project."""
    project_path = Path(output_dir) / project_name
    
    try:
        project_path.mkdir(parents=True, exist_ok=False)
        click.echo(f"Created project directory: {project_path}")
        
        # Create project structure
        create_project_structure(project_path)
        
        # Create main RoboDSL configuration file
        main_config = project_path / f"{project_name}.robodsl"
        with open(main_config, 'w') as f:
            f.write(f"""// {project_name} RoboDSL Configuration

// Project configuration
project_name: {project_name}

// Global includes (will be added to all nodes)
include <rclcpp/rclcpp.hpp>
include <std_msgs/msg/string.hpp>
include <sensor_msgs/msg/image.hpp>

// Main node configuration
node main_node {{
    // Node namespace
    namespace: /{project_name}
    
    // Parameters
    parameter int count = 0
    parameter double rate = 10.0
    parameter string name = "{project_name}"
    parameter bool enabled = true
    
    // Publisher
    publisher /chatter: "std_msgs/msg/String" {{
        qos {{
            reliability: 1
            history: 1
            depth: 10
        }}
        queue_size: 10
    }}
    
    // Subscriber
    subscriber /chatter: "std_msgs/msg/String" {{
        qos {{
            reliability: 0
            history: 1
            depth: 10
        }}
        queue_size: 10
    }}
    
    // Timer
    timer main_timer: 1.0 {{
        callback: on_timer_callback
    }}
    
    // C++ method for timer callback
    method on_timer_callback {{
        input: rclcpp::Time current_time
        code {{
            auto message = std_msgs::msg::String();
            message.data = "Hello from {project_name}!";
            chatter_pub_->publish(message);
        }}
    }}
    
    // C++ method for message callback
    method on_message_received {{
        input: const std_msgs::msg::String::SharedPtr msg
        code {{
            RCLCPP_INFO(this->get_logger(), "Received: %s", msg->data.c_str());
        }}
    }}
}}

// CUDA Kernels section (optional)
cuda_kernels {{
    // Example vector addition kernel
    kernel vector_add {{
        input: float* a, float* b, int size
        output: float* c
        
        block_size: (256, 1, 1)
        
        include <cuda_runtime.h>
        include <device_launch_parameters.h>
        
        code {{
            __global__ void vector_add_kernel(const float* a, const float* b, float* c, int size) {{
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < size) {{
                    c[i] = a[i] + b[i];
                }}
            }}
        }}
    }}
}}
""")
        
        # Create CMakeLists.txt
        cmake_file = project_path / 'CMakeLists.txt'
        with open(cmake_file, 'w') as f:
            f.write(f"""cmake_minimum_required(VERSION 3.8)
project({project_name})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Only add warning flags for C++ (not CUDA)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# Find CUDA if available
find_package(CUDA QUIET)
if(CUDA_FOUND)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# Include directories
include_directories(include)

# Add executable
add_executable(${{PROJECT_NAME}}_node src/main_node.cpp)
ament_target_dependencies(${{PROJECT_NAME}}_node rclcpp std_msgs sensor_msgs)

# Install
install(TARGETS
  ${{PROJECT_NAME}}_node
  DESTINATION lib/${{PROJECT_NAME}}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${{PROJECT_NAME}}
)

# Install configuration files
install(DIRECTORY
  config
  DESTINATION share/${{PROJECT_NAME}}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
""")
        
        # Create package.xml
        package_file = project_path / 'package.xml'
        with open(package_file, 'w') as f:
            f.write(f"""<?xml version="1.0"?>
<?xml-model href="http://download.ros.org/schema/package_format3.xsd" schematypens="http://www.w3.org/2001/XMLSchema"?>
<package format="3">
  <name>{project_name}</name>
  <version>0.1.0</version>
  <description>RoboDSL generated ROS2 package</description>
  <maintainer email="user@example.com">Your Name</maintainer>
  <license>Apache License 2.0</license>

  <buildtool_depend>ament_cmake</buildtool_depend>

  <depend>rclcpp</depend>
  <depend>std_msgs</depend>
  <depend>sensor_msgs</depend>

  <test_depend>ament_lint_auto</test_depend>
  <test_depend>ament_lint_common</test_depend>

  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
""")
        
        # Create README
        readme_file = project_path / 'README.md'
        with open(readme_file, 'w') as f:
            f.write(f"""# {project_name}

A RoboDSL generated ROS2 package.

## Building

```bash
# Build the package
colcon build --packages-select {project_name}

# Source the workspace
source install/setup.bash
```

## Running

```bash
# Run the main node
ros2 run {project_name} {project_name}_node

# Or use the launch file
ros2 launch {project_name} main_node.launch.py
```

## Development

1. Edit the RoboDSL configuration in `{project_name}.robodsl`
2. Regenerate the C++ code: `robodsl generate {project_name}.robodsl`
3. Build and test your changes

## Project Structure

- `{project_name}.robodsl` - Main RoboDSL configuration
- `src/` - Generated C++ source files
- `include/` - Generated C++ header files
- `launch/` - Launch files
- `config/` - Configuration files
- `robodsl/` - Additional RoboDSL node definitions

## Data Structures

RoboDSL supports defining custom data structures:
- **Structs**: Simple data containers
- **Classes**: Object-oriented data structures with methods
- **Enums**: Enumerated types
- **Typedefs**: Type aliases
- **Using declarations**: Modern C++ type aliases

Example:
```robodsl
struct SensorData {{
    double timestamp;
    std::vector<float> values;
    bool valid;
}};

enum class SensorType {{
    CAMERA,
    LIDAR,
    IMU
}};

typedef std::vector<SensorData> SensorDataArray;
```
""")
        
        click.echo(f"Initialized RoboDSL project '{project_name}' in {project_path}")
        click.echo(f"Edit {project_name}.robodsl to define your application")
        click.echo(f"Run 'robodsl generate {project_name}.robodsl' to generate C++ code")
        
    except FileExistsError:
        click.echo(f"Error: Directory {project_path} already exists", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@main.command()
@click.argument('node_name')
@click.option('--template', '-t', default='basic',
              type=click.Choice(['basic', 'publisher', 'subscriber', 'cuda', 'full', 'data_structures']),
              help='Template to use for the node')
@click.option('--project-dir', type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              default=Path.cwd(), help='Project directory (default: current directory)')
def create_node(node_name: str, template: str, project_dir: Path) -> None:
    """Create a new RoboDSL node definition.
    
    This creates a .robodsl file that defines the node's complete configuration
    including publishers, subscribers, parameters, timers, C++ methods, and data structures.
    """
    project_path = project_dir
    
    # Validate node name
    if not node_name.replace('.', '_').replace('-', '_').isidentifier():
        raise click.BadParameter(
            f"Invalid node name: '{node_name}'. "
            "Node names must be valid Python/C++ identifiers"
        )
        
    # Create project directory if it doesn't exist
    if not project_dir.exists():
        project_dir.mkdir(parents=True, exist_ok=True)
        click.echo(f"Created project directory: {project_dir}")
    
    try:
        click.echo(f"Creating RoboDSL node '{node_name}' with template '{template}'...")
        
        # Create the RoboDSL node file
        config_file = create_robodsl_node_file(project_path, node_name, template)
        
        # Generate source files from the created RoboDSL file
        from robodsl.parsers import parse_robodsl
        from robodsl.generators.main_generator import MainGenerator
        
        click.echo(f"Generating source files...")
        
        # Parse the created RoboDSL file
        config = parse_robodsl(config_file.read_text())
        
        # Generate code
        generator = MainGenerator(output_dir=str(project_path), debug=False)
        generated_files = generator.generate(config)
        
        # For basic template, also create Python files and make them executable
        if template == "basic":
            # Create Python node file
            python_file = project_path / "src" / f"{node_name}_node.py"
            python_file.parent.mkdir(parents=True, exist_ok=True)
            with open(python_file, 'w') as f:
                f.write(f"""#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class {node_name.replace('_', '').title()}Node(Node):
    def __init__(self):
        super().__init__('{node_name}')
        self.publisher_ = self.create_publisher(String, 'chatter', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello World: {{self.i}}'
        self.publisher_.publish(msg)
        self.i += 1

def main(args=None):
    rclpy.init(args=args)
    node = {node_name.replace('_', '').title()}Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
""")
            
            # Make Python file executable
            import os
            import stat
            if os.name != 'nt':  # Skip on Windows
                os.chmod(python_file, os.stat(python_file).st_mode | stat.S_IXUSR)
            
            # Create launch file
            launch_file = project_path / "launch" / f"{node_name}.launch.py"
            launch_file.parent.mkdir(parents=True, exist_ok=True)
            with open(launch_file, 'w') as f:
                f.write(f"""from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{node_name}',
            executable='{node_name}_node',
            name='{node_name}',
            output='screen',
            emulate_tty=True,
        )
    ])
""")
            
            # Create config file
            config_file_yaml = project_path / "config" / f"{node_name}.yaml"
            config_file_yaml.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file_yaml, 'w') as f:
                f.write(f"""{node_name}:
  ros__parameters:
    count: 0
    rate: 10.0
    name: "{node_name}"
    enabled: true
""")
        
        # For CUDA template, also create C++ files with flat structure
        elif template == "cuda":
            # Create header file
            header_file = project_path / "include" / f"{node_name}_node.hpp"
            header_file.parent.mkdir(parents=True, exist_ok=True)
            with open(header_file, 'w') as f:
                f.write(f"""#ifndef {node_name.upper()}_NODE_HPP
#define {node_name.upper()}_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <cuda_runtime.h>

class Object_detectorNode : public rclcpp::Node
{{
public:
    {node_name.replace('_', '').title()}Node();

private:
    void on_process_timer();
    void on_input_received(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void process_data_cuda(int size);

    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr output_data_pub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr input_data_sub_;
    rclcpp::TimerBase::SharedPtr process_timer_;
    
    std::vector<float> input_data_;
}};

#endif // {node_name.upper()}_NODE_HPP
""")
            
            # Create source file
            source_file = project_path / "src" / f"{node_name}_node.cpp"
            source_file.parent.mkdir(parents=True, exist_ok=True)
            with open(source_file, 'w') as f:
                f.write(f"""#include "{node_name}_node.hpp"
#include <chrono>

using namespace std::chrono_literals;

Object_detectorNode::Object_detectorNode()
    : Node("{node_name}")
{{
    // Initialize parameters
    this->declare_parameter("data_size", 1024);
    this->declare_parameter("process_rate", 30.0);
    
    // Create publisher and subscriber
    output_data_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        "output_data", 10);
    input_data_sub_ = this->create_subscription<std_msgs::msg::Float32MultiArray>(
        "input_data", 10,
        std::bind(&Object_detectorNode::on_input_received, this, std::placeholders::_1));
    
    // Create timer
    double process_rate = this->get_parameter("process_rate").as_double();
    process_timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / process_rate),
        std::bind(&Object_detectorNode::on_process_timer, this));
}}

void Object_detectorNode::on_process_timer()
{{
    int size = this->get_parameter("data_size").as_int();
    process_data_cuda(size);
}}

void Object_detectorNode::on_input_received(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{{
    input_data_ = msg->data;
}}

void Object_detectorNode::process_data_cuda(int size)
{{
    // CUDA processing would go here
    // For now, just publish a simple message
    auto output_msg = std_msgs::msg::Float32MultiArray();
    output_msg.data = input_data_;
    output_data_pub_->publish(output_msg);
}}

int main(int argc, char** argv)
{{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<Object_detectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}}
""")
        
        click.echo(f"Created RoboDSL node definition: {config_file.relative_to(project_path)}")
        click.echo(f"Generated {len(generated_files)} source files")
        click.echo(f"\nNext steps:")
        click.echo(f"1. Edit the node definition in: {config_file.relative_to(project_path)}")
        click.echo(f"2. Build the project: colcon build")
        
    except Exception as e:
        click.echo(f"Error: Failed to create node: {str(e)}", err=True)
        sys.exit(1)

@main.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              default=None, help='Output directory for generated files')
@click.option('--force', '-f', is_flag=True, help='Overwrite existing files')
@click.option('--debug', '-d', is_flag=True, help='Enable debug output during parsing')
def generate(input_file: Path, output_dir: Optional[Path], force: bool, debug: bool) -> None:
    """
    Generate C++ code from a RoboDSL file.
    
    This command processes a .robodsl file and generates the corresponding
    C++ source files, headers, and build configuration.
    """
    try:
        from robodsl.parsers import parse_robodsl
        from robodsl.generators.main_generator import MainGenerator
        
        # Set default output directory if not specified
        if output_dir is None:
            output_dir = input_file.parent
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Processing {input_file}...")
        
        # Parse the input file
        config = parse_robodsl(input_file.read_text(), debug=debug)
        
        # Generate code
        generator = MainGenerator(output_dir=str(output_dir), debug=debug)
        generated_files = generator.generate(config)
        
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

@main.command()
@click.argument('project_dir', default='.')
def build(project_dir: str) -> None:
    """Build the RoboDSL project."""
    project_dir = Path(project_dir).resolve()
    click.echo(f"Building project in {project_dir}...")
    
    # TODO: Implement build logic
    click.echo("Build command not yet implemented")

@main.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
              default=None, help='Output directory for generated files')
def create_launch_file(input_file: Path, output_dir: Optional[Path]) -> None:
    """Create a launch file for a RoboDSL node.
    
    This generates a ROS2 launch file based on the node definition in the RoboDSL file.
    """
    try:
        from robodsl.parsers import parse_robodsl
        
        # Set default output directory if not specified
        if output_dir is None:
            output_dir = input_file.parent / 'launch'
            
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        click.echo(f"Creating launch file for {input_file}...")
        
        # Parse the input file
        config = parse_robodsl(input_file.read_text())
        
        # TODO: Implement launch file generation based on parsed config
        # For now, create a basic launch file
        node_name = input_file.stem
        launch_file = output_dir / f"{node_name}.launch.py"
        
        with open(launch_file, 'w') as f:
            f.write(f"""from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='{node_name}',
            executable='{node_name}_node',
            name='{node_name}',
            output='screen',
            emulate_tty=True,
        )
    ])
""")
        
        click.echo(f"Created launch file: {launch_file}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)

@main.command()
@click.argument('project_dir', default='.')
def check_dependencies(project_dir: str) -> None:
    """Check dependencies for the RoboDSL project."""
    project_dir = Path(project_dir).resolve()
    click.echo(f"Checking dependencies in {project_dir}...")
    
    checker = DependencyChecker()
    if checker.check_all_dependencies():
        click.echo("All dependencies are satisfied.")
    else:
        click.echo(checker.get_install_instructions())

if __name__ == "__main__":
    main() 