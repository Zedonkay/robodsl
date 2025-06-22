# RoboDSL

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ROS2 Version](https://img.shields.io/badge/ROS2-Humble%20%7C%20Foxy%20%7C%20Galactic-blueviolet)](https://docs.ros.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)

A DSL for building GPU-accelerated robotics applications with ROS2 and CUDA.

## Quick Start

### Prerequisites
- Linux/Windows (WSL2) with Python 3.8+
- ROS2 (Humble/Foxy/Galactic)
- CUDA 11.0+ (for GPU acceleration)
- Build tools (CMake, make, g++)

### Install
```bash
git clone https://github.com/yourusername/robodsl.git
cd robodsl
# Choose one:
./build-deb-direct.sh && sudo dpkg -i robodsl_0.1.0_all.deb  # Debian
# OR
pip install -e .  # Development
# OR
./build-with-minimal-docker.sh  # Docker
```

## Features

- **DSL**: Clean syntax for ROS2 nodes and CUDA kernels
- **Performance**: GPU acceleration with automatic memory management
- **Modular**: Component-based architecture
- **Cross-Platform**: Linux/Windows/macOS support
- **ROS2**: Lifecycle nodes, QoS, namespacing
- **Build**: Automatic CMake and dependency handling

## Documentation

- [DSL Specification](docs/dsl_specification.md)
- [ROS2 Integration](docs/ros2_integration.md)
- [CUDA Acceleration](docs/cuda_acceleration.md)
- [Examples](examples/)

## Development

```bash
git clone https://github.com/yourusername/robodsl.git
cd robodsl
pip install -e .  # Development install
```

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT - See [LICENSE](LICENSE)
- [CMake](https://cmake.org/) - Build system generator
- [Click](https://click.palletsprojects.com/) - Command line interface creation

- **Python**: 3.8 or higher
- **ROS2**: Humble, Foxy, or Galactic (for ROS2 features)
- **CUDA Toolkit**: 11.0 or higher (for GPU acceleration)
- **CMake**: 3.15 or higher
- **pip**: Latest version

### Installation Methods

#### From PyPI (Recommended) - Upcoming 

```bash
pip install robodsl
```

#### Planned release using apt also upcoming

#### From Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/robodsl.git
   cd robodsl
   ```

2. Install in development mode:
   ```bash
   pip install -e ".[dev]"
   ```

3. Verify installation:
   ```bash
   robodsl --version
   ```

## Quick Start

1. Create a new RoboDSL project:
   ```bash
   mkdir my_robodsl_project
   cd my_robodsl_project
   robodsl init
   ```

2. Edit the generated `.robodsl` file to define your nodes and components

3. Generate the code:
   ```bash
   robodsl generate
   ```

4. Build and run your application:
   ```bash
   mkdir -p build && cd build
   cmake ..
   make
   ./my_robodsl_app
   ```

For more detailed examples, see the [examples](examples/) directory.

## Documentation

### Getting Started
- [Installation Guide](docs/installation.md) - How to install and set up RoboDSL
- [Quick Start](docs/quickstart.md) - Build your first RoboDSL application
- [Tutorials](docs/tutorials/) - Step-by-step guides for common tasks

### Core Concepts
- [DSL Specification](docs/dsl_specification.md) - Complete language reference
- [Node Types](docs/nodes.md) - Different node types and their use cases
- [Lifecycle Management](docs/lifecycle.md) - Managing node states and transitions
- [Communication Patterns](docs/communication.md) - Publishers, subscribers, services, and actions

### Advanced Topics
- [GPU Acceleration](docs/gpu_acceleration.md) - CUDA integration and optimization
- [Performance Tuning](docs/performance.md) - Optimizing your RoboDSL applications
- [Build System](docs/build_system.md) - Customizing the build process
- [Deployment](docs/deployment.md) - Packaging and deploying applications

### Reference
- [Standard Library](docs/stdlib.md) - Built-in functions and utilities
- [API Reference](docs/api/) - Detailed API documentation
- [Configuration Reference](docs/configuration.md) - All configuration options
- [FAQ](docs/faq.md) - Common questions and answers

### Development
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to RoboDSL
- [Developer Guide](docs/developer_guide.md) - Architecture and extension points
- [Release Notes](CHANGELOG.md) - Version history and changes
- [Roadmap](ROADMAP.md) - Upcoming features and improvements

## Project Structure

```
robodsl/
├── .github/                # GitHub Actions workflows and templates
│   ├── workflows/          # CI/CD pipelines
│   └── ISSUE_TEMPLATE/     # Issue templates
│
├── cmake/                 # CMake configuration
│   ├── FindCUDA.cmake     # CUDA toolchain setup
│   ├── RoboDSLConfig.cmake
│   └── ...
│
├── docs/                  # Comprehensive documentation
│   ├── api/               # API reference
│   ├── examples/          # Example documentation
│   ├── images/            # Documentation assets
│   ├── advanced/          # Advanced topics
│   ├── guides/            # Tutorials and how-tos
│   ├── conf.py            # Sphinx configuration
│   └── ...
│
├── examples/              # Example projects
│   ├── basic/            # Beginner examples
│   │   ├── pubsub/       # Publisher/subscriber pattern
│   │   ├── parameters/   # Parameter handling
│   │   └── services/     # Service client/server
│   │
│   ├── intermediate/   # Intermediate examples
│   │   ├── lifecycle/    # Lifecycle nodes
│   │   ├── custom_msgs/  # Custom message types
│   │   └── actions/      # Action servers/clients
│   │
│   └── advanced/      # Advanced use cases
│       ├── cuda/         # CUDA acceleration
│       ├── components/   # Reusable components
│       └── multi_node/   # Complex systems
│
├── include/              # Public headers
│   └── robodsl/
│       ├── core/      # Core functionality
│       ├── nodes/      # Node implementations
│       └── utils/      # Utility functions
│
├── src/                  # Source code
│   ├── robodsl/         # Core implementation
│   │   ├── parser/      # Language parser
│   │   ├── generator/   # Code generation
│   │   ├── nodes/       # Node implementations
│   │   └── utils/       # Utility functions
│   └── ...
│
├── templates/            # Code generation templates
│   ├── cpp/             # C++ templates
│   └── python/          # Python templates
│
├── tests/               # Test suite
│   ├── unit/           # Unit tests
│   ├── integration/     # Integration tests
│   └── performance/     # Performance benchmarks
│
├── .clang-format       # Code style configuration
├── .gitignore           # Git ignore rules
├── CMakeLists.txt       # Root CMake configuration
├── pyproject.toml       # Python package configuration
├── README.md            # This file
├── CHANGELOG.md         # Version history
└── ROADMAP.md           # Future development plans
```

## Contributing

We welcome contributions from the community! Here's how you can help:

### How to Contribute

1. **Report Bugs**
   - Search existing issues to avoid duplicates
   - File a detailed bug report with reproduction steps
   - Include version information and environment details

2. **Suggest Features**
   - Check the roadmap for planned features
   - Open an issue to discuss your idea
   - Be ready to help with implementation

3. **Submit Code**
   - Fork the repository
   - Create a feature branch
   - Write tests for your changes
   - Submit a pull request
   - Address code review feedback

4. **Improve Documentation**
   - Fix typos and clarify explanations
   - Add missing documentation
   - Improve examples
   - Translate documentation

### Development Setup

We're grateful to the open source community for their contributions and support.

### ROS2 Node Development

- **Node Definition**: Define ROS2 nodes with publishers, subscribers, services, and parameters
- **Message Generation**: Automatic generation of custom message types
- **Component Support**: Create reusable node components
- **Lifecycle Management**: Built-in support for ROS2 lifecycle nodes

### GPU Acceleration

- **CUDA Integration**: Seamless CUDA kernel integration with ROS2 nodes
- **Thrust Support**: High-level parallel algorithms library integration
- **Memory Management**: Automatic host-device memory management
- **Optimized Transfers**: Efficient host-device memory strategies enabling full GPU acceleration

### Build System

- **Cross-Platform**: Supports Linux, Windows (WSL2), and macOS
- **Dependency Management**: Automatic handling of ROS2 and CUDA dependencies
- **Build Configurations**: Multiple build types (Debug, Release, RelWithDebInfo)
- **Testing Integration**: Built-in support for unit and integration tests

### Developer Experience

- **Code Completion**: IDE support for RoboDSL syntax
- **Error Reporting**: Clear and actionable error messages
- **Documentation Generation**: Automatic API documentation
- **Debugging Support**: Integrated debugging tools for both CPU and GPU code

## Installation

### Prerequisites

Before installing RoboDSL, ensure you have the following installed:

- **Operating System**: Linux (recommended), Windows (WSL2), or macOS
- **Python**: 3.8 or higher
- **ROS2**: Humble, Foxy, or Galactic (recommended: Humble)
- **CUDA Toolkit**: 11.0 or higher (for GPU acceleration)
- **CMake**: 3.16 or higher
- **Git**: For version control

### Installation Methods

#### From PyPI (Recommended)

```bash
pip install robodsl
```

#### From Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/robodsl.git
cd robodsl
```

2. Install in development mode:
```bash
pip install -e .
```

3. Verify installation:
```bash
robodsl --version
```

#### Docker

A Docker image is available for development and testing:
```bash
docker pull yourusername/robodsl:latest
```

## Quick Start

### Create a New Project

```bash
robodsl new my_robot_project
cd my_robot_project
```

### Define Your First Node

Create a file named `my_node.robodsl`:

```python
project "MyRobo"

# Example of a lifecycle node with QoS and namespaces
lifecycle_node navigation_node {
    # Node configuration
    namespace = "robot1"
    remap = {
        "cmd_vel": "cmd_vel_nav"
    }
    
    # QoS configuration for publishers
    publishers = [
        {
            name: "path",
            type: "nav_msgs/msg/Path",
            qos: {
                depth: 10,
                reliability: "reliable",
                durability: "transient_local"
            }
        }
    ]
    
    # Subscriber with custom QoS
    subscribers = [
        {
            topic: "odom",
            type: "nav_msgs/msg/Odometry",
            qos: { reliability: "best_effort" }
        }
    ]
    
    # Action server with CUDA offloading
    actions = [
        {
            name: "navigate_to_pose",
            action_type: "nav2_msgs/action/NavigateToPose"
        }
    ]
    
    # Enable parameter callbacks
    parameter_callbacks = true
    
    # CUDA kernel for path planning
    cuda_kernels = ["plan_path"]
}

# CUDA kernel definition
cuda_kernel plan_path {
    input {
        start_pose: geometry_msgs/msg/Pose
        goal_pose: geometry_msgs/msg/Pose
        map: nav_msgs/msg/OccupancyGrid
    }
    output {
        path: nav_msgs/msg/Path
        cost: float
    }
    code = """
    // CUDA kernel implementation
    __global__ void plan_kernel(...) {
        // GPU-accelerated path planning
    }
    """
}

# Example of a regular node
node my_robot_node {
    # Define publishers
    publishers = [
        { name: "odom", type: "nav_msgs/msg/Odometry" },
        { name: "scan", type: "sensor_msgs/msg/LaserScan" }
    ]
    
    # Define subscribers
    subscribers = [
        { name: "cmd_vel", type: "geometry_msgs/msg/Twist" }
    ]
    
    # Define parameters
    parameters = {
        "max_speed": 1.0,
        "publish_rate": 30.0,
        "use_sim_time": false
    }
}
```

### Add a CUDA Kernel

```python
cuda_kernel process_lidar {
    inputs = [
        { name: "input_scan", type: "sensor_msgs::msg::LaserScan::ConstPtr" },
        { name: "params", type: "const LidarParams&" }
    ]
    outputs = [
        { name: "obstacles", type: "std::vector<Obstacle>&" }
    ]
    
    block_size = [256, 1, 1]
    grid_size = [1, 1, 1]
    shared_mem_bytes = 4096
    
    includes = [
        "sensor_msgs/msg/laser_scan.hpp",
        "vector"
    ]
    
    code = """
    __global__ void process_lidar_kernel(
        const float* ranges, int num_points,
        const LidarParams* params,
        Obstacle* obstacles, int* num_obstacles) {
        // Your CUDA kernel code here
    }
    """
}
```

### Build and Run

```bash
# Generate code
robodsl generate my_node.robodsl

# Build the project
mkdir -p build && cd build
cmake ..
make

# Run the node
./bin/my_robot_node
```

## Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](docs/contributing.md) for details on how to contribute to this project. This includes information about:

- Reporting issues
- Feature requests
- Code contributions
- Documentation improvements
- Community support

## Code of Conduct

This project adheres to a [Code of Conduct](docs/code_of_conduct.md) to ensure a welcoming and inclusive environment for all contributors. By participating, you are expected to uphold this code.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For questions and support, please:
1. Check the [documentation](#documentation)
2. Search the [issue tracker](https://github.com/yourusername/robodsl/issues)
3. Open a new issue if your question hasn't been answered

## Acknowledgements

RoboDSL builds upon these amazing open source projects:

- [ROS2](https://docs.ros.org/) - Robot Operating System
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) - GPU-accelerated computing
- [Python](https://www.python.org/) - Programming language
- [Jinja2](https://palletsprojects.com/p/jinja/) - Templating engine
- [CMake](https://cmake.org/) - Build system generator
- [Click](https://click.palletsprojects.com/) - Command line interface creation

Special thanks to all our contributors and the open source community for their support and feedback.
<!-- 
## Examples

Explore our collection of examples to see RoboDSL in action:

### Beginner Examples
- [Basic Publisher/Subscriber](examples/basic/pubsub) - Simple message passing
- [Parameter Server](examples/basic/parameters) - Dynamic configuration
- [Service Client/Server](examples/basic/services) - Request/response pattern

### Intermediate Examples
- [Lifecycle Node](examples/intermediate/lifecycle) - Managed node states
- [Custom Messages](examples/intermediate/custom_msgs) - Defining and using custom messages
- [Action Server/Client](examples/intermediate/actions) - Long-running operations

### Advanced Examples
- [CUDA-Accelerated Image Processing](examples/advanced/cuda_image_processing) - GPU-accelerated computer vision
- [Multi-Node System](examples/advanced/multi_node) - Complex system composition
- [Custom Components](examples/advanced/custom_components) - Building reusable components

### Real-world Applications
- [Autonomous Navigation](examples/real_world/navigation) - SLAM and path planning
- [Robot Arm Control](examples/real_world/robotics_arm) - Kinematics and control
- [Sensor Fusion](examples/real_world/sensor_fusion) - Combining multiple sensor inputs
- [Neural Network Inference](examples/nn_inference/) - Deep learning integration

Each example includes:
- A `.robodsl` source file
- Build and run instructions
- Expected output

To run an example:

```bash
cd examples/desired_example
robodsl generate example.robodsl
mkdir -p build && cd build
cmake ..
make
./example_node
```

For detailed instructions, refer to each example's README file.
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Features (Planned)

- Declarative DSL for defining robotics applications
- Automatic ROS2 node generation
- CUDA kernel management
- Build system generation
- GPU profiling integration

## Installation

```bash
# Install in development mode
pip install -e .
```

## Usage

```bash
# Initialize a new project
robodsl init my_robot_project

# Build the project
cd my_robot_project
robodsl build
```

## Development

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests:
   ```bash
   pytest
   ```

## License

MIT -->
