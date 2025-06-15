# RoboDSL

A domain-specific language and compiler for GPU-accelerated robotics applications with ROS2 and CUDA integration.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ROS2 Version](https://img.shields.io/badge/ROS2-Humble%20%7C%20Foxy%20%7C%20Galactic-blueviolet)](https://docs.ros.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.0%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)
[![Build Status](https://github.com/yourusername/robodsl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/robodsl/actions)
[![Code Coverage](https://codecov.io/gh/yourusername/robodsl/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/robodsl)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

RoboDSL is a powerful framework that simplifies the development of GPU-accelerated robotics applications by providing a clean, declarative syntax for defining ROS2 nodes and CUDA kernels. It handles all the boilerplate code and build system configuration, allowing developers to focus on algorithm development and system integration.

## Features

- **ROS2 Integration**: Full support for ROS2 nodes, lifecycle management, and communication patterns
- **CUDA Acceleration**: Seamless GPU integration with automatic memory management
- **Declarative Syntax**: Define complex robotics applications with clean, readable code
- **Build System**: Automatic CMake configuration for easy compilation and deployment
- **Extensible**: Custom node types, message types, and code generation templates

## Table of Contents

- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Methods](#installation-methods)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Examples](#examples)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites

- **Python**: 3.8 or higher
- **ROS2**: Humble, Foxy, or Galactic (for ROS2 features)
- **CUDA Toolkit**: 11.0 or higher (for GPU acceleration)
- **CMake**: 3.15 or higher
- **pip**: Latest version

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

Comprehensive documentation is available in the [docs](docs/) directory:

- [DSL Specification](docs/dsl_specification.md) - Complete reference of the RoboDSL language
- [Developer Guide](docs/developer_guide.md) - Architecture, development workflow, and extension points
- [Examples](examples/README.md) - Tutorials and example projects
- [Contributing](docs/contributing.md) - How to contribute to the project
- [Changelog](docs/changelog.md) - Release notes and version history

## Examples

Explore the [examples](examples/) directory for complete, runnable examples:

- **Basic Examples**: Simple publisher/subscriber patterns
- **Intermediate**: Lifecycle nodes, custom message types
- **Advanced**: CUDA acceleration, action servers, component nodes

Each example includes a README with build and run instructions.

## Development

To set up a development environment:

```bash
# Clone the repository
git clone https://github.com/yourusername/robodsl.git
cd robodsl

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

Run the test suite:

```bash
pytest
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for details on how to get started.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

Please ensure your code follows our style guidelines and includes appropriate tests.

## License

RoboDSL is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

RoboDSL builds upon several amazing open source projects:

- [ROS2](https://docs.ros.org/) - Robot Operating System 2
- [CUDA](https://developer.nvidia.com/cuda-toolkit) - NVIDIA's parallel computing platform
- [Thrust](https://github.com/NVIDIA/thrust) - Parallel algorithms library
- [Jinja2](https://palletsprojects.com/p/jinja/) - Templating engine
- [Click](https://click.palletsprojects.com/) - Command line interface creation

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

## Examples

The `examples/` directory contains practical examples demonstrating RoboDSL features:

### Basic Examples

- [Simple Publisher/Subscriber](examples/basic_pubsub/) - Basic ROS2 communication
- [Parameter Server](examples/parameters/) - Working with runtime parameters
- [Lifecycle Node](examples/lifecycle_node/) - Managing node states

### Advanced Examples

- [CUDA Vector Operations](examples/cuda_vector_ops/) - Basic GPU acceleration
- [Image Processing Pipeline](examples/image_processing/) - Computer vision with CUDA
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

MIT
