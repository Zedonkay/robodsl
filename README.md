# RoboDSL

A domain-specific language and compiler for GPU-accelerated robotics applications with ROS2 and CUDA.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ROS2 Version](https://img.shields.io/badge/ROS2-Humble%20%7C%20Foxy%20%7C%20Galactic-blueviolet)](https://docs.ros.org/)

[![Build Status](https://github.com/yourusername/robodsl/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/robodsl/actions)
[![Code Coverage](https://codecov.io/gh/yourusername/robodsl/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/robodsl)

RoboDSL is a powerful framework that simplifies the development of GPU-accelerated robotics applications by providing a clean, declarative syntax for defining ROS2 nodes and CUDA kernels. It handles all the boilerplate code and build system configuration, allowing developers to focus on algorithm development and system integration.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Installation Methods](#installation-methods)
- [Quick Start](#quick-start)
- [Documentation](#documentation)

- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

RoboDSL is designed to bridge the gap between robotic algorithm development and high-performance computing. It provides a clean, intuitive syntax for defining both the computational graph of a robotics application and the GPU-accelerated algorithms that power it.

### Core Concepts

- **Declarative Syntax**: Define your robotics application's structure and behavior in a human-readable format
- **Automatic Code Generation**: Generate optimized C++ and CUDA code from high-level specifications
- **ROS2 Integration**: Seamless integration with the Robot Operating System 2 (ROS2) ecosystem
- **GPU Acceleration**: First-class support for CUDA and Thrust for high-performance computing
- **Build System**: Automatic CMake configuration for easy compilation and deployment

### Architecture

RoboDSL follows a modular architecture:

1. **DSL Parser**: Parses the RoboDSL syntax into an abstract syntax tree (AST)
2. **Code Generator**: Converts the AST into executable code (C++, CUDA, CMake)
3. **Build System**: Manages dependencies and compilation of generated code
4. **Runtime**: Provides necessary libraries and utilities for execution

## Key Features

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
project "MyRobot"

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

## Documentation

Comprehensive documentation is available in the `docs/` directory:

- [Developer Guide](docs/developer_guide.md) - Architecture and development workflow
- [DSL Specification](docs/dsl_specification.md) - Complete language reference
- [Contributing](docs/contributing.md) - How to contribute to the project
- [Code of Conduct](docs/code_of_conduct.md) - Community guidelines
- [Changelog](docs/changelog.md) - Project history and changes

Documentation website coming soon.



## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.md) for details.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run linters
flake8 src/
mypy src/
```

## License

RoboDSL is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

- ROS2 Community
- NVIDIA for CUDA
- The Thrust Development Team
- All our contributors

## Support

### Getting Help

If you encounter any issues or have questions:

1. Check the [FAQ](docs/FAQ.md)
2. Search the [issue tracker](https://github.com/yourusername/robodsl/issues)
3. Open a new issue if your problem isn't addressed

### Community

- [GitHub Discussions](https://github.com/yourusername/robodsl/discussions)
- [ROS Discourse](https://discourse.ros.org/)
- [ROS Answers](https://answers.ros.org/)

## Roadmap

### Upcoming Features

- Support for more ROS2 message types
- Enhanced visualization tools
- Performance profiling
- Additional language bindings
- Cloud deployment support

### Version History

See [CHANGELOG.md](CHANGELOG.md) for a detailed history of changes.

---

RoboDSL is developed with ❤️ by the robotics community.
- ROS2 (Humble, Foxy, or Galactic)
- CUDA Toolkit (for GPU acceleration)
- CMake 3.15+

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/robodsl.git
   cd robodsl
   ```

2. **Set up a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the package in development mode**
   ```bash
   pip install -e .
   ```

4. **Verify installation**
   ```bash
   robodsl --version
   robodsl --help
   ```

## Building with CUDA and Thrust

### Prerequisites
- CUDA Toolkit 11.0 or later
- CMake 3.15 or later
- C++17 compatible compiler
- (Optional) Thrust library (usually included with CUDA)

### Building the Project

1. **Create a build directory**
   ```bash
   mkdir -p build && cd build
   ```

2. **Configure with CMake**
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_THRUST=ON
   ```
   
   Options:
   - `-DENABLE_THRUST=ON/OFF`: Enable/disable Thrust support (default: ON)
   - `-DCMAKE_CUDA_ARCHITECTURES="75"`: Specify target GPU architecture (e.g., 75 for Turing, 86 for Ampere)

3. **Build the project**
   ```bash
   cmake --build . --config Release -- -j$(nproc)
   ```

4. **Install (optional)**
   ```bash
   sudo cmake --install .
   ```

## Quick Start

### 1. Initialize a New Project
```bash
# Create a new project directory
robodsl init my_robot_project
cd my_robot_project
```

### 2. Add a Node
Add a simple Python node:
```bash
robodsl add-node my_node --language python
```

Add a C++ node with publishers and subscribers:
```bash
robodsl add-node sensors.camera \
  --publisher /camera/image sensor_msgs/msg/Image \
  --subscriber /camera/control std_msgs/msg/Bool \
  --language cpp
```

### 3. Build and Run
```bash
# Build the project
colcon build --packages-select my_robot_project --cmake-args -DCMAKE_BUILD_TYPE=Release -DENABLE_THRUST=ON
source install/setup.bash
```

### 4. Launch Files
Launch files are automatically generated for each node:
```bash
ros2 launch my_robot_project sensors.camera.launch.py
```

## DSL Syntax

### Node Definition
```ruby
# Basic node with publishers and subscribers
node my_node {
    # Publishers
    publisher /output_topic std_msgs/msg/String "Hello, World!"
    
    # Subscribers
    subscriber /input_topic std_msgs/msg/String
    
    # Parameters with default values
    parameter my_param 42
    parameter use_gpu true
    
    # Services
    service /process_image example_interfaces/srv/Trigger
}

# Nested node structure
node sensors.camera {
    publisher /camera/image sensor_msgs/msg/Image
    parameter frame_id "camera_frame"
}
```

### CUDA Kernel Definition (Coming Soon)
```ruby
kernel image_processor {
    # Input and output specifications
    input Image input_image
    output Image output_image
    
    # Kernel configuration
    block_size 256
    grid_size "ceil(input_image.width * input_image.height / block_size)"
    
    # Kernel code (CUDA C++)
    code """
    __global__ void process(float* input, float* output, int width, int height) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < width * height) {
            // Your CUDA kernel code here
            output[idx] = input[idx] * 2.0f;
        }
    }
    """
}
```

## Project Structure

```
my_robot_project/
├── src/                    # Source files
│   ├── my_node.py          # Python node implementation
│   └── my_node.cpp         # C++ node implementation
│
├── include/                # C++ header files
│   └── my_node/            # Node-specific headers
│       └── my_node.hpp     # C++ node header
│
├── launch/                 # Launch files
│   ├── my_node.launch.py   # Node launch file
│   └── sensors.camera.launch.py
│
├── config/                 # Configuration files
│   ├── my_node.yaml        # Node parameters
│   └── sensors.camera.yaml
│
├── robodsl/                # RoboDSL configuration
│   └── nodes/              # Node definitions
│       ├── my_node.robodsl
│       └── sensors/
│           └── camera.robodsl
│
├── CMakeLists.txt         # Build configuration
└── package.xml            # ROS2 package definition
```

## Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a **Pull Request**

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Run tests:
   ```bash
   pytest tests/ -v
   ```

3. Run linters:
   ```bash
   black .
   flake8
   mypy .
   ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Resources

- [ROS2 Documentation](https://docs.ros.org/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CMake Documentation](https://cmake.org/documentation/)

## Acknowledgments

- The ROS2 and CUDA communities for their amazing tools and libraries
- All contributors who have helped improve RoboDSL

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
