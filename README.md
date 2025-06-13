# RoboDSL

A domain-specific language and compiler for GPU-accelerated robotics applications with ROS2 and CUDA.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![ROS2 Version](https://img.shields.io/badge/ROS2-Humble%20%7C%20Foxy%20%7C%20Galactic-blueviolet)](https://docs.ros.org/)

RoboDSL simplifies the development of GPU-accelerated robotics applications by providing a clean, declarative syntax for defining ROS2 nodes and CUDA kernels, while handling all the boilerplate code and build system configuration.

## Overview

RoboDSL simplifies the development of GPU-accelerated robotics applications by providing:

- A simple DSL for defining ROS2 nodes and CUDA kernels
- Automatic code generation for both C++ and Python
- Build system integration
- Project scaffolding

## ✨ Features

- **🚀 Easy Node Definition**
  - Define ROS2 nodes with publishers, subscribers, services, and parameters
  - Support for both C++ and Python nodes
  - Nested node organization with dot notation (e.g., `sensors.camera`)

- **⚡ CUDA Integration**
  - Seamless CUDA kernel integration with ROS2 nodes
  - Automatic memory management between CPU and GPU
  - Optimized data transfer patterns

- **🔧 Project Management**
  - Project initialization and scaffolding
  - Add new nodes to existing projects
  - Automatic file organization

- **🛠 Build System**
  - Automatic CMake configuration
  - ROS2 package generation
  - CUDA compilation support

- **🧪 Testing & Validation**
  - Comprehensive test suite
  - Input validation
  - Helpful error messages

## 🚀 Installation

### Prerequisites
- Python 3.8+
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

## 🚀 Quick Start

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
robodsl build

# Source the workspace
source install/setup.bash

# Run a node
ros2 run my_robot_project my_node
```

### 4. Launch Files
Launch files are automatically generated for each node:
```bash
ros2 launch my_robot_project sensors.camera.launch.py
```

## 📝 DSL Syntax

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

## 📁 Project Structure

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

## 🤝 Contributing

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

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [ROS2 Documentation](https://docs.ros.org/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [CMake Documentation](https://cmake.org/documentation/)

## 🙏 Acknowledgments

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
