# RoboDSL

A domain-specific language and compiler for GPU-accelerated robotics applications with ROS2 and CUDA.

## Overview

RoboDSL simplifies the development of GPU-accelerated robotics applications by providing:

- A simple DSL for defining ROS2 nodes and CUDA kernels
- Automatic code generation for both C++ and Python
- Build system integration
- Project scaffolding

## Features

- **Easy Node Definition**: Define ROS2 nodes with publishers, subscribers, and parameters using a clean syntax
- **CUDA Integration**: Seamlessly integrate CUDA kernels with ROS2 nodes
- **Multi-language Support**: Generate both C++ and Python nodes
- **Project Management**: Initialize new projects and add nodes to existing ones
- **Build System**: Automatic CMake configuration

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/robodsl.git
   cd robodsl
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

## Quick Start

1. Initialize a new project:
   ```bash
   robodsl init my_robot_project
   cd my_robot_project
   ```

2. Add a new node:
   ```bash
   robodsl add-node sensor_processor --publisher /output sensor_msgs/msg/Image --subscriber /input sensor_msgs/msg/Image
   ```

3. Build the project:
   ```bash
   robodsl build
   ```

## DSL Syntax

### Node Definition
```
node my_node {
    publisher /output_topic std_msgs/msg/String
    subscriber /input_topic std_msgs/msg/String
    parameter my_param 42
}
```

### CUDA Kernel Definition
```
kernel my_kernel {
    input float* input
    output float* output
    block_size 256
}
```

## Project Structure

```
my_robot_project/
├── src/                    # Source files
├── include/                # Header files (C++)
├── launch/                 # Launch files
├── config/                 # Configuration files
└── robodsl/                # RoboDSL configuration
    └── nodes/              # Node definitions
```

## Contributing

1. Fork the repository
2. Create a new branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/my-feature`
5. Open a pull request

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
