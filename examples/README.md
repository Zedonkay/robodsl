# RoboDSL Examples

This directory contains example projects demonstrating various features of RoboDSL, including ROS2 integration, CUDA acceleration, and advanced robotics patterns.

## Table of Contents
1. [Basic Examples](#basic-examples)
   - [Simple Publisher/Subscriber](#1-simple-publishersubscriber)
   - [ROS2 Parameters](#2-ros2-parameters)
2. [Intermediate Examples](#intermediate-examples)
   - [Lifecycle Node](#1-lifecycle-node)
   - [Custom Message Types](#2-custom-message-types)
3. [Advanced Examples](#advanced-examples)
   - [CUDA Acceleration](#1-cuda-acceleration)
   - [Action Server](#2-action-server)
   - [Component Node](#3-component-node)
4. [Building and Running](#building-and-running)
5. [Contributing](#contributing)

## Basic Examples

### 1. Simple Publisher/Subscriber

#### Overview
Demonstrates basic ROS2 communication with a publisher and subscriber.

#### Key Features
- Topic-based communication
- QoS configuration
- Namespace and remapping support

#### Files
- `basic_pubsub/`
  - `pubsub.robodsl` - Main DSL definition
  - `CMakeLists.txt` - Build configuration
  - `package.xml` - Package manifest

### 2. ROS2 Parameters

#### Overview
Shows how to declare and use parameters in RoboDSL nodes.

#### Key Features
- Parameter declaration with types and defaults
- Dynamic parameter updates
- Parameter validation

## Intermediate Examples

### 1. Lifecycle Node

#### Overview
Implements a managed lifecycle node with state transitions.

#### Key Features
- Lifecycle state management
- State transition callbacks
- Error handling and recovery

### 2. Custom Message Types

#### Overview
Demonstrates defining and using custom message types.

#### Key Features
- Custom message definition
- Nested message types
- Arrays and complex types

## Advanced Examples

### 1. CUDA Acceleration

#### Overview
Shows how to integrate CUDA kernels with ROS2 nodes.

#### Key Features
- CUDA kernel definition
- Host-device memory management
- Thrust algorithm integration

#### Files
- `cuda_acceleration/`
  - `vector_ops.robodsl` - Main DSL file
  - `kernels/` - CUDA kernel implementations
  - `CMakeLists.txt` - CUDA build configuration

### 2. Action Server

#### Overview
Implements a ROS2 action server for long-running operations.

#### Key Features
- Action definition
- Goal, feedback, and result handling
- Cancellation support

### 3. Component Node

#### Overview
Demonstrates component-based node composition.

#### Key Features
- Runtime composition
- Component lifecycle
- Inter-component communication

## Building and Running

### Prerequisites
- RoboDSL installed and in your PATH
- ROS2 Humble or newer
- CUDA Toolkit 11.0+ (for CUDA examples)
- CMake 3.15+

### Building an Example

1. Navigate to the example directory:
   ```bash
   cd examples/<example_name>
   ```

2. Generate the code:
   ```bash
   robodsl generate <example_name>.robodsl
   ```

3. Build the project:
   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

### Running an Example

1. Source your ROS2 environment:
   ```bash
   source /opt/ros/$ROS_DISTRO/setup.bash
   ```

2. Run the example:
   ```bash
   ./<example_name>
   ```

   Or for ROS2 nodes:
   ```bash
   ros2 run <package_name> <node_name>
   ```

## Example: CUDA Vector Operations

### Building and Running

```bash
cd examples/cuda_vector_ops
robodsl generate vector_ops.robodsl
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./vector_ops
```

### Expected Output

```
[INFO] [vector_ops]: Initializing vector operations...
[INFO] [vector_ops]: Running vector addition...
[INFO] [vector_ops]: Result: [3.0, 5.0, 7.0, 9.0, 11.0]
[INFO] [vector_ops]: Running vector multiplication...
[INFO] [vector_ops]: Result: [2.0, 4.0, 6.0, 8.0, 10.0]
[INFO] [vector_ops]: Running vector reduction with Thrust...
[INFO] [vector_ops]: Sum: 15.0
```

## Conditional Compilation

Examples demonstrate how to use conditional compilation flags:

```python
# In your .robodsl file
build_options = {
    "ENABLE_ROS2": true,
    "ENABLE_CUDA": true,
    "CUDA_ARCH": "sm_75"  # Specify your GPU architecture
}

# Conditional code blocks
#ifdef ENABLE_CUDA
cuda_kernel my_kernel {
    # Kernel definition
}
#endif
```

## Contributing

We welcome contributions to expand our examples! Please follow these guidelines:

1. Create a new directory under `examples/` for your example
2. Include a detailed README.md with:
   - Example description
   - Prerequisites
   - Build and run instructions
   - Expected output
3. Keep the example focused and well-documented
4. Follow the existing code style

## License

All examples are part of the RoboDSL project and are licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **CUDA Not Found**
   - Ensure CUDA Toolkit is installed
   - Set `CUDA_TOOLKIT_ROOT_DIR` if not in default location

2. **ROS2 Packages Not Found**
   - Source your ROS2 environment
   - Install missing dependencies with `rosdep`

3. **Build Failures**
   - Check CMake output for missing dependencies
   - Ensure all submodules are initialized

For additional help, please open an issue on our [GitHub repository](https://github.com/yourusername/robodsl).
