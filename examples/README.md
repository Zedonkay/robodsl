# 📚 RoboDSL Examples

Welcome to the RoboDSL examples directory! This collection demonstrates how to leverage RoboDSL's powerful features through practical, runnable examples. Each example is designed to showcase specific capabilities while following best practices for ROS2 and CUDA development.

## 🎯 Getting Started

### Prerequisites
- RoboDSL installed (see main [README](../README.md) for installation instructions)
- ROS2 Humble or later
- CUDA Toolkit (for GPU-accelerated examples)
- Basic familiarity with ROS2 concepts

### How to Use These Examples
1. Navigate to any example directory
2. Follow the README for build and run instructions
3. Experiment with the code to understand how it works
4. Use as a reference for your own projects

## 🏆 Featured Examples

### 🚀 Comprehensive Example

**File**: [comprehensive_example.robodsl](comprehensive_example.robodsl)

A complete showcase of RoboDSL features in a single, well-documented example. Perfect for understanding how different components work together.

**Key Features**:
- Complete lifecycle node implementation
- Custom QoS profiles for all communication
- Parameter server integration
- CUDA-accelerated processing
- Namespace and remapping examples
- Error handling and recovery

### 🤖 MPPI Robot Controller

**Directory**: [mppi_robot/](mppi_robot/)

An advanced implementation of a Model Predictive Path Integral (MPPI) controller for robot navigation, demonstrating real-world usage patterns.

**Key Features**:
- Advanced ROS2 node configuration
- Parameter server with validation
- Python node implementation
- Launch file configuration
- Performance optimization techniques

## Example Files

### 1. Comprehensive Example
- **File**: [comprehensive_example.robodsl](comprehensive_example.robodsl)
- **Description**: A complete example showcasing most RoboDSL features including lifecycle nodes, QoS configuration, namespacing, and CUDA offloading.
- **Features**:
  - Lifecycle node with all callbacks
  - Publishers and subscribers with custom QoS
  - Service and action servers
  - CUDA kernel offloading
  - Conditional compilation

### 2. MPPI Robot Controller
- **Directory**: [mppi_robot/](mppi_robot/)
- **Description**: A more complex example implementing a Model Predictive Path Integral (MPPI) controller for robot navigation.
- **Features**:
  - Advanced ROS2 node configuration
  - Parameter server integration
  - Python node implementation
  - Launch file configuration

## 🛠 Feature Showcase

Explore specific RoboDSL features through these focused examples:

### ♻️ Lifecycle Node Example

RoboDSL's lifecycle nodes provide a structured way to manage your node's resources and state. This example shows a complete implementation with all lifecycle callbacks.

**Key Concepts**:
- Node state management (Unconfigured, Inactive, Active, Finalized)
- Resource lifecycle hooks
- Error handling and recovery
- Parameter handling

**Example Code**:

```python
lifecycle_node sensor_processor {
    namespace = "robot1"
    
    # Enable automatic parameter handling
    allow_undeclared_parameters = false
    automatically_declare_parameters_from_overrides = true
    
    # Topic remapping
    remap = {
        "image_raw": "camera/image_raw",
        "processed_image": "camera/processed"
    }
    
    # Lifecycle callbacks
    on_configure = "on_configure"
    on_activate = "on_activate"
    on_deactivate = "on_deactivate"
    on_cleanup = "on_cleanup"
    on_shutdown = "on_shutdown"
    on_error = "on_error"
    
    # QoS configuration
    state_qos = {
        reliability = "reliable"
        durability = "transient_local"
        depth = 10
    }
}
```

### ⚡ QoS Configuration Example

Quality of Service (QoS) settings are crucial for reliable ROS2 communication. This example demonstrates how to configure QoS profiles for different communication patterns.

**Key Concepts**:
- Reliability vs. best-effort delivery
- Transient local durability
- History depth and queue sizes
- Deadline, lifespan, and liveliness policies

**Example Code**:

```python
# Define custom QoS profiles
sensor_qos = {
    reliability = "best_effort"
    durability = "volatile"
    history = "keep_last"
    depth = 5
}

command_qos = {
    reliability = "reliable"
    durability = "transient_local"
    history = "keep_last"
    depth = 10
}

node control_node {
    publishers = [
        {
            name = "cmd_vel"
            type = "geometry_msgs/msg/Twist"
            topic = "cmd_vel"
            qos = command_qos
        }
    ]
    
    subscribers = [
        {
            name = "lidar_scan"
            type = "sensor_msgs/msg/LaserScan"
            topic = "scan"
            qos = sensor_qos
            callback = "lidar_callback"
        }
    ]
}
```

### 🚀 CUDA Offloading Example

Leverage GPU acceleration for compute-intensive tasks with RoboDSL's seamless CUDA integration.

**Key Concepts**:
- Kernel definition and invocation
- Memory management
- Grid and block configuration
- Host-device data transfer

**Example Code**:

```python
cuda_kernel process_image {
    inputs = [
        { name: "input", type: "const uchar4*" },
        { name: "output", type: "uchar4*" },
        { name: "width", type: "int" },
        { name: "height", type: "int" }
    ]
    
    grid = { x: "(width + 15) / 16", y: "(height + 15) / 16", z: 1 }
    block = { x: 16, y: 16, z: 1 }
    
    code = """
    __global__ void process_image_kernel(const uchar4* input, uchar4* output, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < width && y < height) {
            int idx = y * width + x;
            uchar4 pixel = input[idx];
            
            // Simple grayscale conversion
            unsigned char gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
            output[idx] = make_uchar4(gray, gray, gray, 255);
        }
    }
    """
}
```

### 🔀 Conditional Compilation Example

Write portable code that adapts to different hardware and build configurations using RoboDSL's conditional compilation features.

**Key Concepts**:
- Feature flags
- Platform-specific code paths
- Debug vs. release builds
- Fallback implementations

**Example Code**:

```python
project "robot_controller" {
    features = {
        "ENABLE_CUDA": true,
        "ENABLE_ROS2": true,
        "DEBUG_MODE": false
    }
}

cuda_kernel fast_math {
    inputs = [
        { name: "input", type: "const float*" },
        { name: "output", type: "float*" },
        { name: "size", type: "int" }
    ]
    
    code = """
    #if ENABLE_CUDA && defined(__CUDACC__)
    // Fast CUDA implementation
    __global__ void fast_math_kernel(const float* input, float* output, int size) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < size) {
            // Fast CUDA math operations
            output[idx] = __expf(input[idx]);
        }
    }
    #else
    // Fallback CPU implementation
    void fast_math_cpu(const float* input, float* output, int size) {
        #if DEBUG_MODE
        printf("Using CPU fallback implementation\n");
        #endif
        
        for (int i = 0; i < size; ++i) {
            output[i] = expf(input[i]);
        }
    }
    #endif
    """
}
```

## 🚀 Running Examples

### Building All Examples

```bash
# From the root of the repository
cd /path/to/robodsl

# Build all examples
colcon build --packages-select robodsl_examples

# Or build a specific example
colcon build --packages-select example_package_name

# Build with debug symbols
colcon build --packages-select robodsl_examples --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Running Examples

1. **Source the workspace**:
   ```bash
   # In the root of your workspace
   source install/setup.bash
   ```

2. **Run a specific example**:
   ```bash
   # Run a lifecycle node example
   ros2 run robodsl_examples lifecycle_node_example
   
   # Launch a complete example system
   ros2 launch example_package example_launch.py
   ```

3. **Using Parameters**:
   ```bash
   # Override parameters at runtime
   ros2 run example_package example_node --ros-args -p param_name:=param_value
   ```

### Debugging Tips

- **View Node Graph**:
  ```bash
  rqt_graph
  ```

- **Monitor Topics**:
  ```bash
  ros2 topic list
  ros2 topic echo /topic_name
  ```

- **Inspect Parameters**:
  ```bash
  ros2 param list
  ros2 param get /node_name parameter_name
  ```

- **Visualization**:
  ```bash
  rviz2
  ```

## 🏆 Best Practices

Follow these guidelines to write maintainable and efficient RoboDSL code:

### ♻️ Lifecycle Node Guidelines

1. **State Management**
   - Always implement all lifecycle callbacks, even if empty
   - Validate parameters in `on_configure`
   - Acquire resources in `on_activate`, release in `on_deactivate`
   - Clean up all resources in `on_cleanup` and `on_shutdown`
   - Implement comprehensive error handling in `on_error`

2. **Error Handling**
   - Use the error callback for all error conditions
   - Provide meaningful error messages
   - Implement recovery strategies where possible
   - Log all state transitions and important events

### ⚡ QoS Best Practices

1. **Profile Selection**
   - Use `reliable` QoS for commands and configuration
   - Prefer `best_effort` for high-frequency sensor data
   - Use `transient_local` for late-joining subscribers
   - Adjust history depth based on your requirements

2. **Performance Considerations**
   - Monitor message queue sizes
   - Be mindful of memory usage with large history depths
   - Test with realistic network conditions

### 🚀 CUDA Development

1. **Memory Management**
   - Use RAII patterns for device memory
   - Minimize host-device transfers
   - Use pinned memory for faster transfers
   - Implement proper error checking for CUDA calls

2. **Performance Optimization**
   - Choose appropriate block and grid sizes
   - Use shared memory effectively
   - Minimize thread divergence
   - Profile with Nsight Systems/Compute

### 🏗 Project Organization

1. **Code Structure**
   - Group related functionality in separate files
   - Use namespaces to avoid naming conflicts
   - Keep node definitions focused and single-purpose
   - Document public interfaces thoroughly

2. **Build System**
   - Use CMake presets for different build types
   - Enable all compiler warnings
   - Use static analysis tools
   - Set up CI/CD pipelines

## 📚 Additional Resources

- [RoboDSL Documentation](../README.md)
- [ROS2 Documentation](https://docs.ros.org/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [ROS2 Best Practices](https://docs.ros.org/en/rolling/Concepts/Basic/About-Quality-of-Service-Settings.html)

## 🤝 Contributing

We welcome contributions to our examples! Please see the [Contributing Guide](../CONTRIBUTING.md) for details on how to submit improvements.

## 📄 License

This project is licensed under the [MIT License](../LICENSE).

2. **QoS Configuration**:
   - Match publisher/subscriber QoS profiles
   - Use appropriate reliability and durability settings
   - Consider performance implications of QoS settings

3. **CUDA Offloading**:
   - Pre-allocate GPU buffers when possible
   - Check CUDA error codes
   - Provide CPU fallbacks for non-CUDA systems

4. **Conditional Compilation**:
   - Use feature flags for platform-specific code
   - Document all feature flags
   - Test all code paths

## Troubleshooting

- **QoS Mismatch**: If messages aren't being received, check QoS compatibility
- **CUDA Errors**: Verify CUDA installation and device capabilities
- **Lifecycle Issues**: Ensure proper state transitions and error handling
