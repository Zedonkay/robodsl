# RoboDSL Examples

Practical examples demonstrating RoboDSL features.

## Prerequisites

- RoboDSL (see [README](../README.md))
- ROS2 Humble+
- CUDA Toolkit (for GPU examples)

## Quick Start

```bash
# Build examples
$ cd /path/to/example
$ colcon build --packages-select <example_name>
$ source install/setup.bash
$ ros2 run <example_name> <node_name>
```

## Examples

### 1. Comprehensive Example
`comprehensive_example.robodsl`
- Lifecycle nodes
- Custom QoS profiles
- Parameter server
- CUDA acceleration
- Namespace/remapping

### 2. MPPI Controller
`mppi_robot/`
- Advanced ROS2 config
- Parameter validation
- Python nodes
- Launch files

## Lifecycle Node

```python
lifecycle_node sensor_processor {
    namespace = "robot1"
    allow_undeclared_parameters = false
    remap = {
        "image_raw": "camera/image_raw",
        "processed_image": "camera/processed"
    }
    
    # Callbacks
    on_configure = "on_configure"
    on_activate = "on_activate"
    on_deactivate = "on_deactivate"
    
    # QoS
    state_qos = {
        reliability = "reliable"
        durability = "transient_local"
        depth = 10
    }
}
```

## QoS Configuration

```python
# QoS profiles
sensor_qos = {
    reliability = "best_effort"
    durability = "volatile"
    depth = 5
}

command_qos = {
    reliability = "reliable"
    durability = "transient_local"
    depth = 10
}

node control_node {
    publishers = [{
        name = "cmd_vel"
        type = "geometry_msgs/msg/Twist"
        topic = "cmd_vel"
        qos = command_qos
    }]
    
    subscribers = [{
        name = "lidar_scan"
        type = "sensor_msgs/msg/LaserScan"
        topic = "scan"
        qos = sensor_qos
        callback = "lidar_callback"
    }]
}
```

## CUDA Offloading

```python
cuda_kernel process_image {
    inputs = [
        { name: "input", type: "const uchar4*" },
        { name: "output", type: "uchar4*" },
        { name: "width", type: "int" },
        { name: "height", type: "int" }
    ]
    
    grid = { x: "(width + 15)/16", y: "(height + 15)/16", z: 1 }
    block = { x: 16, y: 16, z: 1 }
    
    code = """
    __global__ void process_image_kernel(
        const uchar4* input, 
        uchar4* output, 
        int width, 
        int height
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < width && y < height) {
            int idx = y * width + x;
            uchar4 pixel = input[idx];
            // Grayscale conversion
            unsigned char gray = 0.299f*pixel.x + 0.587f*pixel.y + 0.114f*pixel.z;
            output[idx] = make_uchar4(gray, gray, gray, 255);
        }
    }
    """
}
```

### Conditional Compilation Example

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

## Running Examples

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

## Best Practices

Follow these guidelines to write maintainable and efficient RoboDSL code:

### Lifecycle Node Guidelines

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

### CUDA Development

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

### Project Organization

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

## Additional Resources

- [RoboDSL Documentation](../README.md)
- [ROS2 Documentation](https://docs.ros.org/)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [ROS2 Best Practices](https://docs.ros.org/en/rolling/Concepts/Basic/About-Quality-of-Service-Settings.html)

## Contributing

We welcome contributions to our examples! Please see the [Contributing Guide](../CONTRIBUTING.md) for details on how to submit improvements.

## License

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
