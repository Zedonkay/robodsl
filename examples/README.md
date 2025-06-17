# RoboDSL Examples

This directory contains example RoboDSL configurations demonstrating various features and best practices.

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

## Feature Examples

### Lifecycle Node Example

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

### QoS Configuration Example

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

### CUDA Offloading Example

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

### Conditional Compilation Example

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

1. **Build the examples**:
   ```bash
   cd /path/to/robodsl
   colcon build --packages-select robodsl_examples
   ```

2. **Source the workspace**:
   ```bash
   source install/setup.bash
   ```

3. **Run an example node**:
   ```bash
   ros2 run robodsl_examples lifecycle_node_example
   ```

## Best Practices

1. **Lifecycle Nodes**:
   - Always implement proper state transitions
   - Clean up resources in cleanup/shutdown callbacks
   - Use error handling callbacks for robustness

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
