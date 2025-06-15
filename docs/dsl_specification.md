# RoboDSL Language Specification

## Table of Contents
1. [Introduction](#introduction)
2. [Syntax Overview](#syntax-overview)
3. [Project Definition](#project-definition)
4. [Node Definition](#node-definition)
   - [Node Types](#node-types)
   - [Lifecycle Nodes](#lifecycle-nodes)
   - [Component Nodes](#component-nodes)
5. [Communication](#communication)
   - [Publishers](#publishers)
   - [Subscribers](#subscribers)
   - [Services](#services)
   - [Actions](#actions)
   - [Parameters](#parameters)
6. [GPU Acceleration](#gpu-acceleration)
   - [CUDA Kernels](#cuda-kernels)
   - [Thrust Integration](#thrust-integration)
7. [Build Configuration](#build-configuration)
8. [Standard Library](#standard-library)
9. [Best Practices](#best-practices)
10. [Examples](#examples)

## Introduction

RoboDSL is a domain-specific language designed for defining ROS2 nodes with CUDA acceleration. This document provides a complete specification of the RoboDSL syntax and semantics.

### Version
This specification applies to RoboDSL version 0.1.0 and later.

### Design Goals
- Provide a clean, declarative syntax for defining ROS2 nodes
- Simplify integration of CUDA-accelerated computations
- Enable code reuse through modular components
- Support both simple and complex robotics applications
- Maintain compatibility with ROS2 ecosystem

## Syntax Overview

RoboDSL uses a Python-like syntax with significant whitespace. The language is case-sensitive and uses UTF-8 encoding.

### Comments

```python
# Single-line comment

"""
Multi-line
comment
"""
```

### Identifiers

- Start with a letter or underscore
- Can contain letters, numbers, and underscores
- Case-sensitive
- Cannot be a reserved keyword

### Reserved Keywords

```
project node message service enum struct
import include from as
publishers subscribers services parameters
cuda_kernel input output code
if else for while return
int float double bool string
void true false null
```

## Project Definition

Every RoboDSL file starts with a project declaration:

```python
project "my_robot" {
    version = "0.1.0"
    description = "My awesome robot"
    author = "Your Name <your.email@example.com>"
    license = "Apache-2.0"
}
```

## Node Definition

Nodes are the main building blocks of a RoboDSL application. RoboDSL supports several node types, each with specific characteristics and use cases.

### Node Types

#### Basic Node
```python
node my_node {
    # Node configuration
    namespace = "robot"
    executable = true
    
    # Dependencies
    depends = ["rclcpp", "std_msgs"]
    
    # Enable debug output
    debug = true
    
    # ROS parameters
    parameters = [
        {
            name = "max_speed"
            type = "double"
            default = 1.0
            description = "Maximum speed in meters per second"
        }
    ]
}
```

### Lifecycle Nodes

Lifecycle nodes provide managed states for better node lifecycle management:

```python
lifecycle_node navigation_node {
    # Lifecycle callbacks
    on_configure = "on_configure"
    on_activate = "on_activate"
    on_deactivate = "on_deactivate"
    on_cleanup = "on_cleanup"
    on_shutdown = "on_shutdown"
    
    # QoS configuration
    qos_overrides = {
        "/cmd_vel": "sensor_data",
        "/odom": "default"
    }
}
```

### Component Nodes

Component nodes enable dynamic composition in ROS2:

```python
component_node camera_node {
    plugin = "my_package::CameraComponent"
    parameters = [
        {
            name = "frame_rate"
            type = "int"
            default = 30
        }
    ]
}
```

## Communication

### Publishers

Publishers send messages to topics:

```python
publishers = [
    {
        name = "odom"
        type = "nav_msgs/msg/Odometry"
        qos = {
            reliability = "reliable"  # or "best_effort"
            durability = "volatile"   # or "transient_local"
            depth = 10
            deadline = 100ms
            lifespan = 1000ms
            liveliness = 1s
            liveliness_lease_duration = 2s
        }
        latch = false
    }
]
```

### Subscribers

Subscribers receive messages from topics:

```python
subscribers = [
    {
        topic = "cmd_vel"
        type = "geometry_msgs/msg/Twist"
        callback = "cmd_vel_callback"
        qos = "sensor_data"
        callback_group = "mutually_exclusive"  # or "reentrant"
    }
]
```

### Services

Services provide request/response functionality:

```python
services = [
    {
        name = "set_parameters"
        type = "rcl_interfaces/srv/SetParameters"
        callback = "set_parameters_callback"
        qos = "services_default"
    }
]
```

### Actions

Actions support long-running operations with feedback and can be accelerated with CUDA:

```python
action_servers = [
    {
        name = "navigate_to_pose"
        type = "nav2_msgs/action/NavigateToPose"
        execute_callback = "navigate_to_pose_callback"
        goal_callback = "navigate_goal_callback"
        cancel_callback = "navigate_cancel_callback"
        result_timeout = 30s
        
        # Optional: Configure action server QoS
        qos = {
            reliability = "reliable"
            durability = "volatile"
            history = { depth = 10, policy = "keep_last" }
        }
        
        # Optional: Enable CUDA acceleration for action processing
        cuda_acceleration = {
            enabled = true
            kernel = "process_navigation"  # Reference to a cuda_kernel
            input_mapping = [
                { action_field: "goal.pose", kernel_param: "target_pose" },
                { action_field: "current_pose", kernel_param: "current_pose" }
            ]
            output_mapping = [
                { kernel_param: "result_path", action_field: "result.path" },
                { kernel_param: "feedback_progress", action_field: "feedback.progress" }
            ]
        }
    }
]
```

#### CUDA-Accelerated Actions

When using CUDA with actions, the action server will automatically handle:
1. Data transfer between host and device
2. Kernel execution
3. Progress feedback
4. Result processing

Example CUDA kernel for action processing:

```python
cuda_kernel process_navigation {
    inputs = [
        { name: "target_pose", type: "geometry_msgs::msg::PoseStamped" },
        { name: "current_pose", type: "geometry_msgs::msg::PoseStamped" },
        { name: "map_data", type: "nav_msgs::msg::OccupancyGrid::ConstPtr" }
    ]
    outputs = [
        { name: "result_path", type: "nav_msgs::msg::Path" },
        { name: "feedback_progress", type: "float" }
    ]
    
    code = """
    __global__ void process_navigation(
        const PoseStamped* target,
        const PoseStamped* current,
        const OccupancyGrid* map,
        Path* result,
        float* progress) {
        
        // CUDA-accelerated path planning
        // ...
    }
    """
}
```

### Parameters

Parameters allow runtime configuration:

```python
parameters = [
    {
        name = "use_sim_time"
        type = "bool"
        default = false
        description = "Use simulation time"
        read_only = true
    },
    {
        name = "sensor_config"
        type = "string"
        default = "default_config.yaml"
        description = "Path to sensor configuration"
    }
]
```

## GPU Acceleration

### CUDA Kernels

Define CUDA kernels for GPU acceleration:

```python
cuda_kernel pointcloud_processor {
    inputs = ["input_cloud"]
    outputs = ["processed_cloud"]
    block_size = 256
    shared_mem = 4096  # bytes
    
    code = """
    __global__ void process_pointcloud(
        const float4* input,
        float4* output,
        int num_points,
        float threshold) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;
        
        float4 point = input[idx];
        // Process point...
        output[idx] = point;
    }
    """
}
```

### Thrust Integration

Use Thrust for high-level parallel algorithms:

```python
thrust_operation sort_points {
    algorithm = "sort"
    input = "point_cloud"
    output = "sorted_cloud"
    
    code = """
        auto policy = thrust::device;
        thrust::sort(policy, input.begin(), input.end(), 
                    [] __device__ (const auto& a, const auto& b) {
                        return a.z < b.z;  # Sort by z-coordinate
                    });
    """
}
```

## Build Configuration

RoboDSL provides flexible build configuration with conditional compilation:

```python
build_options = {
    # Build type
    "CMAKE_BUILD_TYPE" = "Release"  # or "Debug", "RelWithDebInfo", "MinSizeRel"
    
    # Feature flags
    "ENABLE_ROS2" = true      # Enable ROS2 integration
    "ENABLE_CUDA" = true      # Enable CUDA support
    "ENABLE_THRUST" = true    # Enable Thrust library
    
    # CUDA configuration
    "CUDA_ARCH" = "sm_75"     # Compute capability (e.g., sm_75 for Turing)
    "CUDA_STANDARD" = 17      # C++ standard for CUDA
    "CUDA_SEPARABLE_COMPILATION" = true
    
    # Build options
    "BUILD_TESTING" = true
    "BUILD_DOCS" = false
    "BUILD_EXAMPLES" = true
    
    # Installation paths
    "CMAKE_INSTALL_PREFIX" = "install"
    "CMAKE_INSTALL_LIBDIR" = "lib"
    "CMAKE_INSTALL_INCLUDEDIR" = "include"
}

# Conditional compilation
conditional_features = [
    {
        name = "WITH_GPU_ACCELERATION"
        condition = "ENABLE_CUDA"
        description = "Enable GPU acceleration features"
    },
    {
        name = "WITH_ROS2_LIFECYCLE"
        condition = "ENABLE_ROS2"
        description = "Enable ROS2 lifecycle node support"
    }
]

# Dependencies with version constraints
dependencies = [
    "rclcpp >= 2.0.0",
    "std_msgs >= 4.0.0",
    "sensor_msgs >= 4.0.0",
    "nav_msgs >= 4.0.0",
    "tf2_ros >= 0.25.0",
    "rclcpp_lifecycle >= 2.0.0" if build_options["ENABLE_ROS2"] else "",
    "CUDA::cudart" if build_options["ENABLE_CUDA"] else ""
]

# Optional: Custom CMake code to include in the build
cmake_extra = """
# Add custom CMake code here
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_compile_definitions(WITH_CUDA_ACCELERATION)
endif()
"""

## Node Definition

### Node Types

RoboDSL supports different node types:

```python
# Standard ROS2 node
node my_node {
    executable = true
    namespace = "sensors"  # Optional namespace
    # ...
}

# Component node (for composition)
component my_component {
    plugin = "my_pkg::MyComponent"
    # ...
}
```

### Lifecycle Nodes

Lifecycle nodes provide state management and are essential for system robustness. They follow the managed node pattern from `rclcpp_lifecycle`.

#### Lifecycle States
- **Unconfigured**: Initial state, node is not active
- **Inactive**: Configured but not processing
- **Active**: Fully operational
- **Finalized**: Shutdown complete

#### Configuration

```python
lifecycle_node navigation {
    # Standard node configuration
    executable = true
    namespace = "navigation"  # Optional namespace
    
    # Lifecycle callbacks (all optional)
    on_configure = "configure_callback"  # Called during configuration
    on_activate = "activate_callback"    # Called when activating
    on_deactivate = "deactivate_callback" # Called when deactivating
    on_cleanup = "cleanup_callback"      # Cleanup resources
    on_shutdown = "shutdown_callback"    # Shutdown handler
    on_error = "error_callback"          # Error handler
    
    # State transition timeouts (optional, in seconds)
    transition_timeouts = {
        "configure" = 10.0
        "activate" = 5.0
        "deactivate" = 5.0
        "cleanup" = 10.0
        "shutdown" = 10.0
    }
}
```

#### Best Practices
1. Keep callbacks short and non-blocking
2. Use the `on_configure` callback to allocate resources
3. Perform heavy initialization before activation
4. Handle errors gracefully and transition to appropriate states

### Publishers

Define publishers with flexible configuration:

```python
publishers = [
    {
        name = "cmd_vel"
        type = "geometry_msgs/msg/Twist"
        qos = "sensor_data"  # Named QoS profile
        queue_size = 10
        # Optional: Custom QoS overrides
        qos_overrides = {
            reliability = "reliable"  # or "best_effort"
            durability = "volatile"   # or "transient_local"
            history = {
                depth = 10
                policy = "keep_last"  # or "keep_all"
            }
            deadline = "100ms"        # Duration
            lifespan = "500ms"        # Duration
            liveliness = {
                lease_duration = "1s"
                policy = "automatic"  # or "manual_by_topic", "manual_by_namespace"
            }
        }
    }
]
```

### Subscribers

Define subscribers with message callbacks and QoS settings:

```python
subscribers = [
    {
        name = "scan"
        type = "sensor_msgs/msg/LaserScan"
        callback = "scan_callback"
        qos = "sensor_data"
        # Optional: Custom QoS overrides (same options as publishers)
        qos_overrides = {
            reliability = "best_effort"
            history = { depth = 5, policy = "keep_last" }
        }
    }
]
```

### Namespace and Remapping

Nodes can be organized in namespaces and have topic remappings:

```python
node my_node {
    executable = true
    namespace = "sensors"  # Node namespace
    
    # Remap topics
    remappings = [
        {"from": "old_topic", "to": "new_topic"}
    ]
    
    # Parameters can be namespaced
    parameters = [
        {
            name = "sensor.rate"  # Will be 'sensors.sensor.rate'
            type = "double"
            default = 10.0
        }
    ]
}
```

## Standard Library

RoboDSL provides a standard library with common functionality:

### Math Functions

```python
math.sqrt(x)
math.sin(angle)
math.cos(angle)
math.atan2(y, x)
math.min(a, b)
math.max(a, b)
math.clamp(value, min, max)
```

### ROS2 Utilities

```python
ros2.create_timer(period_sec, callback)
ros2.get_parameter(name, default)
ros2.log_info(message)
ros2.log_warn(message)
ros2.log_error(message)
ros2.log_fatal(message)
```

## Build Configuration

Configure the build system:

```python
build {
    cpp_standard = 17
    cuda_arch = "sm_75"
    optimization_level = "O3"
    
    # Dependencies
    find_packages = [
        "rclcpp",
        "std_msgs",
        "sensor_msgs"
    ]
    
    # Additional compiler flags
    cxx_flags = [
        "-Wall",
        "-Wextra",
        "-Wpedantic"
    ]
    
    # Additional linker flags
    link_flags = [
        "-pthread"
    ]
}
```

## Examples

### Complete Example

```python
project "my_robot" {
    version = "0.1.0"
    description = "A simple robot controller"
}

# Custom message type
message Twist2D {
    float64 linear_x
    float64 angular_z
}

# Main node
node robot_controller {
    # Node configuration
    namespace = "robot"
    executable = true
    
    # Dependencies
    depends = ["rclcpp", "std_msgs"]
    
    # Publishers and subscribers
    publishers = [
        {
            name = "cmd_vel"
            type = "geometry_msgs/msg/Twist"
            qos = "default"
        }
    ]
    
    subscribers = [
        {
            name = "odom"
            type = "nav_msgs/msg/Odometry"
            callback = "odom_callback"
            qos = "sensor_data"
        }
    ]
    
    # Parameters
    parameters = {
        "max_speed" = 1.0
        "publish_rate" = 30.0
    }
    
    # CUDA kernel for processing
    cuda_kernel process_sensors {
        inputs = [
            { name: "sensor_data", type: "const SensorData&" }
        ]
        outputs = [
            { name: "processed_data", type: "ProcessedData&" }
        ]
        
        block_size = [256, 1, 1]
        
        code = """
        __global__ void process_kernel(
            const SensorData* input,
            ProcessedData* output) {
            // Kernel implementation
        }
        """
    }
}
```

### CUDA-Accelerated Image Processing

```python
project "image_processor" {
    version = "0.1.0"
}

node image_processor {
    # Node configuration
    executable = true
    
    # Dependencies
    depends = ["rclcpp", "sensor_msgs", "cv_bridge", "opencv2"]
    
    # Subscribers
    subscribers = [
        {
            name = "image_raw"
            type = "sensor_msgs/msg/Image"
            callback = "image_callback"
            qos = "sensor_data"
        }
    ]
    
    # Publishers
    publishers = [
        {
            name = "image_processed"
            type = "sensor_msgs/msg/Image"
            qos = "default"
        }
    ]
    
    # CUDA kernel for image processing
    cuda_kernel process_image {
        inputs = [
            { name: "input_image", type: "const cv::cuda::GpuMat&" },
            { name: "params", type: "const ImageParams&" }
        ]
        outputs = [
            { name: "output_image", type: "cv::cuda::GpuMat&" }
        ]
        
        includes = [
            "opencv2/core/cuda.hpp",
            "opencv2/cudaimgproc.hpp"
        ]
        
        block_size = [32, 32, 1]
        
        code = """
        __global__ void process_image_kernel(
            const cv::cuda::PtrStepSz<uchar3> src,
            cv::cuda::PtrStepSz<uchar3> dst,
            const ImageParams* params) {
            
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= src.cols || y >= src.rows) return;
            
            // Simple grayscale conversion
            uchar3 pixel = src(y, x);
            unsigned char gray = 0.299f * pixel.x + 0.587f * pixel.y + 0.114f * pixel.z;
            
            dst(y, x) = make_uchar3(gray, gray, gray);
        }
        """
    }
}
```

### Using Thrust

```python
cuda_kernel sort_points {
    inputs = [
        { name: "points", type: "thrust::device_vector<Point2D>&" }
    ]
    outputs = [
        { name: "sorted_points", type: "thrust::device_vector<Point2D>&" }
    ]
    
    use_thrust = true
    
    includes = [
        "thrust/sort.h",
        "thrust/execution_policy.h"
    ]
    
    code = """
    void sort_points(
        const thrust::device_vector<Point2D>& points,
        thrust::device_vector<Point2D>& sorted_points) {
        
        // Copy input to output
        sorted_points = points;
        
        // Sort points by x-coordinate
        thrust::sort(
            thrust::device,
            sorted_points.begin(),
            sorted_points.end(),
            [] __device__ (const Point2D& a, const Point2D& b) {
                return a.x < b.x;
            }
        );
    }
    """
}
```

## Conclusion

This document provides a comprehensive reference for the RoboDSL language. For more examples and tutorials, see the [examples](examples/) directory in the RoboDSL repository.
