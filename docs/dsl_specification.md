# RoboDSL Language Specification

## Table of Contents
1. [Introduction](#introduction)
2. [Syntax Overview](#syntax-overview)
3. [Project Definition](#project-definition)
4. [Node Definition](#node-definition)
5. [Message Types](#message-types)
6. [Services](#services)
7. [Parameters](#parameters)
8. [CUDA Kernels](#cuda-kernels)
9. [Expressions and Types](#expressions-and-types)
10. [Standard Library](#standard-library)
11. [Build Configuration](#build-configuration)
12. [Examples](#examples)

## Introduction

RoboDSL is a domain-specific language designed for defining ROS2 nodes with CUDA acceleration. This document provides a complete specification of the RoboDSL syntax and semantics.

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

Nodes are the main building blocks of a RoboDSL application:

```python
node my_node {
    # Node configuration
    namespace = "robot"
    executable = true
    
    # Dependencies
    depends = ["rclcpp", "std_msgs"]
    
    # Publishers
    publishers = [
        {
            name = "odom"
            type = "nav_msgs/msg/Odometry"
            qos = "sensor_data"
            queue_size = 10
        }
    ]
    
    # Subscribers
    subscribers = [
        {
            name = "cmd_vel"
            type = "geometry_msgs/msg/Twist"
            callback = "cmd_vel_callback"
            qos = "default"
        }
    ]
    
    # Parameters
    parameters = {
        "max_speed" = 1.0
        "publish_rate" = 30.0
        "use_sim_time" = false
    }
    
    # Lifecycle callbacks
    on_configure = "on_configure"
    on_activate = "on_activate"
    on_deactivate = "on_deactivate"
    on_cleanup = "on_cleanup"
    on_shutdown = "on_shutdown"
}
```

## Message Types

Define custom message types:

```python
message Point2D {
    float64 x
    float64 y
}

message Pose2D {
    Point2D position
    float64 theta
}
```

## Services

Define ROS2 services:

```python
service AddTwoInts {
    request {
        int64 a
        int64 b
    }
    response {
        int64 sum
    }
}
```

## Parameters

Define node parameters with types and defaults:

```python
parameters = {
    "max_speed" = {
        type = "double"
        default = 1.0
        description = "Maximum speed in m/s"
        min = 0.0
        max = 5.0
    },
    "use_sim_time" = {
        type = "bool"
        default = false
    }
}
```

## CUDA Kernels

Define CUDA kernels for GPU acceleration:

```python
cuda_kernel process_lidar {
    # Input and output specifications
    inputs = [
        { name: "input_scan", type: "sensor_msgs::msg::LaserScan::ConstPtr" },
        { name: "params", type: "const LidarParams&" }
    ]
    outputs = [
        { name: "obstacles", type: "std::vector<Obstacle>&" }
    ]
    
    # Kernel configuration
    block_size = [256, 1, 1]
    grid_size = [1, 1, 1]
    shared_mem_bytes = 4096
    use_thrust = true
    
    # Includes
    includes = [
        "sensor_msgs/msg/laser_scan.hpp",
        "vector"
    ]
    
    # Kernel code
    code = """
    __global__ void process_lidar_kernel(
        const float* ranges, int num_points,
        const LidarParams* params,
        Obstacle* obstacles, int* num_obstacles) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;
        
        // Your CUDA kernel code here
        
    }
    """
}
```

## Expressions and Types

### Literals

```python
# Numbers
42          # Integer
3.14        # Float
true        # Boolean
"hello"     # String
[1, 2, 3]   # List
{x: 1, y: 2} # Dictionary
```

### Type System

- **Primitive Types**: `int`, `float`, `double`, `bool`, `string`
- **ROS2 Types**: `nav_msgs/msg/Odometry`, `sensor_msgs/msg/Image`
- **Containers**: `List[T]`, `Dict[K, V]`
- **Custom Types**: User-defined messages and services

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
