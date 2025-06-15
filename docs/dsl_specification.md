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

Lifecycle nodes implement the ROS2 managed node pattern, providing a state machine for managing the node's lifecycle. This enables better resource management, error recovery, and system composition.

#### Key Features
- **State Management**: Automatic handling of node states (Unconfigured, Inactive, Active, Finalized)
- **Resource Management**: Explicit lifecycle hooks for resource allocation and cleanup
- **Error Recovery**: Built-in error states and recovery mechanisms
- **System Composition**: Better support for system bringup/teardown sequences

#### Basic Usage
```python
lifecycle_node my_lifecycle_node {
    namespace = "robot"
    
    # Lifecycle configuration
    autostart = true  # Default: false, if true, automatically transitions to Active
    
    # Optional callbacks (all return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn)
    on_configure = "on_configure"      # Called during transition to Inactive
    on_cleanup = "on_cleanup"          # Called during transition to Unconfigured
    on_activate = "on_activate"        # Called during transition to Active
    on_deactivate = "on_deactivate"    # Called during transition to Inactive
    on_shutdown = "on_shutdown"        # Called during transition to Finalized
    on_error = "on_error_callback"     # Called on error transitions
}
```

#### Lifecycle States
1. **Unconfigured**: Initial state, no resources allocated
2. **Inactive**: Configured but not active (resources allocated but not processing)
3. **Active**: Fully operational state
4. **Finalized**: Shutdown/error state

#### Error Handling
Lifecycle nodes provide robust error handling through the `on_error` callback:

```python
def on_error_callback(state: LifecycleNodeState) -> CallbackReturn:
    """Handle error state transitions."""
    if state.id() == State.PRIMARY_STATE_ACTIVATING:
        # Handle activation errors
        return CallbackReturn.FAILURE
    return CallbackReturn.SUCCESS
```

#### Best Practices
- Always implement proper cleanup in `on_cleanup` and `on_shutdown`
- Use `on_error` for robust error recovery
- Minimize work in constructors; use `on_configure` for heavy initialization
- Use `on_activate`/`on_deactivate` for runtime resource management

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

### Actions with CUDA Offloading

Actions in RoboDSL support long-running operations with feedback and can be accelerated using CUDA for compute-intensive tasks. This section covers both basic action configuration and advanced CUDA offloading features.

#### Basic Action Server

```python
action_servers = [
    {
        name = "navigate_to_pose"
        type = "nav2_msgs/action/NavigateToPose"
        execute_callback = "navigate_to_pose_callback"
        goal_callback = "navigate_goal_callback"
        cancel_callback = "navigate_cancel_callback"
        result_timeout = 30s  # Default timeout for action completion
        
        # Configure action server QoS
        qos = {
            reliability = "reliable"
            durability = "volatile"
            history = { depth = 10, policy = "keep_last" }
        }
    }
]
```

#### CUDA-Accelerated Actions

For compute-intensive actions, you can leverage CUDA acceleration:

```python
action_servers = [
    {
        name = "process_pointcloud"
        type = "sensor_msgs/action/ProcessPointCloud"
        execute_callback = "process_pointcloud_callback"
        
        # Enable and configure CUDA acceleration
        cuda = {
            enabled = true
            # Reference to a defined cuda_kernel
            kernel = "process_pointcloud_kernel"
            
            # Memory requirements (optional)
            device_memory = 2048  # MB required on GPU
            
            # Input/output mapping between action fields and kernel parameters
            input_mapping = [
                { action_field: "goal.point_cloud", kernel_param: "input_points" },
                { action_field: "goal.params", kernel_param: "params" }
            ],
            output_mapping = [
                { kernel_param: "output_points", action_field: "result.processed_cloud" },
                { kernel_param: "num_processed", action_field: "result.num_processed" }
            ],
            
            # Kernel launch configuration (optional)
            launch_config = {
                block_size = 256
                grid_size = "auto"  # Will be calculated based on input size
                shared_mem = 0       # Bytes of shared memory per block
                stream = 0           # CUDA stream (0 = default stream)
            }
        },
        
        # QoS configuration for action server
        qos = {
            goal_service = "services"
            result_service = "services"
            cancel_service = "services"
            feedback_topic = "sensor_data"
            status_topic = "default"
        }
    }
]
```

#### CUDA Kernel Definition

Define CUDA kernels that can be used with actions:

```python
cuda_kernel process_pointcloud_kernel {
    # Input parameters
    inputs = [
        { name: "input_points", type: "const float4*" },
        { name: "params", type: "const ProcessingParams*" },
        { name: "num_points", type: "size_t" }
    ]
    
    # Output parameters (will be written by the kernel)
    outputs = [
        { name: "output_points", type: "float4*" },
        { name: "num_processed", type: "size_t*" }
    ]
    
    # CUDA code (will be compiled at build time)
    code = """
    __global__ void process_pointcloud_kernel(
        const float4* input_points,
        const ProcessingParams* params,
        size_t num_points,
        float4* output_points,
        size_t* num_processed) {
        
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= num_points) return;
        
        // Process point cloud data here
        float4 point = input_points[idx];
        // ... processing logic ...
        output_points[idx] = point;
        
        // Update count (only once per block)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            *num_processed = num_points;
        }
    }
    """
}
```

#### Action Handler with CUDA Offloading

Example of an action handler that uses CUDA acceleration:

```python
def process_pointcloud_callback(goal_handle):
    # Get goal data
    goal = goal_handle.request
    
    try:
        # Allocate device memory
        num_points = len(goal.point_cloud.data)
        d_points = cuda.mem_alloc(num_points * 16)  # float4 is 16 bytes
        d_processed = cuda.mem_alloc(num_points * 16)
        d_num_processed = cuda.mem_alloc(8)  # size_t
        
        # Copy input data to device
        cuda.memcpy_htod(d_points, goal.point_cloud.data)
        
        # Prepare CUDA kernel launch
        block_size = 256
        grid_size = (num_points + block_size - 1) // block_size
        
        # Get the kernel function
        kernel = get_cuda_kernel("process_pointcloud_kernel")
        
        # Launch kernel
        kernel(
            d_points,               # input_points
            d_params,              # params
            np.uint64(num_points),  # num_points
            d_processed,           # output_points
            d_num_processed,       # num_processed
            block=(block_size, 1, 1),
            grid=(grid_size, 1)
        )
        
        # Copy results back to host
        processed_data = np.empty(num_points * 4, dtype=np.float32)
        num_processed = np.zeros(1, dtype=np.uint64)
        cuda.memcpy_dtoh(processed_data, d_processed)
        cuda.memcpy_dtoh(num_processed, d_num_processed)
        
        # Create and set result
        result = ProcessPointCloud.Result()
        result.processed_cloud.data = processed_data.tolist()
        result.num_processed = int(num_processed[0])
        goal_handle.succeed(result)
        
    except Exception as e:
        goal_handle.abort()
        get_logger().error(f"Action failed: {str(e)}")
        raise e
        
    finally:
        # Clean up device memory
        if 'd_points' in locals():
            d_points.free()
        if 'd_processed' in locals():
            d_processed.free()
        if 'd_num_processed' in locals():
            d_num_processed.free()
```

#### Best Practices for CUDA-Accelerated Actions

1. **Memory Management**:
   - Use RAII patterns for CUDA memory allocation
   - Pre-allocate device memory when possible
   - Use pinned memory for host-device transfers
   - Always free device memory in finally blocks

2. **Error Handling**:
   - Check CUDA errors after kernel launches
   - Handle device memory allocation failures
   - Provide meaningful error messages

3. **Performance Optimization**:
   - Minimize host-device memory transfers
   - Use streams for concurrent execution
   - Profile kernels to identify bottlenecks
   - Consider using CUDA Graphs for repeated operations

4. **Thread Safety**:
   - CUDA contexts are thread-local by default
   - Use separate streams for concurrent operations
   - Consider using the CUDA Runtime API's per-thread default stream

5. **Resource Management**:
   - Set appropriate compute capability requirements
   - Handle multiple GPU configurations
   - Provide fallback CPU implementations when CUDA is not available

#### Conditional Compilation

Use conditional compilation to support both CUDA and CPU-only builds:

```python
# In your node definition
build_features = {
    "cuda": {
        "enabled": true,
        "architectures": ["sm_75"],  # Turing
        "flags": ["-O3", "--use_fast_math"]
    }
}

# In your code
def process_pointcloud_callback(goal_handle):
    try:
        # ...
        
        # Use CUDA if available, fall back to CPU
        if ENABLE_CUDA:
            result = process_with_cuda(goal_handle.request)
        else:
            result = process_on_cpu(goal_handle.request)
            
        goal_handle.succeed(result)
        
    except Exception as e:
        goal_handle.abort()
        get_logger().error(f"Action failed: {str(e)}")
        raise e
```
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

### Communication

RoboDSL provides first-class support for all ROS2 communication patterns with configurable Quality of Service (QoS) settings.

### Quality of Service (QoS) Configuration

QoS policies control how messages are delivered and processed. RoboDSL supports both predefined profiles and custom configurations.

#### Predefined QoS Profiles

| Profile | Reliability | Durability | History | Depth | Use Case |
|---------|-------------|------------|----------|-------|-----------|
| `sensor_data` | Best Effort | Volatile | Keep Last | 1 | Sensor data (LIDAR, camera) |
| `parameters` | Reliable | Volatile | Keep Last | 10 | Parameter updates |
| `services` | Reliable | Volatile | Keep Last | 10 | Service calls |
| `default` | Reliable | Volatile | Keep Last | 10 | General purpose |

#### Custom QoS Configuration

```python
qos_profile "custom_profile" {
    reliability = "reliable"  # or "best_effort"
    durability = "volatile"   # or "transient_local"
    history = "keep_last"     # or "keep_all"
    depth = 10                # Queue size for keep_last
    deadline = "100ms"        # Optional: Maximum expected time between messages
    lifespan = "500ms"        # Optional: Maximum time a message is valid
    liveliness = "automatic"  # or "manual_by_topic", "manual_by_node"
    liveliness_lease = "1s"  # Lease duration for liveliness
}
```

### Publishers

Publishers can specify QoS settings per-topic:

```python
publishers = [
    {
        topic = "/cmd_vel"
        type = "geometry_msgs/msg/Twist"
        qos = "sensor_data"  # Using predefined profile
        queue_size = 10
    },
    {
        topic = "/important_data"
        type = "std_msgs/msg/String"
        qos = {              # Inline QoS configuration
            reliability = "reliable"
            durability = "transient_local"
            history = "keep_last"
            depth = 100
        }
    }
]
```

### Subscribers

Subscribers in RoboDSL allow nodes to receive and process messages from topics. They support flexible QoS configuration and efficient message handling.

#### Basic Subscriber

```python
subscribers = [
    {
        name = "lidar_scan"
        type = "sensor_msgs/msg/LaserScan"
        callback = "lidar_scan_callback"
        queue_size = 10  # Default: 10
    }
]
```

#### QoS Configuration

Configure Quality of Service (QoS) settings for subscribers:

```python
subscribers = [
    {
        name = "camera_image"
        type = "sensor_msgs/msg/Image"
        callback = "image_callback"
        
        # Use a predefined QoS profile
        qos = "sensor_data"  # "reliable", "best_effort", "services", or custom profile
        
        # Or specify QoS settings directly
        qos = {
            reliability = "best_effort"  # or "reliable"
            durability = "volatile"      # or "transient_local"
            history = {
                policy = "keep_last"     # or "keep_all"
                depth = 5               # Only for keep_last
            }
            # Optional advanced QoS settings
            deadline = "100ms"          # Maximum expected time between messages
            lifespan = "500ms"          # Maximum time a message is valid
            liveliness = "automatic"    # or "manual_by_topic", "manual_by_node"
            liveliness_lease = "1s"     # Lease duration for liveliness
        }
    }
]
```

#### Message Callbacks

Message callbacks can be defined in your node's class:

```python
def lidar_scan_callback(self, msg):
    """
    Process incoming LIDAR scan messages.
    
    Args:
        msg (sensor_msgs.msg.LaserScan): The received LIDAR scan message
    """
    try:
        # Process the scan data
        ranges = np.array(msg.ranges)
        angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Example: Find the closest point
        valid_ranges = ranges[ranges > msg.range_min]
        if len(valid_ranges) > 0:
            min_distance = np.min(valid_ranges)
            self.get_logger().info(f"Closest obstacle: {min_distance:.2f} meters")
        
        # Example: Publish processed data
        if self.publisher_ is not None:
            output_msg = ProcessedScan()
            output_msg.header = msg.header
            output_msg.min_distance = float(min_distance)
            self.publisher_.publish(output_msg)
            
    except Exception as e:
        self.get_logger().error(f"Error processing LIDAR scan: {str(e)}")
        raise
```

#### Message Filtering

RoboDSL supports message filtering for efficient processing:

```python
subscribers = [
    {
        name = "filtered_scan"
        type = "sensor_msgs/msg/LaserScan"
        callback = "filtered_scan_callback"
        
        # Message filtering options
        filter = {
            # Only process messages where the x position is positive
            expression = "msg.header.frame_id == 'base_laser' and len(msg.ranges) > 0"
            
            # Throttle message rate (Hz)
            throttle_rate = 10.0
            
            # Queue size for filtered messages
            queue_size = 5
        }
    }
]
```

#### Best Practices

1. **QoS Matching**: Ensure publisher and subscriber QoS settings are compatible
2. **Efficient Processing**:
   - Use `const` message references in callbacks to avoid copies
   - Move expensive processing to separate threads when needed
   - Use message filters for time-synchronized processing of multiple topics
3. **Error Handling**:
   - Validate message data before processing
   - Handle exceptions in callbacks to prevent node crashes
4. **Performance Tuning**:
   - Adjust queue sizes based on message rates
   - Use appropriate history policies (keep_last vs keep_all)
   - Consider using intra-process communication for high-frequency topics

#### Example: Time-Synchronized Callbacks

```python
# Subscribe to multiple topics with time synchronization
message_filters = [
    {
        name = "sensor_sync"
        topics = ["camera/image_raw", "lidar/scan"]
        callback = "sensor_fusion_callback"
        
        # Time synchronization settings
        sync = {
            # Maximum time difference between messages (seconds)
            slop = 0.1
            
            # Queue size for message time synchronization
            queue_size = 10
        }
    }
]

def sensor_fusion_callback(self, image_msg, scan_msg):
    """Process synchronized camera and LIDAR data."""
    # Process time-synchronized messages
    # ...

## Namespace and Remapping

RoboDSL provides powerful namespace and remapping capabilities to help organize complex robotics applications and enable flexible deployment configurations. This section covers node namespacing, topic/service remapping, and parameter namespacing.

### Node Namespaces

Node namespaces help organize nodes into logical groups and prevent naming collisions:

```python
# Basic node with namespace
node navigation_node {
    namespace = "robot1"  # Full node name becomes /robot1/navigation_node
    
    # Nested namespaces using forward slashes
    # namespace = "fleet/robot1"  # Results in /fleet/robot1/navigation_node
    
    # Relative namespaces (relative to parent namespace)
    # namespace = "sensors"  # If parent is /robot1, becomes /robot1/sensors/navigation_node
}

# Multiple nodes in the same namespace
with namespace = "robot1" {
    node navigation_node { /* ... */ }
    node perception_node { /* ... */ }
    node control_node { /* ... */ }
}
```

### Topic and Service Remapping

Remap topics and services at runtime without code changes:

```python
node my_node {
    # Topic remapping
    remappings = [
        # Simple string-based remapping
        "cmd_vel:=base/cmd_vel",
        
        # Structured remapping with options
        {
            from = "/camera/image_raw"
            to = "/sensors/camera/front/image_raw"
            # Optional: Apply only to specific node interfaces
            # when = { node = "camera_driver", interface = "publisher" }
        },
        
        # Remap with regular expressions
        {
            from = "^/sensor(.*)"
            to = "/robot1/sensor\\1"
            regex = true
        }
    ]
    
    # Service remapping (same syntax as topic remapping)
    service_remappings = [
        "set_parameters:=robot1/set_parameters",
        {
            from = "get_map"
            to = "map_server/get_map"
        }
    ]
}
```

### Parameter Namespacing

Organize parameters using namespaces and handle them effectively:

```python
node parameter_node {
    # Flat parameter structure
    parameters = {
        "max_speed" = 1.0,
        "min_speed" = 0.1,
        "timeout" = 5.0
    }
    
    # Hierarchical parameter structure using dots
    parameters = {
        # Access as /parameter_node/motion/max_speed
        "motion.max_speed" = 1.0,
        "motion.min_speed" = 0.1,
        "safety.timeout" = 5.0
        
        # Nested structures using dictionaries
        "controller" = {
            "pid" = {
                "p" = 0.5,
                "i" = 0.1,
                "d" = 0.01
            },
            "rate_hz" = 100.0
        }
    }
    
    # Load parameters from YAML file
    # parameters = file("config/params.yaml")
    
    # Parameter overrides from command line
    # ros2 run my_package my_node --ros-args -p motion.max_speed:=2.0
}
```

### Best Practices

1. **Consistent Naming Conventions**
   - Use forward slashes (/) for namespaces (e.g., `robot1/sensors`)
   - Use dots (.) for parameter hierarchies (e.g., `motion.max_speed`)
   - Prefer relative names when possible for better reusability

2. **Remapping Strategy**
   - Keep default topic names simple and descriptive
   - Use remapping for deployment-specific configurations
   - Document all standard topic and service names

3. **Parameter Organization**
   - Group related parameters together
   - Use descriptive names that include units (e.g., `max_speed_mps`)
   - Document parameter types, ranges, and default values

4. **Deployment Considerations**
   - Use launch files to manage complex remapping scenarios
   - Consider using parameter files for different robot configurations
   - Test with various namespace and remapping combinations

### Advanced: Dynamic Remapping

For more complex scenarios, you can use dynamic remapping:

```python
# In your node's initialization
def __init__(self):
    # Get remappings from command line
    self.declare_parameter('remappings', [])
    remappings = self.get_parameter('remappings').value
    
    # Apply remappings dynamically
    for remap in remappings:
        from_, to = remap.split(':=', 1)
        self.get_logger().info(f"Remapping {from_} to {to}")
        # Apply remapping to specific publishers/subscribers
        
# Launch with: ros2 run my_package my_node --ros-args -p remappings:='["cmd_vel:=base/cmd_vel"]'
```

### Example: Multi-Robot System

```python
# Launch file for multiple robots
robots = ["robot1", "robot2", "robot3"]

for robot in robots:
    with namespace(robot):
        node("navigation_node") {
            remappings = [
                "cmd_vel:=base/cmd_vel",
                "odom:=base/odom",
                {
                    from = "scan"
                    to = "${robot}/base_scan"
                }
            ]
            
            parameters = {
                "robot_id" = robot,
                "motion.max_speed" = 1.0,
                "safety.timeout" = 5.0
            }
        }
```

## GPU-Accelerated Processing Example

This example demonstrates a complete implementation of a GPU-accelerated image processing pipeline using RoboDSL's CUDA support. The node receives images, processes them on the GPU, and publishes the results.

### CUDA Kernel Definition

First, define the CUDA kernel in a `.cuh` file:

```cuda
// kernels/image_processing.cuh
#pragma once
#include <cstdint>

#ifdef __CUDACC__
#define CUDA_CALLABLE __host__ __device__
#else
#define CUDA_CALLABLE
#endif

struct ImageParams {
    int width;
    int height;
    int channels;
    size_t step;
};

// Simple image processing kernel (edge detection)
__global__ void edge_detection_kernel(
    const uint8_t* input,
    uint8_t* output,
    const ImageParams params,
    const int threshold
);

// Memory management utilities
void allocate_gpu_memory(void** ptr, size_t size);
void free_gpu_memory(void* ptr);
void copy_to_gpu(const void* src, void* dst, size_t size);
void copy_from_gpu(const void* src, void* dst, size_t size);
```

### RoboDSL Node Definition

```python
# image_processor.robodsl
node image_processor {
    # Node configuration
    namespace = "perception"
    executable = true
    
    # Dependencies
    build {
        features = {
            cuda = true
            opencv = true
        }
        
        # CUDA-specific settings
        cuda = {
            architectures = ["sm_75", "sm_80"]  # Turing and Ampere
            flags = ["-O3", "--use_fast_math"]
        }
        
        # Additional include directories
        include_directories = [
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ]
        
        # Link against OpenCV and CUDA libraries
        find_packages = [
            "rclcpp",
            "sensor_msgs",
            "cv_bridge",
            "OpenCV"
        ]
    }
    
    # Publishers and Subscribers
    subscribers = [
        {
            name = "image_raw"
            type = "sensor_msgs/msg/Image"
            callback = "image_callback"
            qos = "sensor_data"
            # Enable approximate time synchronization if needed
            # approximate_time = 0.1  # 100ms
        }
    ]
    
    publishers = [
        {
            name = "edges"
            type = "sensor_msgs/msg/Image"
            qos = "sensor_data"
        }
    ]
    
    # Parameters
    parameters = {
        # Edge detection threshold
        "edge_threshold" = 30
        
        # Performance tuning
        "gpu_block_size" = {
            "x" = 16
            "y" = 16
        }
        
        # Enable/disable features
        "enable_gpu" = true
        "enable_debug_output" = false
    }
    
    # CUDA kernel configuration
    cuda_kernel edge_detection {
        # Input parameters
        inputs = [
            { name: "input", type: "const uint8_t*" },
            { name: "output", type: "uint8_t*" },
            { name: "params", type: "const ImageParams*" },
            { name: "threshold", type: "int" }
        ]
        
        # Kernel configuration
        block_dim = "{block_size.x, block_size.y, 1}"
        grid_dim = "{(params.width + block_size.x - 1) / block_size.x, "
                 ."(params.height + block_size.y - 1) / block_size.y, 1}"
        
        # Shared memory configuration (if needed)
        # shared_mem = "sizeof(float) * (block_size.x + 2) * (block_size.y + 2)"
    }
    
    # Node lifecycle callbacks
    lifecycle {
        on_configure = "on_configure"
        on_activate = "on_activate"
        on_deactivate = "on_deactivate"
        on_cleanup = "on_cleanup"
        on_shutdown = "on_shutdown"
        on_error = "on_error"
    }
}

# Implementation
impl image_processor {
    #include "kernels/image_processing.cuh"
    #include <opencv2/opencv.hpp>
    #include <cv_bridge/cv_bridge.h>
    
    // Member variables
    std::shared_ptr<cv::Mat> debug_image_;
    uint8_t* d_input_ = nullptr;
    uint8_t* d_output_ = nullptr;
    ImageParams d_params_;
    
    bool on_configure() {
        // Initialize CUDA resources
        try {
            // Allocate GPU memory for input/output images
            // (actual size will be determined when we receive the first image)
            d_params_ = {0, 0, 1, 0};
            
            RCLCPP_INFO(get_logger(), "Node configured");
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Configuration failed: %s", e.what());
            return false;
        }
    }
    
    bool on_activate() {
        RCLCPP_INFO(get_logger(), "Node activated");
        return true;
    }
    
    bool on_deactivate() {
        RCLCPP_INFO(get_logger(), "Node deactivated");
        return true;
    }
    
    bool on_cleanup() {
        // Free GPU resources
        if (d_input_) {
            free_gpu_memory(d_input_);
            d_input_ = nullptr;
        }
        if (d_output_) {
            free_gpu_memory(d_output_);
            d_output_ = nullptr;
        }
        
        RCLCPP_INFO(get_logger(), "Node cleaned up");
        return true;
    }
    
    bool on_shutdown() {
        // Cleanup is handled by on_cleanup
        RCLCPP_INFO(get_logger(), "Node shutting down");
        return true;
    }
    
    void on_error() {
        RCLCPP_ERROR(get_logger(), "Error occurred, cleaning up...");
        on_cleanup();
    }
    
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
            } catch (cv_bridge::Exception& e) {
                RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
                return;
            }
            
            // Get parameters
            const bool use_gpu = get_parameter("enable_gpu").as_bool();
            const int threshold = get_parameter("edge_threshold").as_int();
            
            // Process image
            if (use_gpu) {
                process_on_gpu(cv_ptr->image, threshold);
            } else {
                process_on_cpu(cv_ptr->image, threshold);
            }
            
            // Publish result
            auto result_msg = cv_bridge::CvImage(
                msg->header,
                sensor_msgs::image_encodings::MONO8,
                cv_ptr->image
            ).toImageMsg();
            
            publishers.edges.publish(*result_msg);
            
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Error processing image: %s", e.what());
        }
    }
    
private:
    void process_on_gpu(cv::Mat& image, int threshold) {
        const int width = image.cols;
        const int height = image.rows;
        const size_t image_size = width * height * image.channels();
        
        // Reallocate GPU memory if image size changed
        if (d_params_.width != width || d_params_.height != height) {
            if (d_input_) free_gpu_memory(d_input_);
            if (d_output_) free_gpu_memory(d_output_);
            
            allocate_gpu_memory((void**)&d_input_, image_size);
            allocate_gpu_memory((void**)&d_output_, image_size);
            
            d_params_ = {
                .width = width,
                .height = height,
                .channels = image.channels(),
                .step = image.step
            };
        }
        
        // Copy input to GPU
        copy_to_gpu(image.data, d_input_, image_size);
        
        // Launch CUDA kernel
        const auto block_size = get_parameter("gpu_block_size").get<std::map<std::string, int>>();
        const dim3 block_dim(block_size.at("x"), block_size.at("y"));
        const dim3 grid_dim(
            (width + block_dim.x - 1) / block_dim.x,
            (height + block_dim.y - 1) / block_dim.y
        );
        
        edge_detection_kernel<<<grid_dim, block_dim>>>(
            d_input_, d_output_, d_params_, threshold
        );
        
        // Check for kernel launch errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA kernel launch failed: " + 
                                  std::string(cudaGetErrorString(err)));
        }
        
        // Copy result back to CPU
        copy_from_gpu(d_output_, image.data, image_size);
        
        // Synchronize to catch any kernel errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA error: " + 
                                  std::string(cudaGetErrorString(err)));
        }
    }
    
    void process_on_cpu(cv::Mat& image, int threshold) {
        // Convert to grayscale
        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        
        // Apply edge detection
        cv::Mat edges;
        cv::Canny(gray, edges, threshold, threshold * 3);
        
        // Convert back to color for visualization
        cv::cvtColor(edges, image, cv::COLOR_GRAY2BGR);
    }
};
```

### CMakeLists.txt Generation

The RoboDSL compiler will generate the following CMake configuration:

1. **CUDA Support**: Automatically detects CUDA and configures the build
2. **Dependencies**: Links against CUDA, OpenCV, and ROS2 libraries
3. **Install Rules**: Creates proper install targets for the node and its resources
4. **Component Export**: Exports the node as a component for composition

### Launch File Example

```python
# launch/image_processor.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='image_processing',
            executable='image_processor',
            namespace='camera',
            parameters=[{
                'enable_gpu': True,
                'edge_threshold': 30,
                'gpu_block_size.x': 16,
                'gpu_block_size.y': 16
            }],
            remappings=[
                ('image_raw', '/camera/image_raw'),
                ('edges', '/perception/edges')
            ]
        )
    ])
```

### Performance Considerations

1. **Memory Management**:
   - Minimize host-device memory transfers
   - Reuse device memory buffers when possible
   - Use pinned memory for asynchronous transfers

2. **Kernel Optimization**:
   - Choose optimal block dimensions (usually 16x16 or 32x32)
   - Use shared memory for data reuse
   - Minimize thread divergence

3. **CPU-GPU Overlap**:
   - Use CUDA streams for concurrent execution
   - Overlap memory transfers with computation

4. **Fallback Implementation**:
   - Always provide a CPU fallback for development and testing
   - Use feature flags to enable/disable GPU features at runtime

### Debugging Tips

1. **CUDA Error Checking**:
   ```c++
   #define CHECK_CUDA_ERROR(call) {\
       cudaError_t err = call;\
       if (err != cudaSuccess) {\
           throw std::runtime_error(\
               std::string("CUDA error: ") + \
               cudaGetErrorString(err) + \
               " at " + __FILE__ + ":" + std::to_string(__LINE__));\
       }\
   }
   
   // Usage
   CHECK_CUDA_ERROR(cudaMalloc(&d_ptr, size));
   ```

2. **Profiling**:
   - Use `nvprof` or Nsight Systems for performance analysis
   - Profile both kernel execution and memory transfers

3. **Debug Builds**:
   - Compile with `-G -g` for device debug symbols
   - Use `cuda-gdb` or Nsight for debugging

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

## Build Configuration

RoboDSL provides a powerful build configuration system that supports conditional compilation, dependency management, and platform-specific settings. This section covers the available build options and how to use them effectively.

### Core Build Options

```python
build {
    # C++ standard (default: 17)
    cpp_standard = 17  # 14, 17, 20, or 23
    
    # Build type (default: Release)
    build_type = "Release"  # Debug, Release, RelWithDebInfo, MinSizeRel
    
    # Enable/disable features
    features = {
        ros2 = true    # Enable ROS2 integration (ENABLE_ROS2)
        cuda = true    # Enable CUDA support (ENABLE_CUDA)
        opencv = true  # Enable OpenCV support
        debug = false  # Enable debug symbols and assertions
        tests = true   # Build tests
        examples = true # Build examples
    }
    
    # CUDA-specific settings (only used when cuda = true)
    cuda = {
        # Target compute architectures (comma-separated)
        architectures = ["sm_75", "sm_80"]  # Turing, Ampere
        
        # CUDA toolkit path (auto-detected if not specified)
        # toolkit_path = "/usr/local/cuda"
        
        # Additional CUDA compilation flags
        flags = ["-O3", "--use_fast_math"]
    }
    
    # Compiler flags
    compile_options = {
        cpp = [
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-parameter",
            "-Wno-missing-field-initializers"
        ],
        cuda = [
            "-Xcompiler", "-fPIC",
            "--extended-lambda",
            "--expt-relaxed-constexpr"
        ]
    }
    
    # Linker flags
    link_options = [
        "-Wl,--as-needed",
        "-Wl,--no-undefined"
    ]
    
    # Dependencies
    find_packages = [
        "rclcpp",
        "std_msgs",
        "sensor_msgs",
        "nav_msgs",
        "tf2_ros",
        "cv_bridge",
        "OpenCV"
    ]
    
    # Additional include directories
    include_directories = [
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ]
    
    # Additional link directories
    link_directories = [
        "${CMAKE_CURRENT_BINARY_DIR}"
    ]
    
    # Additional libraries to link against
    link_libraries = [
        "pthread",
        "dl"
    ]
    
    # Install configuration
    install = {
        # Install target (default: all)
        components = ["runtime", "development", "python"]
        
        # Install prefix (default: /usr/local)
        prefix = "${CMAKE_INSTALL_PREFIX}"
        
        # Additional install rules
        rules = [
            {
                type = "directory"
                destination = "share/${PROJECT_NAME}/launch"
                files = ["launch/*.launch.py"]
            },
            {
                type = "directory"
                destination = "share/${PROJECT_NAME}/config"
                files = ["config/*.yaml"]
            }
        ]
    }
}

### Conditional Compilation

Use the `ENABLE_ROS2` and `ENABLE_CUDA` flags to conditionally compile code:

```python
# In your .robodsl file
build {
    features = {
        ros2 = true
        cuda = true
    }
}

# In your C++ code
#ifdef ENABLE_ROS2
// ROS2-specific code
#include <rclcpp/rclcpp.hpp>
#endif

#ifdef ENABLE_CUDA
// CUDA-specific code
#include <cuda_runtime.h>
#endif

class MyNode {
public:
    void process() {
        #ifdef ENABLE_CUDA
        // CUDA-accelerated implementation
        process_with_cuda();
        #else
        // CPU fallback implementation
        process_on_cpu();
        #endif
    }
}

### Cross-Platform Support

RoboDSL supports cross-platform development with platform-specific settings:

```python
build {
    # Platform-specific settings
    platform = {
        linux = {
            compile_options = ["-fPIC"]
            link_libraries = ["rt"]
        },
        windows = {
            compile_options = ["/W4", "/WX"]
            link_libraries = ["ws2_32.lib"]
        },
        macos = {
            compile_options = ["-stdlib=libc++"]
            link_libraries = []
        }
    }
    
    # Architecture-specific settings
    architecture = {
        x86_64 = {
            compile_options = ["-march=native"]
        },
        aarch64 = {
            compile_options = ["-mcpu=native"]
        }
    }
}

### Build System Integration

RoboDSL generates standard CMake files that can be integrated with existing build systems:

```bash
# Configure the build
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON

# Build the project
make -j$(nproc)

# Install the project
make install

# Create a debian package
cpack -G DEB

### Best Practices

1. **Feature Flags**:
   - Use `ENABLE_ROS2` and `ENABLE_CUDA` to enable/disable features
   - Provide fallback implementations when possible
   - Document feature dependencies in your package.xml

2. **Dependency Management**:
   - List all required dependencies in `find_packages`
   - Use version constraints when necessary
   - Document optional dependencies

3. **Cross-Platform Development**:
   - Use platform-agnostic code when possible
   - Test on all target platforms
   - Handle platform-specific dependencies gracefully

4. **Installation**:
   - Follow the FHS for file locations
   - Include necessary runtime dependencies
   - Generate and install package configuration files

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
