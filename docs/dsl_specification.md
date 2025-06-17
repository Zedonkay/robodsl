# RoboDSL Language Specification

## Introduction

RoboDSL (Robot Domain-Specific Language) is a high-level language designed for building robust, performant, and maintainable robot applications. It provides a clean, declarative syntax for defining ROS2 nodes, components, and their interactions, with built-in support for advanced features like lifecycle management, QoS configuration, and GPU acceleration.

### Key Features

- **ROS2 Lifecycle Node Support**: Built-in support for managed nodes with configurable lifecycle states and transitions
- **Quality of Service (QoS) Configuration**: Fine-grained control over communication reliability, durability, and resource usage
- **Namespace and Remapping**: Flexible namespace management and topic/service remapping
- **CUDA Offloading**: Seamless integration of GPU-accelerated computations
- **Conditional Compilation**: Feature flags for building different configurations from the same codebase
- **Component-Based Architecture**: Modular design for better code organization and reuse
- **Type Safety**: Strong typing for messages, services, and parameters
- **Build System Integration**: Native CMake integration with support for cross-platform development

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
- **Parameter Management**: Automatic parameter handling with validation
- **Namespacing**: Support for nested namespaces
- **Remapping**: Topic and service name remapping

#### Complete Example
```python
## Lifecycle Nodes

Lifecycle nodes in RoboDSL provide a structured way to manage the state and resources of your ROS2 nodes. They follow the ROS2 managed node pattern, allowing for controlled state transitions and better system management.

### Basic Lifecycle Node Definition

```python
lifecycle_node my_lifecycle_node {
    # Node identification
    namespace = "robot1"
    
    # Enable automatic parameter declaration and handling
    automatically_declare_parameters_from_overrides = true
    allow_undeclared_parameters = false
    
    # Remap topics/services
    remap = {
        "cmd_vel": "cmd_vel_nav",
        "/camera/image_raw": "/sensors/camera/front/image_raw"
    }
    
    # Autostart configuration (default: false)
    autostart = true
    
    # State QoS configuration
    state_qos = {
        reliability = "reliable"
        durability = "transient_local"
        depth = 10
    }
    
    # Parameters
    parameters = [
        {
            name = "max_speed"
            type = "double"
            default = 1.0
            description = "Maximum speed in m/s"
            range = { min: 0.1, max: 5.0, step: 0.1 }
        },
        {
            name = "sensor_frame"
            type = "string"
            default = "base_laser"
            description = "TF frame for sensor data"
        }
    ]
    
    # Lifecycle callbacks (all optional)
    on_configure = "on_configure_callback"
    on_activate = "on_activate_callback"
    on_deactivate = "on_deactivate_callback"
    on_cleanup = "on_cleanup_callback"
    on_shutdown = "on_shutdown_callback"
    on_error = "on_error_callback"
}
    allow_undeclared_parameters = false
    automatically_declare_parameters_from_overrides = true
    
    # Topic and service remapping
    remap = {
        "cmd_vel": "cmd_vel_nav",
        "/camera/image_raw": "/sensors/camera/front/image_raw"
    }
    
    # Lifecycle configuration
    autostart = true  # Automatically transition to Active state
    
    # Lifecycle callbacks (all optional)
    on_configure = "on_configure"      # Called during transition to Inactive
    on_cleanup = "on_cleanup"          # Called during transition to Unconfigured
    on_activate = "on_activate"        # Called during transition to Active
    on_deactivate = "on_deactivate"    # Called during transition to Inactive
    on_shutdown = "on_shutdown"        # Called during transition to Finalized
    on_error = "on_error_callback"     # Called on error transitions
    
    # QoS configuration for lifecycle state transitions
    state_qos = {
        reliability = "reliable"
        durability = "transient_local"
        depth = 10
    }
    
    # Parameters with descriptions and constraints
    parameters = [
        {
            name = "max_speed"
            type = "double"
            default = 1.0
            description = "Maximum speed in m/s"
            range = { min: 0.1, max: 5.0, step: 0.1 }
        },
        {
            name = "sensor_frame"
            type = "string"
            default = "base_laser"
            description = "TF frame for sensor data"
        }
    ]
}
```

#### Lifecycle States
1. **Unconfigured**: 
   - Initial state, no resources allocated
   - Node parameters are declared but not accessible
   - Only configuration-related operations allowed

2. **Inactive**: 
   - Node is configured but not active
   - Resources are allocated but not processing
   - Parameters are accessible
   - Publishers/Subscribers are inactive

3. **Active**: 
   - Node is fully operational
   - Processing callbacks are active
   - Publishers/Subscribers are active
   - Timers are running

4. **Finalized**: 
   - Node is shutting down
   - All resources are released
   - No further state transitions possible

#### Error Handling
Lifecycle nodes provide robust error handling through the `on_error` callback:

```python
def on_error_callback(state: LifecycleNodeState) -> CallbackReturn:
    """
    Handle error state transitions.
    
    Args:
        state: The state that was active when the error occurred
        
    Returns:
        CallbackReturn: `SUCCESS` if error was handled, `FAILURE` otherwise
    """
    error_msg = f"Error in state: {state.label()}"
    
    if state.id() == State.PRIMARY_STATE_ACTIVATING:
        # Handle activation errors
        get_logger().error(f"{error_msg} - Activation failed")
        # Attempt recovery or return FAILURE to trigger shutdown
        return CallbackReturn.FAILURE
        
    elif state.id() == State.PRIMARY_STATE_ACTIVE:
        # Handle runtime errors
        get_logger().error(f"{error_msg} - Runtime error")
        # Try to deactivate gracefully
        return CallbackReturn.SUCCESS
        
    return CallbackReturn.FAILURE
```

#### Best Practices
1. **Resource Management**:
   - Allocate resources in `on_configure`
   - Acquire hardware in `on_activate`
   - Release hardware in `on_deactivate`
   - Clean up resources in `on_cleanup` and `on_shutdown`

2. **Error Handling**:
   - Always implement `on_error` for robust error recovery
   - Log detailed error messages
   - Return appropriate status codes
   
3. **State Transitions**:
   - Keep transition logic simple and fast
   - Avoid blocking operations in transition callbacks
   - Use async operations when necessary
   
4. **Parameters**:
   - Validate parameters in `on_configure`
   - Use parameter constraints where possible
   - Document all parameters with descriptions and units
   
5. **Threading**:
   - Be aware of thread safety in callbacks
   - Use thread-safe data structures
   - Protect shared resources with mutexes
   
6. **Logging**:
   - Use appropriate log levels (DEBUG, INFO, WARN, ERROR, FATAL)
   - Include context in log messages
   - Use rate-limited logging for high-frequency messages

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

### Quality of Service (QoS)

RoboDSL provides fine-grained control over Quality of Service (QoS) settings for all ROS2 communication channels. QoS policies determine how messages are handled in terms of reliability, durability, and resource usage, allowing you to optimize communication for different types of data and network conditions.

#### QoS Policies

QoS policies can be configured at different levels:

1. **Global Defaults**: Set in the node configuration
2. **Per Communication Channel**: Override for specific publishers/subscribers/services
3. **Runtime Overrides**: Modify QoS settings dynamically

#### Policy Types

1. **Reliability**:
   - `reliable`: Ensures message delivery with retries (TCP-like)
     - Use for: Commands, critical control messages
     - Impact: Higher latency, guaranteed delivery
   - `best_effort`: No delivery guarantees (UDP-like)
     - Use for: High-frequency sensor data where occasional loss is acceptable
     - Impact: Lower latency, potential message loss
   - **Default**: `reliable`

2. **Durability**:
   - `volatile`: Messages not persisted for late-joining subscribers
     - Use for: Real-time data where only the latest message matters
   - `transient_local`: Last message stored for late-joining subscribers
     - Use for: Parameter servers, configuration data
   - **Default**: `volatile`

3. **History**:
   - `keep_last`: Store last N messages
     - `depth`: Number of messages to keep in the queue
     - Use for: Most use cases where only recent data is relevant
   - `keep_all`: Store all messages (use with caution)
     - Use for: Critical data where no loss is acceptable
   - **Default**: `keep_last` with depth 10

4. **Deadline**:
   - Expected maximum time between messages
   - Format: Duration string (e.g., "100ms", "1s")
   - Use for: Real-time systems with strict timing requirements

5. **Lifespan**:
   - Maximum time a message is considered valid
   - Format: Duration string
   - Use for: Stale data detection and cleanup

6. **Liveliness**:
   - `automatic`: Node is alive if process is running
   - `manual_by_topic`: Node must signal liveliness per topic
   - `manual_by_node`: Node must signal liveliness for all topics
   - **Default**: `automatic`
   - Use for: Failure detection and system monitoring

7. **Liveliness Lease Duration**:
   - Time after which a node is considered not alive if no liveliness signal is received
   - Format: Duration string
   - Must be greater than the expected liveliness signal period

#### Predefined QoS Profiles

RoboDSL provides several predefined QoS profiles for common use cases:

```python
# Sensor Data (best effort, small queue)
sensor_data_qos = {
    reliability = "best_effort"
    durability = "volatile"
    history = { kind: "keep_last", depth: 5 }
    deadline = "100ms"
}

# Commands (reliable, transient local for late joiners)
command_qos = {
    reliability = "reliable"
    durability = "transient_local"
    history = { kind: "keep_last", depth: 10 }
    deadline = "1s"
}

# Parameters (reliable, persistent)
parameter_qos = {
    reliability = "reliable"
    durability = "transient_local"
    history = { kind: "keep_all" }
    deadline = "10s"
}

# Services (reliable, volatile)
service_qos = {
    reliability = "reliable"
    durability = "volatile"
    history = { kind: "keep_last", depth: 10 }
    deadline = "5s"
}

# Action Servers
action_qos = {
    goal_service = service_qos
    result_service = service_qos
    cancel_service = service_qos
    feedback_topic = sensor_data_qos
    status_topic = parameter_qos
}

1. **Reliability**:
   - `reliable`: Ensures message delivery with retries (TCP-like)
   - `best_effort`: No delivery guarantees (UDP-like)
   - **Default**: `reliable`
   - **Use Case**: Use `reliable` for commands, `best_effort` for high-frequency sensor data

2. **Durability**:
   - `volatile`: Messages not persisted for late-joining subscribers
   - `transient_local`: Last message stored for late-joining subscribers
   - **Default**: `volatile`
   - **Use Case**: Use `transient_local` for parameter servers, `volatile` for real-time data

3. **History**:
   - `keep_last`: Store last N messages
   - `keep_all`: Store all messages (use with caution)
   - **Default**: `keep_last` with depth 10
   - **Use Case**: `keep_last` for most cases, `keep_all` for critical data

4. **Deadline**:
   - Expected maximum time between messages
   - **Format**: Duration string (e.g., "100ms", "1s")
   - **Use Case**: Detect missing sensor updates

5. **Lifespan**:
   - Maximum time a message is considered valid
   - **Format**: Duration string
   - **Use Case**: Stale data detection

6. **Liveliness**:
   - `automatic`: Node is alive if process is running
   - `manual_by_topic`: Node must signal liveliness per topic
   - `manual_by_node`: Node must signal liveliness for all topics
   - **Default**: `automatic`
   - **Use Case**: Detect node failures

#### QoS Profiles

RoboDSL provides predefined QoS profiles for common use cases:

```python
# Sensor Data (best effort, small queue)
sensor_data_qos = {
    reliability = "best_effort"
    durability = "volatile"
    history = "keep_last"
    depth = 5
}

# Commands (reliable, transient local for late joiners)
command_qos = {
    reliability = "reliable"
    durability = "transient_local"
    history = "keep_last"
    depth = 10
}

# Parameters (reliable, persistent)
parameter_qos = {
    reliability = "reliable"
    durability = "transient_local"
    history = "keep_all"
}

# Services (reliable, volatile)
service_qos = {
    reliability = "reliable"
    durability = "volatile"
    history = "keep_last"
    depth = 10
}
```

#### QoS Configuration

QoS settings can be configured at different levels:

1. **Global Defaults**: Set in the node configuration
2. **Per Communication Channel**: Override for specific publishers/subscribers
3. **Runtime Overrides**: Modify QoS settings dynamically

Example with multiple QoS configurations:

```python
node sensor_processor {
    # Default QoS for all publishers/subscribers
    default_qos = {
        reliability = "best_effort"
        durability = "volatile"
        depth = 5
    }
    
    publishers = [
        {
            name = "processed_data"
            type = "sensor_msgs/msg/Image"
            topic = "processed_image"
            # Use default QoS
        },
        {
            name = "alerts"
            type = "std_msgs/msg/String"
            topic = "alerts"
            # Override QoS for critical alerts
            qos = {
                reliability = "reliable"
                durability = "transient_local"
                depth = 100
            }
        }
    ]
    
    subscribers = [
        {
            name = "raw_data"
            type = "sensor_msgs/msg/Image"
            topic = "camera/image_raw"
            # Use sensor-specific QoS
            qos = sensor_data_qos
        }
    ]
}
```

#### Best Practices

1. **Matching QoS**: Ensure publishers and subscribers have compatible QoS settings
2. **Resource Management**: Be mindful of memory usage with large queues and `keep_all` history
3. **Performance**: Use `best_effort` for high-frequency data where occasional loss is acceptable
4. **Reliability**: Use `reliable` for critical commands and configuration
5. **Monitoring**: Monitor QoS compatibility warnings in the console
6. **Testing**: Test with different network conditions and system loads

#### Debugging QoS Issues

Common issues and solutions:

1. **No Communication**: Check if QoS profiles are compatible between publisher and subscriber
2. **Missed Messages**: Increase queue depth or adjust reliability settings
3. **High Latency**: Reduce queue depth or switch to `best_effort` for non-critical data
4. **Memory Usage**: Monitor memory usage with large queues or `keep_all` history

#### Example: Dynamic QoS Reconfiguration

```python
# In a node implementation
def update_qos_settings(self):
    # Get current QoS settings
    qos = self.get_effective_qos("processed_data")
    
    # Modify settings
    qos.depth = 20
    qos.deadline = Duration(seconds=1)
    
    # Apply new settings
    self.update_qos("processed_data", qos)
```

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
```python
# In your code
def process_pointcloud_callback(goal_handle):
    """
    Process point cloud data with CUDA acceleration if available.
    
    Args:
        goal_handle: The action goal handle containing request data
    """
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
        raise
```
            output_mapping = [
                { kernel_param: "result_path", action_field: "result.path" },
                { kernel_param: "feedback_progress", action_field: "feedback.progress" }
            ]
{{ ... }}
# Main node
node robot_controller {
    # Node configuration
    namespace = "robot"
    executable = true
    ```python
# Example of a simple node
node my_node {
    executable = true
    namespace = "sensors"
    
    # Publishers
    publishers = [
        {
            name = "temperature"
            type = "std_msgs/msg/Float32"
            qos = "sensor_data"
        }
    ]
    
    # Subscribers
    subscribers = [
        {
            name = "command"
            type = "std_msgs/msg/String"
            callback = "command_callback"
            qos = "default"
        }
    ]
    
    # Parameters
    parameters = [
        {
            name = "update_rate"
            type = "double"
            default = 1.0
            description = "Update rate in Hz"
        }
    ]
}
```cuda_kernel process_sensors {
        inputs = [
            { name: "sensor_data", type: "const SensorData&" }
        ]
        outputs = [
            { name: "processed_data", type: "ProcessedData&" }
{{ ... }}
        
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

## Troubleshooting and FAQ

This section provides solutions to common issues and answers to frequently asked questions about RoboDSL.

### Common Issues

#### 1. CUDA Kernel Compilation Failures

**Symptom**: Errors during CUDA kernel compilation with messages about unsupported architectures or syntax errors.

**Solution**:
- Ensure your GPU's compute capability is correctly specified in the build configuration:
  ```python
  build_options = {
      "CUDA_ARCH": ["sm_75"],  # Adjust for your GPU architecture
      # ...
  }
  ```
- Verify CUDA toolkit version compatibility with your GPU
- Check for syntax errors in kernel code, especially with device-specific keywords

#### 2. ROS2 Lifecycle Node State Transition Failures

**Symptom**: Nodes getting stuck during state transitions (e.g., configuring → inactive).

**Solution**:
- Implement proper error handling in all lifecycle callbacks
- Ensure `on_configure()` and `on_activate()` complete within the timeout period
- Check for unhandled exceptions in callbacks
- Verify all required parameters are properly declared

#### 3. QoS Incompatibility Warnings

**Symptom**: Warnings about incompatible QoS settings between publishers and subscribers.

**Solution**:
- Ensure matching QoS profiles between publishers and subscribers
- Use compatible reliability and durability settings:
  ```python
  # Example of compatible QoS settings
  qos_profile "custom_qos" {
      reliability = "reliable"
      durability = "volatile"  # or "transient_local" for late-joining subscribers
      history = "keep_last"
      depth = 10
  }
  ```

### Frequently Asked Questions

#### Q1: How do I enable CUDA support in my project?

**A**: Enable CUDA in your build configuration:
```python
build_options = {
    "ENABLE_CUDA": true,
    "CUDA_ARCH": ["sm_75"],  # Adjust for your GPU
    # ...
}
```

Then declare CUDA kernels in your RoboDSL files and use them in your nodes.

#### Q2: What's the difference between `volatile` and `transient_local` durability?

- **Volatile**: Messages are not stored for late-joining subscribers
- **Transient Local**: Last message is stored for late-joining subscribers

Use `transient_local` for parameters and important state updates, and `volatile` for high-frequency sensor data.

#### Q3: How do I handle different configurations for simulation vs. real hardware?

Use conditional compilation and parameters:

```python
# In your node definition
parameters = [
    {
        name = "use_simulation"
        type = "bool"
        default = false
        description = "Whether to use simulation mode"
    },
    # ...
]

# In your implementation
initialize = """
void initialize() {
    bool use_sim = this->get_parameter("use_simulation").as_bool();
    if (use_sim) {
        // Simulation-specific initialization
    } else {
        // Hardware-specific initialization
    }
}
"""
```

#### Q4: How can I improve the performance of my CUDA kernels?

- Use shared memory for data reuse
- Ensure memory coalescing for global memory access
- Minimize thread divergence in warps
- Use constant memory for read-only data
- Profile with Nsight Systems to identify bottlenecks

#### Q5: How do I debug issues in my RoboDSL nodes?

1. Enable debug logging:
   ```python
   node my_node {
       log_level = "debug"
       # ...
   }
   ```

2. Use the `rqt_console` tool to view ROS2 logs
3. For CUDA issues, use `cuda-memcheck` and Nsight tools
4. Enable core dumps for segmentation faults

#### Q6: How do I handle version compatibility between RoboDSL and ROS2 distributions?

Specify version constraints in your project configuration:

```python
dependencies = [
    "rclcpp >= 2.4.0",
    "std_msgs >= 4.2.0",
    # ...
]

# In your build configuration
build_options = {
    "ROS2_DISTRO": "humble",  # Or your target distribution
    # ...
}
```

### Getting Help

If you encounter issues not covered here:
1. Check the [GitHub Issues](https://github.com/yourorg/robodsl/issues) for similar problems
2. Search the [ROS Answers](https://answers.ros.org/) forum
3. For bugs, open a new issue with:
   - RoboDSL version
   - ROS2 distribution and version
   - Steps to reproduce
   - Error messages and logs
   - Relevant code snippets

## Conclusion

This document provides a comprehensive reference for the RoboDSL language. For more examples and tutorials, see the [examples](examples/) directory in the RoboDSL repository.
