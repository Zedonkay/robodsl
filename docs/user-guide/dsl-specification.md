# RoboDSL DSL Reference

> **Version**: 0.1.0+

RoboDSL is a high-level language for building robot applications with ROS2 and CUDA.

## Key Features

- ROS2 Lifecycle nodes & components
- QoS configuration
- Namespace/remapping
- CUDA integration
- Type safety
- CMake integration

## Quick Start

```robodsl
// Project definition
project "my_robot" {
    version = "0.1.0"
}

// Node with publisher and subscriber
node my_node {
    namespace = "robot1"
    
    publishers = [{
        name = "odom"
        type = "nav_msgs/msg/Odometry"
    }]
    
    subscribers = [{
        name = "cmd_vel"
        type = "geometry_msgs/msg/Twist"
        callback = "on_cmd_vel"
    }]
}
```

## Full Reference

### Project Definition

```robodsl
project "name" {
    version = "0.1.0"
    description = "..."
    license = "..."
    authors = ["..."]
}
```

### Node Definition

```robodsl
node name {
    namespace = "..."
    enable_lifecycle = true/false
}
```

### Lifecycle Node

```robodsl
node name {
    enable_lifecycle = true
    on_configure = "..."
    on_activate = "..."
    on_deactivate = "..."
    on_cleanup = "..."
    on_shutdown = "..."
    on_error = "..."
}
```

### Publishers

```robodsl
publishers = [{
    name = "topic_name"
    type = "pkg/msg/Type"
    qos = {
        reliability = "reliable|best_effort"
        durability = "volatile|transient_local"
        depth = 10
    }
}]
```

### Subscribers

```robodsl
subscribers = [{
    name = "topic_name"
    type = "pkg/msg/Type"
    callback = "function_name"
    qos = { ... }  // same as publisher
}]
```

### Services

```robodsl
services = [{
    name = "service_name"
    type = "pkg/srv/Type"
    callback = "function_name"
}]
```

### Actions

```robodsl
actions = [{
    name = "action_name"
    type = "pkg/action/Type"
    execute_callback = "execute_cb"
    goal_callback = "goal_cb"
    cancel_callback = "cancel_cb"
}]
```

### Parameters

```robodsl
parameters = [{
    name = "param_name"
    type = "int|double|string|bool"
    value = 42
    description = "..."
}]
```

### CUDA Integration

```robodsl
node gpu_node {
    cuda_kernels = ["kernel1", "kernel2"]
    cuda_sources = ["kernels.cu"]
    cuda_flags = ["-O3"]
}
```

## Build Configuration

```robodsl
build {
    cpp_std = "17"
    cmake_min_version = "3.16"
    dependencies = [
        "rclcpp",
        "std_msgs"
    ]
}
```

## Full Example

```robodsl
project "robot_arm" {
    version = "1.0.0"
    description = "Robot Arm Control"
}

node arm_controller {
    namespace = "arm"
    enable_lifecycle = true
    
    publishers = [{
        name = "joint_states"
        type = "sensor_msgs/msg/JointState"
    }]
    
    subscribers = [{
        name = "joint_commands"
        type = "trajectory_msgs/msg/JointTrajectory"
        callback = "on_joint_commands"
    }]
    
    services = [{
        name = "calibrate"
        type = "std_srvs/srv/Trigger"
        callback = "on_calibrate"
    }]
    
    parameters = [{
        name = "max_velocity"
        type = "double"
        value = 1.0
    }]
}
```

## Build System

RoboDSL generates CMake files for building your application. The build system supports:

- Cross-platform compilation
- Dependency management
- CUDA compilation
- Installation rules
- Testing

### Example CMake Configuration

```cmake
# Minimum required CMake version
cmake_minimum_required(VERSION 3.16)


# Project name and version
project(my_robot VERSION 0.1.0)


# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

# Add your node executables
add_executable(${PROJECT_NAME}_node
  src/my_node.cpp
)

target_include_directories(${PROJECT_NAME}_node
  PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME}_node
  rclcpp
  std_msgs::std_msgs
)

# Install targets
install(TARGETS
  ${PROJECT_NAME}_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

# Install config files
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}/
)

# Install parameter files
install(DIRECTORY
  params
  DESTINATION share/${PROJECT_NAME}/
)

# Export dependencies
ament_export_dependencies(
  rclcpp
  std_msgs
)

# Export include directory
ament_export_include_directories(include)

# Export libraries
ament_export_libraries(${PROJECT_NAME}_node)

# Export build type
ament_export_build_type()

# Install Python modules if any
install(PROGRAMS
  scripts/my_script.py
  DESTINATION lib/${PROJECT_NAME}
)

# Add tests if test directory exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test")
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  
  find_package(ament_cmake_gtest REQUIRED)
  
  # Add a test
  ament_add_gtest(test_my_node
    test/test_my_node.cpp
  )
  
  target_link_libraries(test_my_node
    ${PROJECT_NAME}_node
  )
  
  # Add a Python test if needed
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_my_python_node
    test/test_my_node.py
    PYTHON_EXECUTABLE "${PYTHON_EXECUTABLE}"
  )
endif()

# Install the package.xml
do_package_setup()
```

## FAQ

### How do I enable lifecycle management for a node?

Set `enable_lifecycle = true` in your node definition and implement the required callbacks.

### What QoS settings should I use for my publishers/subscribers?

- **Commands**: Use `reliable` reliability and `transient_local` durability
- **Sensor Data**: Use `best_effort` reliability and `volatile` durability
- **Parameters**: Use `reliable` reliability and `transient_local` durability
- **High-Frequency Data**: Use `best_effort` reliability and `volatile` durability with small queue sizes

### How do I add a new message type?

1. Define the message in a `.msg` file in the `msg` directory
2. Add the message to `CMakeLists.txt`
3. Rebuild your package
4. Use the new message type in your node definitions

### How do I debug my RoboDSL application?

1. Enable debug logging in your node configuration:
   ```robodsl
   node my_node {
       log_level = "debug"
   }
   ```
2. Use ROS2's built-in logging tools:
   ```bash
   ros2 run --prefix 'ros2 run --debug' my_package my_node
   ```
3. Use GDB for debugging:
   ```bash
   gdb --args ros2 run my_package my_node
   ```

### How do I profile my CUDA code?

1. Enable profiling in your CUDA configuration:
   ```robodsl
   node my_node {
       cuda_flags = ["-lineinfo", "-G", "-g"]
   }
   ```
2. Use NVIDIA Nsight Systems for profiling:
   ```bash
   nsys profile -o my_profile ./my_node
   ```
3. Use NVIDIA Nsight Compute for detailed kernel analysis:
   ```bash
   ncu -o my_kernel_profile ./my_node
   ```

### How do I handle different configurations for simulation vs. real hardware?

Use conditional compilation and parameters:

```robodsl
node my_node {
    // Default to simulation mode
    param use_simulation = true
    
    // Configure based on simulation flag
    if (use_simulation) {
        // Simulation-specific configuration
        publishers = [
            {
                name = "sim/odom"
                type = "nav_msgs/msg/Odometry"
            }
        ]
    } else {
        // Hardware-specific configuration
        publishers = [
            {
                name = "odom"
                type = "nav_msgs/msg/Odometry"
            }
        ]
    }
}
```
