# RoboDSL Language Specification

## Introduction

RoboDSL (Robot Domain-Specific Language) is a high-level language designed for building robust, performant, and maintainable robot applications. It provides a clean, declarative syntax for defining ROS2 nodes, components, and their interactions, with built-in support for advanced features like lifecycle management, QoS configuration, and GPU acceleration.

### Version
This specification applies to RoboDSL version 0.1.0 and later.

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
1. [Syntax Overview](#syntax-overview)
2. [Project Definition](#project-definition)
3. [Node Definition](#node-definition)
   - [Node Types](#node-types)
   - [Lifecycle Nodes](#lifecycle-nodes)
   - [Component Nodes](#component-nodes)
4. [Communication](#communication)
   - [Publishers](#publishers)
   - [Subscribers](#subscribers)
   - [Services](#services)
   - [Actions](#actions)
5. [Parameters](#parameters)
6. [CUDA Integration](#cuda-integration)
7. [Build System](#build-system)
8. [FAQ](#faq)

## Syntax Overview

RoboDSL uses a C-like syntax with a focus on readability and simplicity. The language is statically typed and supports both imperative and declarative programming styles.

### Basic Structure

```robodsl
// Single-line comment
/* Multi-line
   comment */

// Import other RoboDSL files
import "common.robodsl"

// Project definition
project "my_robot" {
    version = "0.1.0"
    description = "My Robot Application"
}

// Node definition
node my_node {
    // Node configuration
    namespace = "robot1"
    enable_lifecycle = true
    
    // Publishers
    publishers = [
        {
            name = "odom"
            type = "nav_msgs/msg/Odometry"
            qos = {
                reliability = "reliable"
                durability = "transient_local"
                depth = 10
            }
        }
    ]
    
    // Subscribers
    subscribers = [
        {
            name = "cmd_vel"
            type = "geometry_msgs/msg/Twist"
            callback = "on_cmd_vel"
        }
    ]
    
    // CUDA kernels
    cuda_kernels = ["process_image"]
}
```

## Project Definition

Every RoboDSL file must begin with a project definition that specifies the project name and version.

```robodsl
project "my_robot" {
    version = "0.1.0"
    description = "My Robot Application"
    license = "Apache-2.0"
    authors = ["Your Name <your.email@example.com>"]
}
```

## Node Definition

Nodes are the fundamental building blocks of a RoboDSL application. They represent individual components that can communicate with each other through topics, services, and actions.

### Basic Node

```robodsl
node my_node {
    // Node configuration
    namespace = "robot1"
    enable_lifecycle = true
    
    // Publishers
    publishers = [
        {
            name = "odom"
            type = "nav_msgs/msg/Odometry"
        }
    ]
}
```

### Lifecycle Nodes

Lifecycle nodes provide a structured way to manage the state and resources of your ROS2 nodes. They follow the ROS2 managed node pattern, allowing for controlled state transitions and better system management.

```robodsl
node my_lifecycle_node {
    enable_lifecycle = true
    
    // Lifecycle callbacks
    on_configure = "on_configure"
    on_activate = "on_activate"
    on_deactivate = "on_deactivate"
    on_cleanup = "on_cleanup"
    on_shutdown = "on_shutdown"
    
    // Error handling
    on_error = "on_error"
}
```

## Communication

### Publishers

Publishers allow nodes to send messages to specific topics.

```robodsl
publishers = [
    {
        name = "odom"
        type = "nav_msgs/msg/Odometry"
        qos = {
            reliability = "reliable"
            durability = "transient_local"
            depth = 10
        }
    }
]
```

### Subscribers

Subscribers receive messages from topics and invoke callback functions.

```robodsl
subscribers = [
    {
        name = "cmd_vel"
        type = "geometry_msgs/msg/Twist"
        callback = "on_cmd_vel"
        qos = {
            reliability = "best_effort"
            durability = "volatile"
            depth = 1
        }
    }
]
```

## CUDA Integration

RoboDSL provides first-class support for CUDA acceleration. You can define CUDA kernels directly in your node definitions.

```robodsl
node image_processor {
    // Enable CUDA support
    cuda_kernels = ["process_image"]
    
    // CUDA source files
    cuda_sources = ["src/kernels/image_processing.cu"]
    
    // CUDA compilation flags
    cuda_flags = ["-O3", "--ptxas-options=-v"]
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
