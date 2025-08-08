# robodsl_package

Generated ROS2 package from RoboDSL specification

## Overview

This package contains 5 ROS2 nodes and 3 CUDA kernels.

## Package Structure

```
robodsl_package/
├── CMakeLists.txt
├── package.xml
├── README.md
├── src/
│   └── robodsl_package/
│       ├── main_node.cpp
│       ├── main_node.hpp
│       ├── perception_node.cpp
│       ├── perception_node.hpp
│       ├── navigation_node.cpp
│       ├── navigation_node.hpp
│       ├── safety_node.cpp
│       ├── safety_node.hpp
│       ├── robot_cpp_node.cpp
│       ├── robot_cpp_node.hpp
│       ├── vector_add.cu
│       ├── vector_add.cuh
│       ├── matrix_multiply.cu
│       ├── matrix_multiply.cuh
│       ├── image_filter.cu
│       └── image_filter.cuh
├── launch/
│   ├── robodsl_package_launch.py
│   ├── main_node_launch.py
│   ├── perception_node_launch.py
│   ├── navigation_node_launch.py
│   ├── safety_node_launch.py
│   └── robot_cpp_node_launch.py
├── include/
│   └── robodsl_package/
│       ├── main_node.hpp
│       ├── perception_node.hpp
│       ├── navigation_node.hpp
│       ├── safety_node.hpp
│       └── robot_cpp_node.hpp
├── config/
│   └── params.yaml
└── test/
    └── test_robodsl_package.py
```

## Nodes

### main_node

- **Type**: Lifecycle Node- **Publishers**: 3
- **Subscribers**: 2
- **Services**: 1
- **Actions**: 0
- **Timers**: 2
- **Parameters**: 6

### perception_node

- **Type**: Standard Node- **Publishers**: 2
- **Subscribers**: 1
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 3

### navigation_node

- **Type**: Standard Node- **Publishers**: 2
- **Subscribers**: 1
- **Services**: 0
- **Actions**: 1
- **Timers**: 1
- **Parameters**: 3

### safety_node

- **Type**: Standard Node- **Publishers**: 2
- **Subscribers**: 2
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 3

### robot_cpp_node

- **Type**: Standard Node- **Publishers**: 1
- **Subscribers**: 1
- **Services**: 0
- **Actions**: 0
- **Timers**: 0
- **Parameters**: 1


## Standalone CUDA Kernels

### vector_add

- **Parameters**: 4
- **Block Size**: 256
- **Use Thrust**: True

### matrix_multiply

- **Parameters**: 6
- **Block Size**: 16
- **Use Thrust**: False

### image_filter

- **Parameters**: 4
- **Block Size**: 32
- **Use Thrust**: False


## Building

```bash
colcon build
```

## Running

```bash
# Source the workspace
source install/setup.bash

# Launch all nodes
ros2 launch robodsl_package robodsl_package_launch.py

# Or launch individual nodes
ros2 launch robodsl_package main_node_launch.py
ros2 launch robodsl_package perception_node_launch.py
ros2 launch robodsl_package navigation_node_launch.py
ros2 launch robodsl_package safety_node_launch.py
ros2 launch robodsl_package robot_cpp_node_launch.py
```

## License

Apache-2.0 