# robodsl_package

Generated ROS2 package from RoboDSL specification.

## Overview

This package contains 3 ROS2 nodes and 1 standalone CUDA kernels.

## Nodes

### comprehensive_processor

- **Type**: Lifecycle Node
- **Publishers**: 2
- **Subscribers**: 2
- **Services**: 1
- **Actions**: 1
- **Timers**: 3
- **Parameters**: 9
- **CUDA Kernels**: 1

### simple_publisher

- **Type**: Standard Node
- **Publishers**: 1
- **Subscribers**: 0
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 2
- **CUDA Kernels**: 0

### data_processor

- **Type**: Standard Node
- **Publishers**: 1
- **Subscribers**: 1
- **Services**: 0
- **Actions**: 0
- **Timers**: 0
- **Parameters**: 5
- **CUDA Kernels**: 0

## Standalone CUDA Kernels

### neural_network_kernel

- **Parameters**: 5
- **Block Size**: 256
- **Use Thrust**: True

## Building

```bash
colcon build
```

## Running

```bash
# Source the workspace
source install/setup.bash

# Launch all nodes
ros2 launch {package_name} main_launch.py

# Or launch individual nodes
ros2 launch {package_name} <node_name>_launch.py
```

## License

Apache-2.0
