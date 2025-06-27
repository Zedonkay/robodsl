# robodsl_package

Generated ROS2 package from RoboDSL specification.

## Overview

This package contains 1 ROS2 nodes and 0 standalone CUDA kernels.

## Nodes

### TestNode

- **Type**: Standard Node
- **Publishers**: 0
- **Subscribers**: 0
- **Services**: 0
- **Actions**: 0
- **Timers**: 0
- **Parameters**: 0
- **CUDA Kernels**: 1

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
