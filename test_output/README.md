# robodsl_package

Generated ROS2 package from RoboDSL specification.

## Overview

This package contains 0 ROS2 nodes and 3 standalone CUDA kernels.

## Nodes

## Standalone CUDA Kernels

### vector_add

- **Parameters**: 3
- **Block Size**: 256
- **Use Thrust**: False

### matrix_multiply

- **Parameters**: 3
- **Block Size**: 16
- **Use Thrust**: False

### image_filter

- **Parameters**: 3
- **Block Size**: 32
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
