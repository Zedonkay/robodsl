# robodsl_package

Generated ROS2 package from RoboDSL specification

## Overview

This package contains 2 ROS2 nodes and 3 CUDA kernels.

## Package Structure

```
robodsl_package/
├── CMakeLists.txt
├── package.xml
├── README.md
├── src/
│   └── robodsl_package/
│       ├── autonomous_navigator.cpp
│       ├── autonomous_navigator.hpp
│       ├── robot_localizer.cpp
│       ├── robot_localizer.hpp
│       ├── point_cloud_filter.cu
│       ├── point_cloud_filter.cuh
│       ├── occupancy_grid_update.cu
│       ├── occupancy_grid_update.cuh
│       ├── path_planning_astar.cu
│       └── path_planning_astar.cuh
├── launch/
│   ├── robodsl_package_launch.py
│   ├── autonomous_navigator_launch.py
│   └── robot_localizer_launch.py
├── include/
│   └── robodsl_package/
│       ├── autonomous_navigator.hpp
│       └── robot_localizer.hpp
├── config/
│   └── params.yaml
└── test/
    └── test_robodsl_package.py
```

## Nodes

### autonomous_navigator

- **Type**: Lifecycle Node- **Publishers**: 5
- **Subscribers**: 4
- **Services**: 0
- **Actions**: 0
- **Timers**: 2
- **Parameters**: 12

### robot_localizer

- **Type**: Standard Node- **Publishers**: 2
- **Subscribers**: 3
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 3


## Standalone CUDA Kernels

### point_cloud_filter

- **Parameters**: 6
- **Block Size**: 256
- **Use Thrust**: False

### occupancy_grid_update

- **Parameters**: 8
- **Block Size**: 16
- **Use Thrust**: True

### path_planning_astar

- **Parameters**: 10
- **Block Size**: 256
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
ros2 launch robodsl_package autonomous_navigator_launch.py
ros2 launch robodsl_package robot_localizer_launch.py
```

## License

Apache-2.0 