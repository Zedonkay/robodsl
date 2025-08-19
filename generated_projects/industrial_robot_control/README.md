# robodsl_package

Generated ROS2 package from RoboDSL specification

## Overview

This package contains 3 ROS2 nodes and 5 CUDA kernels.

## Package Structure

```
robodsl_package/
├── CMakeLists.txt
├── package.xml
├── README.md
├── src/
│   └── robodsl_package/
│       ├── industrial_robot_controller.cpp
│       ├── industrial_robot_controller.hpp
│       ├── safety_monitor.cpp
│       ├── safety_monitor.hpp
│       ├── robot_coordinator.cpp
│       ├── robot_coordinator.hpp
│       ├── trajectory_interpolation.cu
│       ├── trajectory_interpolation.cuh
│       ├── inverse_kinematics.cu
│       ├── inverse_kinematics.cuh
│       ├── collision_detection.cu
│       ├── collision_detection.cuh
│       ├── force_control.cu
│       ├── force_control.cuh
│       ├── motion_prediction.cu
│       └── motion_prediction.cuh
├── launch/
│   ├── robodsl_package_launch.py
│   ├── industrial_robot_controller_launch.py
│   ├── safety_monitor_launch.py
│   └── robot_coordinator_launch.py
├── include/
│   └── robodsl_package/
│       ├── industrial_robot_controller.hpp
│       ├── safety_monitor.hpp
│       └── robot_coordinator.hpp
├── config/
│   └── params.yaml
└── test/
    └── test_robodsl_package.py
```

## Nodes

### industrial_robot_controller

- **Type**: Lifecycle Node- **Publishers**: 6
- **Subscribers**: 5
- **Services**: 0
- **Actions**: 0
- **Timers**: 3
- **Parameters**: 14

### safety_monitor

- **Type**: Standard Node- **Publishers**: 2
- **Subscribers**: 4
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 4

### robot_coordinator

- **Type**: Standard Node- **Publishers**: 3
- **Subscribers**: 3
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 2


## Standalone CUDA Kernels

### trajectory_interpolation

- **Parameters**: 6
- **Block Size**: 256
- **Use Thrust**: False

### inverse_kinematics

- **Parameters**: 4
- **Block Size**: 256
- **Use Thrust**: False

### collision_detection

- **Parameters**: 5
- **Block Size**: 256
- **Use Thrust**: False

### force_control

- **Parameters**: 5
- **Block Size**: 256
- **Use Thrust**: False

### motion_prediction

- **Parameters**: 4
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
ros2 launch robodsl_package industrial_robot_controller_launch.py
ros2 launch robodsl_package safety_monitor_launch.py
ros2 launch robodsl_package robot_coordinator_launch.py
```

## License

Apache-2.0 