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
│       ├── autonomous_vehicle_controller.cpp
│       ├── autonomous_vehicle_controller.hpp
│       ├── safety_monitor.cpp
│       ├── safety_monitor.hpp
│       ├── sensor_fusion.cu
│       ├── sensor_fusion.cuh
│       ├── trajectory_planning.cu
│       ├── trajectory_planning.cuh
│       ├── safety_monitoring.cu
│       └── safety_monitoring.cuh
├── launch/
│   ├── robodsl_package_launch.py
│   ├── autonomous_vehicle_controller_launch.py
│   └── safety_monitor_launch.py
├── include/
│   └── robodsl_package/
│       ├── autonomous_vehicle_controller.hpp
│       └── safety_monitor.hpp
├── config/
│   └── params.yaml
└── test/
    └── test_robodsl_package.py
```

## Nodes

### autonomous_vehicle_controller

- **Type**: Lifecycle Node- **Publishers**: 6
- **Subscribers**: 4
- **Services**: 0
- **Actions**: 0
- **Timers**: 3
- **Parameters**: 10

### safety_monitor

- **Type**: Standard Node- **Publishers**: 2
- **Subscribers**: 2
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 2


## Standalone CUDA Kernels

### sensor_fusion

- **Parameters**: 6
- **Block Size**: 256
- **Use Thrust**: False

### trajectory_planning

- **Parameters**: 6
- **Block Size**: 256
- **Use Thrust**: False

### safety_monitoring

- **Parameters**: 6
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
ros2 launch robodsl_package autonomous_vehicle_controller_launch.py
ros2 launch robodsl_package safety_monitor_launch.py
```

## License

Apache-2.0 