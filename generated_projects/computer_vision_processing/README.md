# robodsl_package

Generated ROS2 package from RoboDSL specification

## Overview

This package contains 2 ROS2 nodes and 5 CUDA kernels.

## Package Structure

```
robodsl_package/
├── CMakeLists.txt
├── package.xml
├── README.md
├── src/
│   └── robodsl_package/
│       ├── vision_processor.cpp
│       ├── vision_processor.hpp
│       ├── camera_synchronizer.cpp
│       ├── camera_synchronizer.hpp
│       ├── image_preprocessing.cu
│       ├── image_preprocessing.cuh
│       ├── image_resize.cu
│       ├── image_resize.cuh
│       ├── gaussian_blur.cu
│       ├── gaussian_blur.cuh
│       ├── edge_detection.cu
│       ├── edge_detection.cuh
│       ├── feature_matching.cu
│       └── feature_matching.cuh
├── launch/
│   ├── robodsl_package_launch.py
│   ├── vision_processor_launch.py
│   └── camera_synchronizer_launch.py
├── include/
│   └── robodsl_package/
│       ├── vision_processor.hpp
│       └── camera_synchronizer.hpp
├── config/
│   └── params.yaml
└── test/
    └── test_robodsl_package.py
```

## Nodes

### vision_processor

- **Type**: Lifecycle Node- **Publishers**: 8
- **Subscribers**: 4
- **Services**: 0
- **Actions**: 0
- **Timers**: 1
- **Parameters**: 18

### camera_synchronizer

- **Type**: Standard Node- **Publishers**: 3
- **Subscribers**: 3
- **Services**: 0
- **Actions**: 0
- **Timers**: 0
- **Parameters**: 2


## Standalone CUDA Kernels

### image_preprocessing

- **Parameters**: 5
- **Block Size**: 16
- **Use Thrust**: False

### image_resize

- **Parameters**: 7
- **Block Size**: 16
- **Use Thrust**: False

### gaussian_blur

- **Parameters**: 5
- **Block Size**: 16
- **Use Thrust**: False

### edge_detection

- **Parameters**: 4
- **Block Size**: 16
- **Use Thrust**: False

### feature_matching

- **Parameters**: 9
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
ros2 launch robodsl_package vision_processor_launch.py
ros2 launch robodsl_package camera_synchronizer_launch.py
```

## License

Apache-2.0 