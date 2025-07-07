# robodsl_package

Generated ROS2 package from RoboDSL specification

## Overview

This package contains 0 ROS2 nodes.

## Package Structure

```
robodsl_package/
├── CMakeLists.txt
├── package.xml
├── README.md
├── src/
│   └── robodsl_package/
├── launch/
│   └── robodsl_package_launch.py
├── include/
│   └── robodsl_package/
├── config/
│   └── params.yaml
└── test/
    └── test_robodsl_package.py
```

## Nodes



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
```

## License

Apache-2.0 