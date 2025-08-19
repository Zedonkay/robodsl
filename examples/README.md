# RoboDSL Examples

This directory contains 5 comprehensive robotics examples that showcase the full capabilities of RoboDSL, including ROS2 integration, CUDA acceleration, ONNX model support, and advanced pipeline orchestration.

## Overview

Each example demonstrates different aspects of modern robotics systems:

1. **Autonomous Navigation System** - Complete SLAM and navigation with multi-sensor fusion
2. **Computer Vision Processing** - Real-time video processing with multiple ML models
3. **Industrial Robot Control** - Advanced manufacturing robotics with safety systems
4. **Drone Swarm Control** - Multi-agent coordination and swarm intelligence
5. **Autonomous Vehicle System** - Self-driving car capabilities with sensor fusion

## Example 1: Autonomous Navigation System (`01_autonomous_navigation_system.robodsl`)

### Features Demonstrated
- **CUDA-accelerated SLAM**: Point cloud processing and occupancy grid mapping
- **Multi-stage pipeline**: Point cloud filtering → SLAM mapping → Object detection → Path planning → Motion control
- **ONNX models**: YOLO object detection and semantic segmentation
- **ROS2 integration**: Complete navigation stack with QoS configuration
- **Real-time processing**: 20Hz SLAM updates, 10Hz path planning

### Key Components
- `point_cloud_filter` kernel: GPU-accelerated point cloud filtering
- `occupancy_grid_update` kernel: Real-time map building
- `path_planning_astar` kernel: GPU-accelerated A* path planning
- YOLO detector: Real-time object detection
- Semantic segmentation: Environment understanding
- Multi-stage pipeline orchestration

### Usage
```bash
# Generate code
robodsl generate 01_autonomous_navigation_system.robodsl

# Build and run
colcon build
source install/setup.bash
ros2 launch robodsl_package autonomous_navigation_launch.py
```

## Example 2: Computer Vision Processing (`02_computer_vision_processing.robodsl`)

### Features Demonstrated
- **Multi-camera processing**: Synchronized multi-camera vision system
- **Advanced image processing**: CUDA-accelerated filtering, resizing, and enhancement
- **Multiple ONNX models**: Object detection, face recognition, pose estimation, depth estimation, optical flow
- **Real-time pipeline**: 30Hz processing with GPU optimization
- **Feature matching**: CUDA-accelerated computer vision algorithms

### Key Components
- `image_preprocessing` kernel: GPU-accelerated image normalization
- `image_resize` kernel: Bilinear interpolation for image scaling
- `gaussian_blur` kernel: 5x5 Gaussian filtering
- `edge_detection` kernel: Sobel edge detection
- `feature_matching` kernel: SIFT-like feature matching
- Multiple ONNX models for different vision tasks
- Multi-camera synchronization

### Usage
```bash
# Generate code
robodsl generate 02_computer_vision_processing.robodsl

# Build and run
colcon build
source install/setup.bash
ros2 launch robodsl_package computer_vision_launch.py
```

## Example 3: Industrial Robot Control (`03_industrial_robot_control.robodsl`)

### Features Demonstrated
- **Real-time control**: 1000Hz control loop with CUDA acceleration
- **Advanced trajectory planning**: Cubic spline interpolation and optimization
- **Force control**: Impedance control with CUDA-accelerated computation
- **Safety monitoring**: Collision detection and emergency systems
- **Multi-robot coordination**: Synchronized multi-robot operations

### Key Components
- `trajectory_interpolation` kernel: Smooth trajectory generation
- `inverse_kinematics` kernel: Real-time IK solving
- `collision_detection` kernel: GPU-accelerated collision checking
- `force_control` kernel: Impedance control implementation
- `motion_prediction` kernel: ML-based motion prediction
- ONNX models for trajectory optimization and collision prediction
- Safety monitoring system

### Usage
```bash
# Generate code
robodsl generate 03_industrial_robot_control.robodsl

# Build and run
colcon build
source install/setup.bash
ros2 launch robodsl_package industrial_robot_launch.py
```

## Example 4: Drone Swarm Control (`04_drone_swarm_control.robodsl`)

### Features Demonstrated
- **Swarm intelligence**: Particle Swarm Optimization for swarm behavior
- **Formation control**: Multi-agent coordination and formation flying
- **Collision avoidance**: Real-time collision detection and avoidance
- **Communication optimization**: Network-aware position optimization
- **Distributed computing**: GPU-accelerated swarm algorithms

### Key Components
- `formation_control` kernel: Multi-drone formation control
- `collision_avoidance` kernel: Repulsive force-based avoidance
- `swarm_optimization` kernel: PSO-based swarm intelligence
- `path_planning` kernel: A* path planning for each drone
- `communication_optimization` kernel: Network-aware positioning
- ONNX models for behavior prediction and formation optimization
- Multi-drone coordination system

### Usage
```bash
# Generate code
robodsl generate 04_drone_swarm_control.robodsl

# Build and run
colcon build
source install/setup.bash
ros2 launch robodsl_package drone_swarm_launch.py
```

## Example 5: Autonomous Vehicle System (`05_autonomous_vehicle_system.robodsl`)

### Features Demonstrated
- **Sensor fusion**: Multi-sensor data fusion with CUDA acceleration
- **Advanced perception**: Traffic sign detection, lane detection, behavior prediction
- **Safety systems**: Real-time safety monitoring and emergency handling
- **Trajectory planning**: Hybrid A* path planning for autonomous driving
- **ML-based prediction**: Behavior prediction and traffic analysis

### Key Components
- `sensor_fusion` kernel: Lidar-camera-radar fusion
- `trajectory_planning` kernel: Hybrid A* path planning
- `safety_monitoring` kernel: Real-time safety assessment
- ONNX models for traffic sign detection, lane detection, and behavior prediction
- Complete autonomous driving pipeline

### Usage
```bash
# Generate code
robodsl generate 05_autonomous_vehicle_system.robodsl

# Build and run
colcon build
source install/setup.bash
ros2 launch robodsl_package autonomous_vehicle_launch.py
```

## Common Features Across All Examples

### CUDA Acceleration
All examples utilize CUDA kernels for computationally intensive tasks:
- **Parallel processing**: GPU-accelerated algorithms
- **Memory optimization**: Shared memory and memory pooling
- **Stream processing**: Multi-stream execution for overlapping operations
- **Advanced features**: Dynamic parallelism, mixed precision, memory hierarchy

### ONNX Model Integration
Machine learning models are seamlessly integrated:
- **TensorRT optimization**: GPU-accelerated inference
- **Dynamic batching**: Efficient batch processing
- **Memory optimization**: Optimized memory usage
- **Profiling**: Performance monitoring and optimization

### ROS2 Integration
Complete ROS2 ecosystem integration:
- **QoS configuration**: Reliable and real-time communication
- **Lifecycle management**: Automatic startup and shutdown
- **Parameter management**: Runtime parameter configuration
- **Multi-node systems**: Distributed robotics applications

### Pipeline Orchestration
Multi-stage processing pipelines:
- **Stage coordination**: Synchronized multi-stage processing
- **Data flow**: Automatic data routing between stages
- **Topic management**: ROS2 topic integration
- **Error handling**: Robust error recovery

## Prerequisites

### Hardware Requirements
- **NVIDIA GPU**: CUDA-compatible GPU (RTX 3060 or better recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space for models and generated code

### Software Requirements
- **CUDA Toolkit**: Version 11.8 or later
- **ROS2**: Humble or Iron
- **Python**: 3.8 or later
- **RoboDSL**: Latest version with all dependencies

### ONNX Models
Each example requires specific ONNX models:
- YOLO models for object detection
- Semantic segmentation models
- Behavior prediction models
- Traffic sign detection models

## Building and Running

### 1. Environment Setup
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Activate RoboDSL virtual environment
source .venv/bin/activate
```

### 2. Generate Code
```bash
# Generate code for a specific example
robodsl generate examples/01_autonomous_navigation_system.robodsl

# Or generate all examples
for file in examples/*.robodsl; do
    robodsl generate "$file"
done
```

### 3. Build and Run
```bash
# Build the generated code
colcon build

# Source the workspace
source install/setup.bash

# Run a specific example
ros2 launch robodsl_package <example_name>_launch.py
```

## Customization

### Modifying Examples
Each example can be customized by:
1. **Adjusting parameters**: Modify parameter values in the `.robodsl` files
2. **Adding new kernels**: Extend CUDA kernels for additional functionality
3. **Integrating new models**: Add ONNX models for new ML capabilities
4. **Extending pipelines**: Add new processing stages to pipelines

### Performance Tuning
- **CUDA optimization**: Adjust block sizes, grid sizes, and shared memory
- **ONNX optimization**: Configure TensorRT optimization levels
- **ROS2 tuning**: Adjust QoS settings and frequencies
- **Memory management**: Optimize memory allocation and transfer

## Troubleshooting

### Common Issues
1. **CUDA errors**: Check GPU compatibility and CUDA installation
2. **ONNX model loading**: Verify model paths and TensorRT compatibility
3. **ROS2 communication**: Check QoS settings and network configuration
4. **Memory issues**: Reduce batch sizes or model complexity

### Debugging
- Enable verbose logging in RoboDSL
- Use ROS2 diagnostic tools
- Monitor GPU utilization with `nvidia-smi`
- Check ONNX model compatibility with TensorRT

## Contributing

To add new examples:
1. Create a new `.robodsl` file in the examples directory
2. Follow the established patterns and conventions
3. Include comprehensive documentation
4. Test with different hardware configurations
5. Update this README with new example details

## License

These examples are provided under the same license as RoboDSL (MIT License).
