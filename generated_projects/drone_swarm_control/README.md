# drone_swarm_control

A RoboDSL generated ROS2 package.

## Building

```bash
# Build the package
colcon build --packages-select drone_swarm_control

# Source the workspace
source install/setup.bash
```

## Running

```bash
# Run the main node
ros2 run drone_swarm_control drone_swarm_control_node

# Or use the launch file
ros2 launch drone_swarm_control main_node.launch.py
```

## Development

1. Edit the RoboDSL configuration in `drone_swarm_control.robodsl`
2. Regenerate the C++ code: `robodsl generate drone_swarm_control.robodsl`
3. Build and test your changes

## Project Structure

- `drone_swarm_control.robodsl` - Main RoboDSL configuration
- `src/` - Generated C++ source files
- `include/` - Generated C++ header files
- `launch/` - Launch files
- `config/` - Configuration files
- `robodsl/` - Additional RoboDSL node definitions

## Data Structures

RoboDSL supports defining custom data structures:
- **Structs**: Simple data containers
- **Classes**: Object-oriented data structures with methods
- **Enums**: Enumerated types
- **Typedefs**: Type aliases
- **Using declarations**: Modern C++ type aliases

Example:
```robodsl
struct SensorData {
    double timestamp;
    std::vector<float> values;
    bool valid;
};

enum class SensorType {
    CAMERA,
    LIDAR,
    IMU
};

typedef std::vector<SensorData> SensorDataArray;
```
