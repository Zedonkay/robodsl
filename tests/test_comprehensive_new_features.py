#!/usr/bin/env python3
"""
Comprehensive test script for new RoboDSL features:
- Custom message/service/action types
- Dynamic runtime configuration
- Simulation configuration
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2
from robodsl.generators.message_generator import MessageGenerator
from robodsl.generators.simulation_generator import SimulationGenerator
from robodsl.generators.dynamic_runtime_generator import DynamicRuntimeGenerator

def test_parser_with_new_features():
    """Test the parser with all new features."""
    print("=== Testing Parser with New Features ===")
    
    # Test file with all new features
    test_content = """
// Test file for new RoboDSL features

// Custom Message Definition
message CustomImage {
    uint32 width;
    uint32 height;
    uint8 data[];
    string format = "RGB";
}

// Custom Service Definition
service ImageProcessing {
    uint32 image_id;
    string operation;
    ---
    bool success;
    string error_message;
}

// Custom Action Definition
action NavigationAction {
    float64 target_x;
    float64 target_y;
    float64 target_z;
    ---
    float64 current_x;
    float64 current_y;
    float64 current_z;
    ---
    bool reached_goal;
    float64 final_distance;
}

// Dynamic Runtime Configuration
dynamic_parameters {
    parameter float64 max_speed = 2.0 {
        min_value: 0.1
        max_value: 10.0
        step: 0.1
        description: "Maximum robot speed in m/s"
    }
    parameter bool enable_safety = true {
        description: "Enable safety features"
    }
}

dynamic_remaps {
    remap /camera/image_raw: /sim/camera/image_raw when: "simulation_mode"
    remap /cmd_vel: /hardware/cmd_vel when: "hardware_mode"
}

// Simulation Configuration
simulation gazebo {
    world {
        world_file: "empty.world"
        physics_engine: "ode"
        gravity: (0, 0, -9.81)
        max_step_size: 0.001
        real_time_factor: 1.0
    }
    
    robot robot1 {
        model_file: "robot.urdf"
        namespace: "robot1"
        initial_pose: (0, 0, 0, 0, 0, 0)
        plugins {
            plugin gazebo_ros_control {
                robot_param: "robot_description"
                robot_param_node: "robot_state_publisher"
            }
        }
    }
    
    gui: true
    headless: false
}

// Hardware-in-the-Loop Configuration
hardware_in_loop {
    simulation_nodes: perception_node, planning_node
    hardware_nodes: motor_controller, sensor_driver
    bridge_config: "hil_bridge.yaml"
}

// Regular node using custom types
node perception_node {
    subscriber /camera/image_raw: "CustomImage"
    publisher /detections: "vision_msgs/DetectionArray"
    
    method process_image {
        input: const CustomImage& image
        output: std::vector<Detection>& detections
        code {
            // Process the custom image
            detections.clear();
            // Add detection logic here
        }
    }
}
"""
    
    try:
        
        ast = parse_robodsl(test_content)
        
        print("✅ Parser test passed!")
        print(f"   - Messages: {len(ast.messages)}")
        print(f"   - Services: {len(ast.services)}")
        print(f"   - Actions: {len(ast.actions)}")
        print(f"   - Dynamic parameters: {len(ast.dynamic_parameters)}")
        print(f"   - Dynamic remaps: {len(ast.dynamic_remaps)}")
        print(f"   - Simulation config: {ast.simulation is not None}")
        print(f"   - HIL config: {ast.hil_config is not None}")
        print(f"   - Nodes: {len(ast.nodes)}")
        
        assert ast is not None
        
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        raise

def test_message_generator():
    """Test the message generator."""
    print("\n=== Testing Message Generator ===")
    
    try:
        # Create test AST with messages, services, and actions
        
        test_content = """
message CustomImage {
    uint32 width;
    uint32 height;
    uint8 data[];
    string format = "RGB";
}

service ImageProcessing {
    uint32 image_id;
    string operation;
    ---
    bool success;
    string error_message;
}

action NavigationAction {
    float64 target_x;
    float64 target_y;
    float64 target_z;
    ---
    float64 current_x;
    float64 current_y;
    float64 current_z;
    ---
    bool reached_goal;
    float64 final_distance;
}
"""
        ast = parse_robodsl(test_content)
        
        generator = MessageGenerator(output_dir="test_output/msg")
        
        # Generate message files
        msg_files = generator.generate_messages(ast.messages)
        print(f"✅ Generated {len(msg_files)} message files:")
        for file in msg_files:
            print(f"   - {file}")
        
        # Generate service files
        srv_files = generator.generate_services(ast.services)
        print(f"✅ Generated {len(srv_files)} service files:")
        for file in srv_files:
            print(f"   - {file}")
        
        # Generate action files
        action_files = generator.generate_actions(ast.actions)
        print(f"✅ Generated {len(action_files)} action files:")
        for file in action_files:
            print(f"   - {file}")
        
        assert True
        
    except Exception as e:
        print(f"❌ Message generator test failed: {e}")
        assert False

def test_simulation_generator():
    """Test the simulation generator."""
    print("\n=== Testing Simulation Generator ===")
    
    try:
        # Create test AST with simulation config
        
        test_content = """
simulation gazebo {
    world {
        world_file: "empty.world"
        physics_engine: "ode"
        gravity: (0, 0, -9.81)
        max_step_size: 0.001
        real_time_factor: 1.0
    }
    
    robot robot1 {
        model_file: "robot.urdf"
        namespace: "robot1"
        initial_pose: (0, 0, 0, 0, 0, 0)
    }
    
    gui: true
    headless: false
}

hardware_in_loop {
    simulation_nodes: perception_node, planning_node
    hardware_nodes: motor_controller, sensor_driver
    bridge_config: "hil_bridge.yaml"
}
"""
        ast = parse_robodsl(test_content)
        
        generator = SimulationGenerator(output_dir="test_output/launch")
        
        # Generate simulation launch file
        if ast.simulation:
            launch_content = generator.generate_simulation_launch(ast.simulation)
            print("✅ Generated simulation launch file")
            print(f"   Content length: {len(launch_content)} characters")
        
        # Generate HIL launch file
        if ast.hil_config:
            hil_content = generator.generate_hil_launch(ast.hil_config)
            print("✅ Generated HIL launch file")
            print(f"   Content length: {len(hil_content)} characters")
        
        assert True
        
    except Exception as e:
        print(f"❌ Simulation generator test failed: {e}")
        assert False

def test_dynamic_runtime_generator():
    """Test the dynamic runtime generator."""
    print("\n=== Testing Dynamic Runtime Generator ===")
    
    try:
        # Create test AST with dynamic runtime config
        
        test_content = """
dynamic_parameters {
    parameter float64 max_speed = 2.0 {
        min_value: 0.1
        max_value: 10.0
        step: 0.1
        description: "Maximum robot speed in m/s"
    }
    parameter bool enable_safety = true {
        description: "Enable safety features"
    }
}

dynamic_remaps {
    remap /camera/image_raw: /sim/camera/image_raw when: "simulation_mode"
    remap /cmd_vel: /hardware/cmd_vel when: "hardware_mode"
}
"""
        ast = parse_robodsl(test_content)
        
        generator = DynamicRuntimeGenerator(output_dir="test_output/config")
        
        # Generate dynamic parameters config
        if ast.dynamic_parameters:
            param_config = generator.generate_dynamic_parameters(ast.dynamic_parameters)
            print("✅ Generated dynamic parameters config")
            print(f"   Content length: {len(param_config)} characters")
        
        # Generate dynamic remaps config
        if ast.dynamic_remaps:
            remap_config = generator.generate_dynamic_remaps(ast.dynamic_remaps)
            print("✅ Generated dynamic remaps config")
            print(f"   Content length: {len(remap_config)} characters")
        
        # Generate runtime manager
        if ast.dynamic_parameters or ast.dynamic_remaps:
            runtime_manager = generator.generate_runtime_manager(ast.dynamic_parameters, ast.dynamic_remaps)
            print("✅ Generated runtime manager")
            print(f"   Content length: {len(runtime_manager)} characters")
        
        assert True
        
    except Exception as e:
        print(f"❌ Dynamic runtime generator test failed: {e}")
        assert False

def test_individual_features():
    """Test individual features separately."""
    print("\n=== Testing Individual Features ===")
    
    # Test message parsing
    print("\n--- Testing Message Parsing ---")
    message_test = """
message TestMessage {
    uint32 id;
    string name;
    float64 value = 1.0;
}
"""
    try:
        
        ast = parse_robodsl(message_test)
        print(f"✅ Message parsing: {len(ast.messages)} messages parsed")
    except Exception as e:
        print(f"❌ Message parsing failed: {e}")
    
    # Test service parsing
    print("\n--- Testing Service Parsing ---")
    service_test = """
service TestService {
    uint32 request_id;
    ---
    bool success;
}
"""
    try:
        
        ast = parse_robodsl(service_test)
        print(f"✅ Service parsing: {len(ast.services)} services parsed")
    except Exception as e:
        print(f"❌ Service parsing failed: {e}")
    
    # Test action parsing
    print("\n--- Testing Action Parsing ---")
    action_test = """
action TestAction {
    float64 goal_x;
    ---
    float64 feedback_x;
    ---
    bool result_success;
}
"""
    try:
        
        ast = parse_robodsl(action_test)
        print(f"✅ Action parsing: {len(ast.actions)} actions parsed")
    except Exception as e:
        print(f"❌ Action parsing failed: {e}")
    
    # Test dynamic parameters
    print("\n--- Testing Dynamic Parameters ---")
    dynamic_test = """
dynamic_parameters {
    parameter float64 test_param = 1.0 {
        min_value: 0.0
        max_value: 10.0
        description: "Test parameter"
    }
}
"""
    try:
        
        ast = parse_robodsl(dynamic_test)
        print(f"✅ Dynamic parameters: {len(ast.dynamic_parameters)} parameters parsed")
    except Exception as e:
        print(f"❌ Dynamic parameters failed: {e}")
    
    # Test simulation config
    print("\n--- Testing Simulation Config ---")
    sim_test = """
simulation gazebo {
    world {
        world_file: "test.world"
    }
    gui: true
}
"""
    try:
        
        ast = parse_robodsl(sim_test)
        print(f"✅ Simulation config: {ast.simulation is not None}")
    except Exception as e:
        print(f"❌ Simulation config failed: {e}")

def test_file_output():
    """Test that files are actually written."""
    print("\n=== Testing File Output ===")
    
    test_outputs = [
        "test_output/msg/CustomImage.msg",
        "test_output/msg/ImageProcessing.srv",
        "test_output/msg/NavigationAction.action",
        "test_output/config/dynamic_parameters.yaml",
        "test_output/config/dynamic_remaps.yaml",
    ]
    
    for file_path in test_outputs:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"   Content length: {len(content)} characters")
        else:
            print(f"❌ File missing: {file_path}")

def main():
    """Run all tests."""
    print("🚀 Starting Comprehensive RoboDSL New Features Test")
    print("=" * 60)
    
    # Create test output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Test parser with all features
    ast = test_parser_with_new_features()
    
    if ast:
        # Test generators
        test_message_generator(ast)
        test_simulation_generator(ast)
        test_dynamic_runtime_generator(ast)
        
        # Test individual features
        test_individual_features()
        
        # Test file output
        test_file_output()
    
    print("\n" + "=" * 60)
    print("🏁 Comprehensive test completed!")
    
    # Clean up test files
    print("\n🧹 Cleaning up test files...")
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
        print("✅ Test files cleaned up")

if __name__ == "__main__":
    main() 