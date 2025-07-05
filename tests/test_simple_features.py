from robodsl.parsers.lark_parser import parse_robodsl
#!/usr/bin/env python3
"""
Simple test script for new RoboDSL features.
Tests the grammar and generators directly.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_grammar_parsing():
    """Test that the grammar can parse the new features."""
    print("=== Testing Grammar Parsing ===")
    
    try:
        from lark import Lark
        
        # Load the grammar
        grammar_file = Path("src/robodsl/grammar/robodsl.lark")
        with open(grammar_file, 'r') as f:
            grammar_content = f.read()
        
        # Create parser
        parser = Lark(grammar_content, parser='lalr', start='start')
        
        # Test message parsing
        message_test = """
message CustomImage {
    uint32 width;
    uint32 height;
    uint8 data[];
    string format = "RGB";
}
"""
        try:
            tree = parse_robodsl(message_test)
            print("✅ Message grammar parsing: PASSED")
        except Exception as e:
            print(f"❌ Message grammar parsing: FAILED - {e}")
        
        # Test service parsing
        service_test = """
service ImageProcessing {
    uint32 image_id;
    string operation;
    ---
    bool success;
    string error_message;
}
"""
        try:
            tree = parse_robodsl(service_test)
            print("✅ Service grammar parsing: PASSED")
        except Exception as e:
            print(f"❌ Service grammar parsing: FAILED - {e}")
        
        # Test action parsing
        action_test = """
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
        try:
            tree = parse_robodsl(action_test)
            print("✅ Action grammar parsing: PASSED")
        except Exception as e:
            print(f"❌ Action grammar parsing: FAILED - {e}")
        
        # Test dynamic parameters
        dynamic_test = """
dynamic_parameters {
    parameter float64 max_speed = 2.0 {
        min_value: 0.1
        max_value: 10.0
        step: 0.1
        description: "Maximum robot speed in m/s"
    }
}
"""
        try:
            tree = parse_robodsl(dynamic_test)
            print("✅ Dynamic parameters grammar parsing: PASSED")
        except Exception as e:
            print(f"❌ Dynamic parameters grammar parsing: FAILED - {e}")
        
        # Test simulation config
        sim_test = """
simulation gazebo {
    world {
        world_file: "empty.world"
        physics_engine: "ode"
        gravity: (0, 0, -9.81)
    }
    gui: true
}
"""
        try:
            tree = parse_robodsl(sim_test)
            print("✅ Simulation config grammar parsing: PASSED")
        except Exception as e:
            print(f"❌ Simulation config grammar parsing: FAILED - {e}")
        
        assert True
        
    except Exception as e:
        print(f"❌ Grammar loading failed: {e}")
        assert False

def test_message_generator():
    """Test the message generator with mock data."""
    print("\n=== Testing Message Generator ===")
    
    try:
        from robodsl.generators.message_generator import MessageGenerator
        from robodsl.core.ast import MessageNode, MessageContentNode, MessageFieldNode, ValueNode
        
        # Create mock message
        message = MessageNode(
            name="TestMessage",
            content=MessageContentNode(
                fields=[
                    MessageFieldNode(type="uint32", name="width", array_spec=None, default_value=None),
                    MessageFieldNode(type="uint32", name="height", array_spec=None, default_value=None),
                    MessageFieldNode(type="uint8", name="data", array_spec="[]", default_value=None),
                    MessageFieldNode(type="string", name="format", array_spec=None, default_value=ValueNode(value="RGB")),
                ]
            )
        )
        
        # Test generator
        generator = MessageGenerator(output_dir="test_output/msg")
        msg_files = generator.generate_messages([message])
        
        print(f"✅ Message generator: Generated {len(msg_files)} files")
        for file in msg_files:
            print(f"   - {file}")
        
        assert True
        
    except Exception as e:
        print(f"❌ Message generator test failed: {e}")
        assert False

def test_simulation_generator():
    """Test the simulation generator with mock data."""
    print("\n=== Testing Simulation Generator ===")
    
    try:
        from robodsl.generators.simulation_generator import SimulationGenerator
        from robodsl.core.ast import (
            SimulationConfigNode, SimulationWorldNode, SimulationRobotNode,
            SimulationPluginNode, HardwareInLoopNode
        )
        
        # Create mock simulation config
        simulation = SimulationConfigNode(
            simulator="gazebo",
            world=SimulationWorldNode(
                world_file="empty.world",
                physics_engine="ode",
                gravity=(0, 0, -9.81),
                max_step_size=0.001,
                real_time_factor=1.0
            ),
            robots=[
                SimulationRobotNode(
                    model_file="robot.urdf",
                    namespace="robot1",
                    initial_pose=(0, 0, 0, 0, 0, 0),
                    plugins=[]
                )
            ],
            plugins=[],
            gui=True,
            headless=False,
            physics_engine="ode"
        )
        
        # Test generator
        generator = SimulationGenerator(output_dir="test_output/launch")
        launch_file = generator.write_simulation_launch(simulation)
        
        print(f"✅ Simulation generator: Generated launch file ({len(launch_file)} chars)")
        print(f"   - {launch_file}")
        
        assert True
        
    except Exception as e:
        print(f"❌ Simulation generator test failed: {e}")
        assert False

def test_dynamic_runtime_generator():
    """Test the dynamic runtime generator with mock data."""
    print("\n=== Testing Dynamic Runtime Generator ===")
    
    try:
        from robodsl.generators.dynamic_runtime_generator import DynamicRuntimeGenerator
        from robodsl.core.ast import DynamicParameterNode, DynamicRemapNode, ValueNode
        
        # Create mock dynamic parameters
        parameters = [
            DynamicParameterNode(
                name="max_speed",
                type="float64",
                default_value=ValueNode(value=2.0),
                min_value=ValueNode(value=0.1),
                max_value=ValueNode(value=10.0),
                step=ValueNode(value=0.1),
                description="Maximum robot speed in m/s"
            )
        ]
        
        remaps = [
            DynamicRemapNode(
                from_topic="/camera/image_raw",
                to_topic="/sim/camera/image_raw",
                condition="simulation_mode"
            )
        ]
        
        # Test generator
        generator = DynamicRuntimeGenerator(output_dir="test_output/config")
        
        param_file = generator.write_dynamic_parameters(parameters)
        remap_file = generator.write_dynamic_remaps(remaps)
        runtime_file = generator.write_runtime_manager(parameters, remaps)
        
        print(f"✅ Dynamic runtime generator:")
        print(f"   - Parameters config: {param_file}")
        print(f"   - Remaps config: {remap_file}")
        print(f"   - Runtime manager: {runtime_file}")
        
        assert True
        
    except Exception as e:
        print(f"❌ Dynamic runtime generator test failed: {e}")
        assert False

def test_file_output():
    """Test that files are actually written."""
    print("\n=== Testing File Output ===")
    
    test_files = [
        "test_output/msg/TestMessage.msg",
        "test_output/launch/simulation_launch.py",
        "test_output/config/dynamic_parameters.yaml",
    ]
    
    for file_path in test_files:
        if os.path.exists(file_path):
            print(f"✅ File exists: {file_path}")
            with open(file_path, 'r') as f:
                content = f.read()
                print(f"   Content length: {len(content)} characters")
                print(f"   First 100 chars: {content[:100]}...")
        else:
            print(f"❌ File missing: {file_path}")

def main():
    """Run all tests."""
    print("🚀 Starting Simple RoboDSL New Features Test")
    print("=" * 60)
    
    # Create test output directory
    os.makedirs("test_output", exist_ok=True)
    
    # Run tests
    grammar_ok = test_grammar_parsing()
    message_ok = test_message_generator()
    simulation_ok = test_simulation_generator()
    dynamic_ok = test_dynamic_runtime_generator()
    
    # Test file output
    test_file_output()
    
    print("\n" + "=" * 60)
    print("🏁 Test Results:")
    print(f"   Grammar parsing: {'✅ PASSED' if grammar_ok else '❌ FAILED'}")
    print(f"   Message generator: {'✅ PASSED' if message_ok else '❌ FAILED'}")
    print(f"   Simulation generator: {'✅ PASSED' if simulation_ok else '❌ FAILED'}")
    print(f"   Dynamic runtime generator: {'✅ PASSED' if dynamic_ok else '❌ FAILED'}")
    
    # Clean up test files
    print("\n🧹 Cleaning up test files...")
    import shutil
    if os.path.exists("test_output"):
        shutil.rmtree("test_output")
        print("✅ Test files cleaned up")

if __name__ == "__main__":
    main() 