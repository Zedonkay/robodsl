#!/usr/bin/env python3
"""
Test script for the updated RoboDSL generator.
This script tests the generator with both ROS2 and non-ROS2 configurations.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from robodsl.parser import RoboDSLConfig, NodeConfig, CudaKernelConfig
from robodsl.generator_fixed import CodeGenerator

def create_test_config(ros2_enabled=True):
    """Create a test configuration."""
    config = RoboDSLConfig()
    
    # Create a test node with publishers, subscribers, and services
    node = NodeConfig(
        name="test_node",
        publishers=[
            {"topic": "/test_topic", "msg_type": "std_msgs.msg.Float32MultiArray"},
            {"topic": "/test_topic2", "msg_type": "std_msgs.msg.String"}
        ],
        subscribers=[
            {"topic": "/test_sub", "msg_type": "sensor_msgs.msg.Image"}
        ],
        services=[
            {"service": "/test_service", "srv_type": "example_interfaces.srv.AddTwoInts"}
        ],
        parameters={
            "param1": "int",
            "param2": "string"
        }
    )
    
    config.nodes = [node]
    config.cuda_kernels = []
    
    return config

def test_generator():
    """Test the generator with both ROS2 and non-ROS2 configurations."""
    # Test with ROS2 enabled
    print("Testing with ROS2 enabled...")
    config = create_test_config(ros2_enabled=True)
    generator = CodeGenerator(config, output_dir="test_output/ros2")
    generator._generate_node_header(config.nodes[0])
    
    # Test with ROS2 disabled
    print("\nTesting with ROS2 disabled...")
    config = create_test_config(ros2_enabled=False)
    generator = CodeGenerator(config, output_dir="test_output/no_ros2")
    generator._generate_node_header(config.nodes[0])
    
    print("\nTest complete. Check the generated files in test_output/")

if __name__ == "__main__":
    test_generator()
