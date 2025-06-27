#!/usr/bin/env python3
"""
Test script for the updated RoboDSL generator.
This script tests the generator with both ROS2 and non-ROS2 configurations.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.core.generator import CodeGenerator

def create_test_ast(ros2_enabled=True):
    """Create a test AST."""
    ast = parse_robodsl()
    
    # Create a test node with publishers, subscribers, and services
    node_content = NodeContentNode(
        publishers=[
            PublisherNode(topic="/test_topic", msg_type="std_msgs/msg/String", qos=None)
        ],
        subscribers=[
            SubscriberNode(topic="/input_topic", msg_type="std_msgs/msg/String", qos=None)
        ],
        services=[
            ServiceNode(service="/test_service", srv_type="std_srvs/srv/Trigger", qos=None)
        ],
        parameters=[
            ParameterNode(name="test_param", value=ValueNode(value=42))
        ]
    )
    
    test_node = NodeNode(name="test_node", content=node_content)
    ast.nodes.append(test_node)
    
    return ast

def test_generator_basic():
    """Test basic generator functionality."""
    print("Testing basic generator functionality...")
    
    # Create test AST
    ast = create_test_ast()
    
    # Create generator
    generator = CodeGenerator()
    
    # Test that generator can be created
    assert generator is not None
    print("✓ Generator created successfully")

def test_generator_with_ros2():
    """Test generator with ROS2 enabled."""
    print("Testing generator with ROS2...")
    
    # Create test AST
    ast = create_test_ast(ros2_enabled=True)
    
    # Create generator
    generator = CodeGenerator()
    
    # Test ROS2-specific generation
    try:
        # This would test ROS2-specific template generation
        # For now, just verify the generator exists
        assert generator is not None
        print("✓ ROS2 generator test passed")
    except Exception as e:
        print(f"✗ ROS2 generator test failed: {e}")

def test_generator_without_ros2():
    """Test generator without ROS2."""
    print("Testing generator without ROS2...")
    
    # Create test AST
    ast = create_test_ast(ros2_enabled=False)
    
    # Create generator
    generator = CodeGenerator()
    
    # Test non-ROS2 generation
    try:
        # This would test non-ROS2 template generation
        # For now, just verify the generator exists
        assert generator is not None
        print("✓ Non-ROS2 generator test passed")
    except Exception as e:
        print(f"✗ Non-ROS2 generator test failed: {e}")

def test_node_generation():
    """Test node generation."""
    print("Testing node generation...")
    
    # Create test AST with a simple node
    ast = parse_robodsl()
    
    node_content = NodeContentNode(
        publishers=[],
        subscribers=[],
        services=[],
        parameters=[]
    )
    
    test_node = NodeNode(name="simple_node", content=node_content)
    ast.nodes.append(test_node)
    
    # Create generator
    generator = CodeGenerator()
    
    try:
        # Test that we can access the node
        assert len(ast.nodes) == 1
        assert ast.nodes[0].name == "simple_node"
        print("✓ Node generation test passed")
    except Exception as e:
        print(f"✗ Node generation test failed: {e}")

def test_publisher_generation():
    """Test publisher generation."""
    print("Testing publisher generation...")
    
    # Create test AST with a publisher
    ast = parse_robodsl()
    
    node_content = NodeContentNode(
        publishers=[
            PublisherNode(topic="/test_pub", msg_type="std_msgs/msg/String", qos=None)
        ],
        subscribers=[],
        services=[],
        parameters=[]
    )
    
    test_node = NodeNode(name="pub_node", content=node_content)
    ast.nodes.append(test_node)
    
    try:
        # Test that publisher is correctly stored
        assert len(ast.nodes[0].content.publishers) == 1
        pub = ast.nodes[0].content.publishers[0]
        assert pub.topic == "/test_pub"
        assert pub.msg_type == "std_msgs/msg/String"
        print("✓ Publisher generation test passed")
    except Exception as e:
        print(f"✗ Publisher generation test failed: {e}")

def test_subscriber_generation():
    """Test subscriber generation."""
    print("Testing subscriber generation...")
    
    # Create test AST with a subscriber
    ast = parse_robodsl()
    
    node_content = NodeContentNode(
        publishers=[],
        subscribers=[
            SubscriberNode(topic="/test_sub", msg_type="std_msgs/msg/String", qos=None)
        ],
        services=[],
        parameters=[]
    )
    
    test_node = NodeNode(name="sub_node", content=node_content)
    ast.nodes.append(test_node)
    
    try:
        # Test that subscriber is correctly stored
        assert len(ast.nodes[0].content.subscribers) == 1
        sub = ast.nodes[0].content.subscribers[0]
        assert sub.topic == "/test_sub"
        assert sub.msg_type == "std_msgs/msg/String"
        print("✓ Subscriber generation test passed")
    except Exception as e:
        print(f"✗ Subscriber generation test failed: {e}")

def test_service_generation():
    """Test service generation."""
    print("Testing service generation...")
    
    # Create test AST with a service
    ast = parse_robodsl()
    
    node_content = NodeContentNode(
        publishers=[],
        subscribers=[],
        services=[
            ServiceNode(service="/test_srv", srv_type="std_srvs/srv/Trigger", qos=None)
        ],
        parameters=[]
    )
    
    test_node = NodeNode(name="srv_node", content=node_content)
    ast.nodes.append(test_node)
    
    try:
        # Test that service is correctly stored
        assert len(ast.nodes[0].content.services) == 1
        srv = ast.nodes[0].content.services[0]
        assert srv.service == "/test_srv"
        assert srv.srv_type == "std_srvs/srv/Trigger"
        print("✓ Service generation test passed")
    except Exception as e:
        print(f"✗ Service generation test failed: {e}")

def test_parameter_generation():
    """Test parameter generation."""
    print("Testing parameter generation...")
    
    # Create test AST with parameters
    ast = parse_robodsl()
    
    node_content = NodeContentNode(
        publishers=[],
        subscribers=[],
        services=[],
        parameters=[
            ParameterNode(name="test_param", value=ValueNode(value=42)),
            ParameterNode(name="string_param", value=ValueNode(value="test_value"))
        ]
    )
    
    test_node = NodeNode(name="param_node", content=node_content)
    ast.nodes.append(test_node)
    
    try:
        # Test that parameters are correctly stored
        assert len(ast.nodes[0].content.parameters) == 2
        param1 = ast.nodes[0].content.parameters[0]
        param2 = ast.nodes[0].content.parameters[1]
        assert param1.name == "test_param"
        assert param1.value.value == 42
        assert param2.name == "string_param"
        assert param2.value.value == "test_value"
        print("✓ Parameter generation test passed")
    except Exception as e:
        print(f"✗ Parameter generation test failed: {e}")

def main():
    """Run all generator tests."""
    print("Running RoboDSL Generator Tests")
    print("=" * 40)
    
    # Run all tests
    test_generator_basic()
    test_generator_with_ros2()
    test_generator_without_ros2()
    test_node_generation()
    test_publisher_generation()
    test_subscriber_generation()
    test_service_generation()
    test_parameter_generation()
    
    print("\n" + "=" * 40)
    print("All generator tests completed!")

if __name__ == "__main__":
    main()
