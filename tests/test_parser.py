"""Tests for the RoboDSL parser."""

import pytest
from robodsl.parser import parse_robodsl, NodeConfig, CudaKernelConfig, RoboDSLConfig

def test_parse_empty_config():
    """Test parsing an empty configuration."""
    config = parse_robodsl("")
    assert isinstance(config, RoboDSLConfig)
    assert len(config.nodes) == 0
    assert len(config.cuda_kernels) == 0

def test_parse_node_with_publisher():
    """Test parsing a node with a publisher."""
    config = parse_robodsl("""
    node test_node {
        publisher /camera/image_raw sensor_msgs/msg/Image
    }
    """)
    
    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "test_node"
    assert len(node.publishers) == 1
    assert node.publishers[0]["topic"] == "/camera/image_raw"
    assert node.publishers[0]["msg_type"] == "sensor_msgs/msg/Image"

def test_parse_node_with_subscriber():
    """Test parsing a node with a subscriber."""
    config = parse_robodsl("""
    node test_node {
        subscriber /cmd_vel geometry_msgs/msg/Twist
    }
    """)
    
    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "test_node"
    assert len(node.subscribers) == 1
    assert node.subscribers[0]["topic"] == "/cmd_vel"
    assert node.subscribers[0]["msg_type"] == "geometry_msgs/msg/Twist"

def test_parse_node_with_service():
    """Test parsing a node with a service."""
    config = parse_robodsl("""
    node test_node {
        service /get_status std_srvs/srv/Trigger
    }
    """)
    
    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "test_node"
    assert len(node.services) == 1
    assert node.services[0]["service"] == "/get_status"
    assert node.services[0]["srv_type"] == "std_srvs/srv/Trigger"

def test_parse_node_with_parameters():
    """Test parsing a node with parameters."""
    config = parse_robodsl("""
    node test_node {
        parameter max_speed = 1.5
        parameter use_gpu: true
        parameter sensor_name: "front_camera"
    }
    """)
    
    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "test_node"
    assert len(node.parameters) == 3
    assert node.parameters["max_speed"] == "1.5"
    assert node.parameters["use_gpu"] == "true"
    assert node.parameters["sensor_name"] == "\"front_camera\""

def test_parse_complete_node():
    """Test parsing a complete node with all features."""
    config = parse_robodsl("""
    node robot_controller {
        # Publishers
        publisher /odom nav_msgs/msg/Odometry
        publisher /status std_msgs/msg/String
        
        # Subscribers
        subscriber /cmd_vel geometry_msgs/msg/Twist
        
        # Services
        service /get_status std_srvs/srv/Trigger
        service /reset_odom std_srvs/srv/Empty
        
        # Parameters
        parameter wheel_radius = 0.1
        parameter max_speed: 2.5
        parameter use_gpu: true
        parameter robot_name: "turtlebot1"
    }
    """)
    
    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "robot_controller"
    
    # Verify publishers
    assert len(node.publishers) == 2
    assert node.publishers[0] == {"topic": "/odom", "msg_type": "nav_msgs/msg/Odometry"}
    assert node.publishers[1] == {"topic": "/status", "msg_type": "std_msgs/msg/String"}
    
    # Verify subscribers
    assert len(node.subscribers) == 1
    assert node.subscribers[0] == {"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist"}
    
    # Verify services
    assert len(node.services) == 2
    assert node.services[0] == {"service": "/get_status", "srv_type": "std_srvs/srv/Trigger"}
    assert node.services[1] == {"service": "/reset_odom", "srv_type": "std_srvs/srv/Empty"}
    
    # Verify parameters
    assert len(node.parameters) == 4
    assert node.parameters["wheel_radius"] == "0.1"
    assert node.parameters["max_speed"] == "2.5"
    assert node.parameters["use_gpu"] == "true"
    assert node.parameters["robot_name"] == "\"turtlebot1\""

def test_parse_cuda_kernel():
    """Test parsing a CUDA kernel definition."""
    config = parse_robodsl("""
    cuda_kernels {
        kernel process_image {
            input: Image (width=640, height=480)
            output: Image
            block_size: (16, 16, 1)
        }
    }
    """)
    
    assert len(config.cuda_kernels) == 1
    kernel = config.cuda_kernels[0]
    assert kernel.name == "process_image"
    assert len(kernel.inputs) == 1
    assert kernel.inputs[0]["type"] == "Image"
    assert kernel.inputs[0]["width"] == "640"
    assert kernel.inputs[0]["height"] == "480"
    assert len(kernel.outputs) == 1
    assert kernel.outputs[0]["type"] == "Image"
    assert kernel.block_size == (16, 16, 1)

def test_parse_complete_example():
    """Test parsing a complete example configuration."""
    config = parse_robodsl("""
    # Main processing node
    node image_processor {
        subscriber /camera/image_raw sensor_msgs/msg/Image
        publisher /processed_image sensor_msgs/msg/Image
    }

    # CUDA kernels
    cuda_kernels {
        kernel process_image {
            input: Image (width=640, height=480, channels=3)
            output: Image (channels=1)
            block_size: [16, 16, 1]
        }
        
        kernel detect_edges {
            input: Image
            output: Image
            block_size: 32, 32, 1
        }
    }
    """)
    
    # Verify nodes
    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "image_processor"
    assert len(node.subscribers) == 1
    assert node.subscribers[0]["topic"] == "/camera/image_raw"
    assert node.subscribers[0]["msg_type"] == "sensor_msgs/msg/Image"
    
    assert len(node.publishers) == 1
    assert node.publishers[0]["topic"] == "/processed_image"
    assert node.publishers[0]["msg_type"] == "sensor_msgs/msg/Image"
    
    # Verify CUDA kernels
    assert len(config.cuda_kernels) == 2
    
    # First kernel
    kernel1 = config.cuda_kernels[0]
    assert kernel1.name == "process_image"
    assert len(kernel1.inputs) == 1
    assert kernel1.inputs[0]["type"] == "Image"
    assert kernel1.inputs[0]["width"] == "640"
    assert kernel1.inputs[0]["height"] == "480"
    assert kernel1.inputs[0]["channels"] == "3"
    assert len(kernel1.outputs) == 1
    assert kernel1.outputs[0]["type"] == "Image"
    assert kernel1.outputs[0]["channels"] == "1"
    assert kernel1.block_size == (16, 16, 1)
    
    # Second kernel
    kernel2 = config.cuda_kernels[1]
    assert kernel2.name == "detect_edges"
    assert len(kernel2.inputs) == 1
    assert kernel2.inputs[0]["type"] == "Image"
    assert len(kernel2.outputs) == 1
    assert kernel2.outputs[0]["type"] == "Image"
    assert kernel2.block_size == (32, 32, 1)
