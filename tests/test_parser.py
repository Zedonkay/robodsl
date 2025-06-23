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
    assert node.publishers[0].topic == "/camera/image_raw"
    assert node.publishers[0].msg_type == "sensor_msgs/msg/Image"

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
    assert node.subscribers[0].topic == "/cmd_vel"
    assert node.subscribers[0].msg_type == "geometry_msgs/msg/Twist"

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
    assert node.services[0].service == "/get_status"
    assert node.services[0].srv_type == "std_srvs/srv/Trigger"

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
    # Convert parameters to a dict for easier lookup
    param_dict = {param.name: param for param in node.parameters}
    
    assert "max_speed" in param_dict
    assert param_dict["max_speed"].default == 1.5
    assert param_dict["max_speed"].type == "double"
    
    assert "use_gpu" in param_dict
    assert param_dict["use_gpu"].default is True
    assert param_dict["use_gpu"].type == "bool"
    
    assert "sensor_name" in param_dict
    assert param_dict["sensor_name"].default == "front_camera"
    assert param_dict["sensor_name"].type == "string"

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
    pub_topics = [p.topic for p in node.publishers]
    assert "/odom" in pub_topics
    assert "/status" in pub_topics
    
    odom_pub = next(p for p in node.publishers if p.topic == "/odom")
    status_pub = next(p for p in node.publishers if p.topic == "/status")
    
    assert odom_pub.msg_type == "nav_msgs/msg/Odometry"
    assert status_pub.msg_type == "std_msgs/msg/String"
    
    # Verify subscribers
    assert len(node.subscribers) == 1
    assert node.subscribers[0].topic == "/cmd_vel"
    assert node.subscribers[0].msg_type == "geometry_msgs/msg/Twist"
    
    # Verify services
    assert len(node.services) == 2
    service_names = [s.service for s in node.services]
    assert "/get_status" in service_names
    assert "/reset_odom" in service_names
    
    status_srv = next(s for s in node.services if s.service == "/get_status")
    reset_srv = next(s for s in node.services if s.service == "/reset_odom")
    
    assert status_srv.srv_type == "std_srvs/srv/Trigger"
    assert reset_srv.srv_type == "std_srvs/srv/Empty"
    
    # Verify parameters
    assert len(node.parameters) == 4
    param_names = [p.name for p in node.parameters]
    assert "wheel_radius" in param_names
    assert "max_speed" in param_names
    assert "use_gpu" in param_names
    assert "robot_name" in param_names
    
    wheel_param = next(p for p in node.parameters if p.name == "wheel_radius")
    max_speed_param = next(p for p in node.parameters if p.name == "max_speed")
    use_gpu_param = next(p for p in node.parameters if p.name == "use_gpu")
    robot_name_param = next(p for p in node.parameters if p.name == "robot_name")
    
    assert wheel_param.value == 0.1
    assert max_speed_param.value == 2.5
    assert use_gpu_param.value is True
    assert robot_name_param.value == "turtlebot1"

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
    
    # Check inputs (parameters with direction='in')
    input_params = [p for p in kernel.parameters if p.direction == 'in']
    assert len(input_params) == 1
    assert input_params[0].type == "Image"
    assert input_params[0].get('width') == 640
    assert input_params[0].get('height') == 480
    
    # Check outputs (parameters with direction='out')
    output_params = [p for p in kernel.parameters if p.direction == 'out']
    assert len(output_params) == 1
    assert output_params[0].type == "Image"
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
    assert node.subscribers[0].topic == "/camera/image_raw"
    assert node.subscribers[0].msg_type == "sensor_msgs/msg/Image"
    assert len(node.publishers) == 1
    assert node.publishers[0].topic == "/processed_image"
    assert node.publishers[0].msg_type == "sensor_msgs/msg/Image"
    
    # Verify CUDA kernels
    assert len(config.cuda_kernels) == 2
    
    # First kernel
    kernel1 = config.cuda_kernels[0]
    assert kernel1.name == "process_image"
    assert len(kernel1.inputs) == 1
    assert kernel1.inputs[0]["type"] == "Image"
    assert kernel1.inputs[0]["width"] == 640
    assert kernel1.inputs[0]["height"] == 480
    assert kernel1.inputs[0]["channels"] == 3
    assert len(kernel1.outputs) == 1
    assert kernel1.outputs[0]["type"] == "Image"
    assert kernel1.outputs[0]["channels"] == 1
    assert kernel1.block_size == (16, 16, 1)
    
    # Second kernel
    kernel2 = config.cuda_kernels[1]
    assert kernel2.name == "detect_edges"
    assert len(kernel2.inputs) == 1
    assert kernel2.inputs[0]["type"] == "Image"
    assert len(kernel2.outputs) == 1
    assert kernel2.outputs[0]["type"] == "Image"
    assert kernel2.block_size == (32, 32, 1)
