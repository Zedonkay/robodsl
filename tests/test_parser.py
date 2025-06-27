"""Tests for the RoboDSL parser."""

import pytest
from robodsl.parser import parse_robodsl, RoboDSLAST, NodeNode, PublisherNode, SubscriberNode, ServiceNode, ParameterNode, KernelNode

def get_param(node, name):
    for p in node.content.parameters:
        if p.name == name:
            return p
    return None

def test_parse_empty_config():
    """Test parsing an empty configuration."""
    ast = parse_robodsl("")
    assert isinstance(ast, RoboDSLAST)
    assert len(ast.nodes) == 0
    assert ast.cuda_kernels is None

def test_parse_node_with_publisher():
    """Test parsing a node with a publisher."""
    ast = parse_robodsl("""
    node test_node {
        publisher /camera/image_raw: "sensor_msgs/msg/Image"
    }
    """)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "test_node"
    assert len(node.content.publishers) == 1
    pub = node.content.publishers[0]
    assert pub.topic == "/camera/image_raw"
    assert pub.msg_type == "sensor_msgs/msg/Image"

def test_parse_node_with_subscriber():
    """Test parsing a node with a subscriber."""
    ast = parse_robodsl("""
    node test_node {
        subscriber /camera/image_raw: "sensor_msgs/msg/Image"
    }
    """)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "test_node"
    assert len(node.content.subscribers) == 1
    sub = node.content.subscribers[0]
    assert sub.topic == "/camera/image_raw"
    assert sub.msg_type == "sensor_msgs/msg/Image"

def test_parse_node_with_service():
    """Test parsing a node with a service."""
    ast = parse_robodsl("""
    node test_node {
        service /get_status: "std_srvs/srv/Trigger"
    }
    """)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "test_node"
    assert len(node.content.services) == 1
    service = node.content.services[0]
    assert service.service == "/get_status"
    assert service.srv_type == "std_srvs/srv/Trigger"

def test_parse_node_with_parameters():
    """Test parsing a node with parameters."""
    ast = parse_robodsl("""
    node test_node {
        parameter camera_fps: 30
        parameter camera_resolution: "640x480"
        parameter enable_debug: true
    }
    """)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "test_node"
    assert len(node.content.parameters) == 3
    
    fps_param = get_param(node, "camera_fps")
    assert fps_param is not None
    assert fps_param.value.value == 30
    
    res_param = get_param(node, "camera_resolution")
    assert res_param is not None
    assert res_param.value.value == "640x480"
    
    debug_param = get_param(node, "enable_debug")
    assert debug_param is not None
    assert debug_param.value.value is True

def test_parse_node_with_qos():
    """Test parsing a node with QoS configuration."""
    ast = parse_robodsl("""
    node test_node {
        publisher /test_topic: "std_msgs/msg/String" {
            qos {
                reliability: reliable
                depth: 10
            }
        }
    }
    """)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert len(node.content.publishers) == 1
    pub = node.content.publishers[0]
    assert pub.qos is not None
    # Debug: print QoS settings
    print(f"QoS settings: {pub.qos.settings}")
    # Extract settings from QoSNode
    reliability = None
    depth = None
    for setting in pub.qos.settings:
        if setting.name == "reliability":
            reliability = setting.value
        if setting.name == "depth":
            depth = setting.value
    assert reliability == "reliable"
    assert depth == 10

def test_parse_cuda_kernels():
    """Test parsing CUDA kernels."""
    ast = parse_robodsl("""
    cuda_kernels {
        kernel image_processor {
            param in Image input (width)
            param out Image output (width)
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
        }
    }
    """)
    
    assert ast.cuda_kernels is not None
    assert len(ast.cuda_kernels.kernels) == 1
    kernel = ast.cuda_kernels.kernels[0]
    assert kernel.name == "image_processor"
    # Use correct attribute for kernel parameters
    assert len(kernel.content.parameters) == 2
    
    input_param = kernel.content.parameters[0]
    assert input_param.direction.value == "in"
    assert input_param.param_type == "Image"
    assert input_param.param_name == "input"
    assert input_param.size_expr == "width"
    
    output_param = kernel.content.parameters[1]
    assert output_param.direction.value == "out"
    assert output_param.param_type == "Image"
    assert output_param.param_name == "output"
    assert output_param.size_expr == "width"

def test_parse_include():
    """Test parsing include statements."""
    ast = parse_robodsl("""
    include "common_config.robodsl"
    include <system_header.h>
    
    node test_node {
        publisher /test: "std_msgs/msg/String"
    }
    """)
    
    assert len(ast.includes) == 2
    assert ast.includes[0].path == "common_config.robodsl"
    assert ast.includes[0].is_system is False
    assert ast.includes[1].path == "system_header.h"
    assert ast.includes[1].is_system is True

def test_parse_complex_config():
    """Test parsing a complex configuration with multiple nodes and features."""
    ast = parse_robodsl("""
    include "common.robodsl"
    
    node camera_node {
        publisher /camera/image_raw: "sensor_msgs/msg/Image" {
            qos {
                reliability: reliable
                depth: 5
            }
        }
        parameter fps: 30
        parameter resolution: "1920x1080"
    }
    
    node processor_node {
        subscriber /camera/image_raw: "sensor_msgs/msg/Image"
        publisher /processed/image: "sensor_msgs/msg/Image"
        service /process_image: "image_processing/srv/ProcessImage"
        parameter algorithm: "gaussian_blur"
    }
    
    cuda_kernels {
        kernel image_filter {
            param in Image input (width)
            param out Image output (width)
            param in float kernel_size
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
        }
    }
    """)
    
    assert len(ast.includes) == 1
    assert len(ast.nodes) == 2
    assert ast.cuda_kernels is not None
    
    # Check camera node
    camera_node = ast.nodes[0]
    assert camera_node.name == "camera_node"
    assert len(camera_node.content.publishers) == 1
    assert len(camera_node.content.parameters) == 2
    
    # Check processor node
    processor_node = ast.nodes[1]
    assert processor_node.name == "processor_node"
    assert len(processor_node.content.subscribers) == 1
    assert len(processor_node.content.publishers) == 1
    assert len(processor_node.content.services) == 1
    assert len(processor_node.content.parameters) == 1
    
    # Check CUDA kernel
    kernel = ast.cuda_kernels.kernels[0]
    assert kernel.name == "image_filter"
    assert len(kernel.content.parameters) == 3
