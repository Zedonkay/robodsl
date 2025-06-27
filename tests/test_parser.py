"""Tests for the RoboDSL parser."""

import pytest
from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.core.ast import RoboDSLAST, NodeNode, PublisherNode, SubscriberNode, ServiceNode, ParameterNode, KernelNode
from robodsl.parsers.semantic_analyzer import SemanticAnalyzer, SemanticError

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
            input: float* input (width)
            output: float* output (width)
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
    assert input_param.param_type == "float*"
    assert input_param.param_name == "input"
    assert input_param.size_expr == ["width"]
    
    output_param = kernel.content.parameters[1]
    assert output_param.direction.value == "out"
    assert output_param.param_type == "float*"
    assert output_param.param_name == "output"
    assert output_param.size_expr == ["width"]

def test_parse_cuda_kernels_new_syntax():
    """Test parsing CUDA kernels with new input: and output: syntax."""
    ast = parse_robodsl("""
    cuda_kernels {
        kernel image_processor {
            input: float* input_data (width, height)
            output: float* output_data (width, height)
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
    assert input_param.param_type == "float*"
    assert input_param.param_name == "input_data"
    assert input_param.size_expr == ["width", "height"]
    
    output_param = kernel.content.parameters[1]
    assert output_param.direction.value == "out"
    assert output_param.param_type == "float*"
    assert output_param.param_name == "output_data"
    assert output_param.size_expr == ["width", "height"]

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
            input: float* input (width)
            output: float* output (width)
            input: float kernel_size
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

def test_cross_reference_validation():
    """Test cross-reference validation with various scenarios."""
    # Test 1: Valid configuration with matching publisher/subscriber
    valid_config = """
    node publisher_node {
        publisher /test_topic: "std_msgs/msg/String"
    }
    
    node subscriber_node {
        subscriber /test_topic: "std_msgs/msg/String"
    }
    """
    
    ast = parse_robodsl(valid_config)
    analyzer = SemanticAnalyzer()
    assert analyzer.analyze(ast)
    assert len(analyzer.get_errors()) == 0
    assert len(analyzer.get_warnings()) == 0
    
    # Test 2: Subscriber without publisher (should warn)
    missing_publisher_config = """
    node subscriber_node {
        subscriber /test_topic: "std_msgs/msg/String"
    }
    """
    
    ast = parse_robodsl(missing_publisher_config)
    analyzer = SemanticAnalyzer()
    assert analyzer.analyze(ast)
    assert len(analyzer.get_errors()) == 0
    assert len(analyzer.get_warnings()) == 1
    assert "not published by any node" in analyzer.get_warnings()[0]
    
    # Test 3: Incompatible message types (should error)
    incompatible_types_config = """
    node publisher_node {
        publisher /test_topic: "std_msgs/msg/String"
    }
    
    node subscriber_node {
        subscriber /test_topic: "std_msgs/msg/Int32"
    }
    """
    
    try:
        ast = parse_robodsl(incompatible_types_config)
        assert False, "Expected SemanticError for incompatible message types"
    except SemanticError as e:
        assert "incompatible message types" in str(e)
    
    # Test 4: Multiple publishers with different types (should error)
    multiple_publishers_config = """
    node publisher1 {
        publisher /test_topic: "std_msgs/msg/String"
    }
    
    node publisher2 {
        publisher /test_topic: "std_msgs/msg/Int32"
    }
    """
    
    try:
        ast = parse_robodsl(multiple_publishers_config)
        assert False, "Expected SemanticError for multiple publishers with different types"
    except SemanticError as e:
        assert "multiple publishers with different message types" in str(e)
    
    # Test 5: Service with multiple providers (should error)
    multiple_service_providers_config = """
    node provider1 {
        service /test_service: "std_srvs/srv/Trigger"
    }
    
    node provider2 {
        service /test_service: "std_srvs/srv/Trigger"
    }
    """
    
    try:
        ast = parse_robodsl(multiple_service_providers_config)
        assert False, "Expected SemanticError for multiple service providers"
    except SemanticError as e:
        assert "provided by multiple nodes" in str(e)
    
    # Test 6: Circular remap (should error)
    circular_remap_config = """
    node test_node {
        publisher /test_topic: "std_msgs/msg/String"
        remap /test_topic: /test_topic
    }
    """
    
    try:
        ast = parse_robodsl(circular_remap_config)
        assert False, "Expected SemanticError for circular remap"
    except SemanticError as e:
        assert "circular remap" in str(e)
    
    # Test 7: Remap to non-existent topic (should warn)
    remap_nonexistent_config = """
    node test_node {
        subscriber /test_topic: "std_msgs/msg/String"
        remap /nonexistent_topic: /test_topic
    }
    """
    
    ast = parse_robodsl(remap_nonexistent_config)
    analyzer = SemanticAnalyzer()
    assert analyzer.analyze(ast)
    warnings = analyzer.get_warnings()
    assert len(warnings) == 2
    assert any("not defined in the current configuration" in w for w in warnings)
    assert any("not published by any node" in w for w in warnings)
