"""Tests for the new Lark-based parser."""

import pytest
from robodsl.parser import parse_robodsl, SemanticError
from lark import ParseError


def test_basic_node_parsing():
    """Test basic node parsing with the new Lark parser."""
    content = """
    node test_node {
        parameter test_param: 42
        publisher /test_topic : "std_msgs/String"
        subscriber /input_topic : "std_msgs/String"
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "test_node"
    assert len(node.content.parameters) == 1
    assert node.content.parameters[0].name == "test_param"
    assert node.content.parameters[0].value.value == 42
    assert len(node.content.publishers) == 1
    assert node.content.publishers[0].topic == "/test_topic"
    assert len(node.content.subscribers) == 1
    assert node.content.subscribers[0].topic == "/input_topic"


def test_cuda_kernel_parsing():
    """Test CUDA kernel parsing with the new Lark parser."""
    content = """
    cuda_kernels {
        kernel test_kernel {
            param in float* input_data (N)
            param out float* output_data (N)
            block_size: (256, 1, 1)
            code: "int idx = blockIdx.x * blockDim.x + threadIdx.x;\nif (idx < N) {\n    output_data[idx] = input_data[idx] * 2.0f;\n}"
        }
    }
    """
    
    ast = parse_robodsl(content)
    
    assert ast.cuda_kernels is not None
    assert len(ast.cuda_kernels.kernels) == 1
    kernel = ast.cuda_kernels.kernels[0]
    assert kernel.name == "test_kernel"
    assert len(kernel.content.parameters) == 2
    assert kernel.content.block_size == (256, 1, 1)
    assert "output_data[idx] = input_data[idx] * 2.0f" in kernel.content.code


def test_qos_configuration():
    """Test QoS configuration parsing."""
    content = """
    node test_node {
        publisher /test_topic : "std_msgs/String" {
            qos {
                reliability: reliable
                durability: transient_local
                history: keep_last
                depth: 10
            }
        }
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert len(node.content.publishers) == 1
    qos = node.content.publishers[0].qos
    assert qos is not None
    assert len(qos.settings) == 4
    
    # Check settings
    setting_values = [s.value for s in qos.settings]
    assert 'reliable' in setting_values
    assert 'transient_local' in setting_values
    assert 'keep_last' in setting_values
    assert 10 in setting_values


def test_lifecycle_configuration():
    """Test lifecycle configuration parsing."""
    content = """
    node test_node {
        lifecycle {
            autostart: true
            cleanup_on_shutdown: false
        }
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.content.lifecycle is not None
    assert len(node.content.lifecycle.settings) == 2
    
    setting_names = [s.name for s in node.content.lifecycle.settings]
    setting_values = [s.value for s in node.content.lifecycle.settings]
    
    assert 'autostart' in setting_names
    assert 'cleanup_on_shutdown' in setting_names
    assert True in setting_values
    assert False in setting_values


def test_timer_configuration():
    """Test timer configuration parsing."""
    content = """
    node test_node {
        timer timer_callback: 1.0 {
            oneshot: true
            autostart: false
        }
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert len(node.content.timers) == 1
    timer = node.content.timers[0]
    assert timer.name == "timer_callback"
    assert timer.period == 1.0
    assert len(timer.settings) == 2
    
    setting_names = [s.name for s in timer.settings]
    setting_values = [s.value for s in timer.settings]
    
    assert 'oneshot' in setting_names
    assert 'autostart' in setting_names
    assert True in setting_values
    assert False in setting_values


def test_include_statements():
    """Test include statement parsing."""
    content = """
    include <ros/ros.h>
    include "custom_header.h"
    
    node test_node {
        parameter test_param: 42
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.includes) == 2
    include_paths = [inc.path for inc in ast.includes]
    assert "ros/ros.h" in include_paths
    assert "custom_header.h" in include_paths
    assert len(ast.nodes) == 1


def test_comments():
    """Test comment handling."""
    content = """
    // This is a comment
    node test_node {
        parameter test_param: 42  // Another comment
        // publisher "/test_topic" "std_msgs/String"  // Commented out
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert len(node.content.parameters) == 1
    assert len(node.content.publishers) == 0  # Should be ignored due to comment


def test_semantic_errors():
    """Test semantic error detection."""
    content = """
    node test_node {
        parameter test_param: 42
        parameter test_param: 43  // Duplicate parameter name
    }
    """
    
    with pytest.raises(SemanticError) as exc_info:
        parse_robodsl(content)
    
    assert "Duplicate parameter name" in str(exc_info.value)


def test_parse_errors():
    """Test parse error handling."""
    content = """
    node test_node {
        parameter test_param: 42
        invalid_syntax_here
    }
    """
    
    with pytest.raises(ParseError):
        parse_robodsl(content)


def test_complex_value_types():
    """Test complex value types (arrays, nested dicts)."""
    content = """
    node test_node {
        parameter int_array: [1, 2, 3, 4, 5]
        parameter float_array: [1.0, 2.5, 3.14]
        parameter string_array: ["hello", "world"]
        parameter nested_dict: {
            key1: "value1",
            key2: 42,
            key3: true
        }
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert len(node.content.parameters) == 4
    
    # Check array parameters
    int_array_param = next(p for p in node.content.parameters if p.name == "int_array")
    assert int_array_param.value.value == [1, 2, 3, 4, 5]
    
    float_array_param = next(p for p in node.content.parameters if p.name == "float_array")
    assert float_array_param.value.value == [1.0, 2.5, 3.14]
    
    string_array_param = next(p for p in node.content.parameters if p.name == "string_array")
    assert string_array_param.value.value == ["hello", "world"]


def test_remapping():
    """Test topic remapping."""
    content = """
    node test_node {
        remap from: /original_topic to: /new_topic
        subscriber /original_topic : "std_msgs/String"
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert len(node.content.remaps) == 1
    remap = node.content.remaps[0]
    assert remap.from_topic == "/original_topic"
    assert remap.to_topic == "/new_topic"


def test_namespace():
    """Test namespace configuration."""
    content = """
    node test_node {
        namespace : /my/namespace
        parameter test_param: 42
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.content.namespace is not None
    assert node.content.namespace.namespace == "/my/namespace"


def test_cpp_method_parsing():
    """Test parsing of a node with an inline C++ method."""
    content = '''
    node cpp_node {
        cpp_method do_something : code : "int x = 42;\nstd::cout << x << std::endl;"
    }
    '''
    ast = parse_robodsl(content)
    node = ast.nodes[0]
    assert node.name == 'cpp_node'
    cpp_methods = node.content.cpp_methods
    assert len(cpp_methods) == 1
    assert cpp_methods[0].name == 'do_something'
    assert 'int x = 42;' in cpp_methods[0].code
    assert 'std::cout' in cpp_methods[0].code


if __name__ == "__main__":
    # Run tests
    test_basic_node_parsing()
    test_cuda_kernel_parsing()
    test_qos_configuration()
    test_lifecycle_configuration()
    test_timer_configuration()
    test_include_statements()
    test_comments()
    test_complex_value_types()
    test_remapping()
    test_namespace()
    test_cpp_method_parsing()
    
    print("All tests passed!") 