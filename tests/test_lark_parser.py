"""Tests for the new Lark-based parser."""

import pytest
from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.parsers.semantic_analyzer import SemanticError
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
            code: {
                int idx = blockIdx.x * blockDim.x + threadIdx.x;
                if (idx < N) {
                    output_data[idx] = input_data[idx] * 2.0f;
                }
            }
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
    """Test basic C++ method parsing (legacy syntax)."""
    content = """
node test_node {
    method do_something {
        code: {
            int x = 42;
            std::cout << x << std::endl;
        }
    }
}
"""
    
    ast = parse_robodsl(content)
    assert len(ast.nodes) == 1
    
    node = ast.nodes[0]
    assert node.name == 'test_node'
    
    cpp_methods = node.content.cpp_methods
    assert len(cpp_methods) == 1
    assert cpp_methods[0].name == 'do_something'
    assert 'int x = 42;' in cpp_methods[0].code
    assert 'std::cout' in cpp_methods[0].code


def test_node_with_cuda_kernels():
    """Test parsing a node with CUDA kernels inside it."""
    content = """
    node processing_node {
        parameter input_size: 1024
        parameter output_size: 1024
        
        publisher /processed_data : "std_msgs/msg/Float32MultiArray"
        subscriber /raw_data : "std_msgs/msg/Float32MultiArray"
        
        kernel process_kernel {
            block_size: (256, 1, 1)
            grid_size: (4, 1, 1)
            shared_memory: 1024
            use_thrust: true
            param in float input_data (size)
            param out float output_data (size)
        }
        
        kernel filter_kernel {
            block_size: (128, 1, 1)
            grid_size: (8, 1, 1)
            shared_memory: 512
            use_thrust: false
            param in float data (width, height)
            param out float filtered_data (width, height)
        }
    }
    """
    
    ast = parse_robodsl(content)
    
    assert len(ast.nodes) == 1
    node = ast.nodes[0]
    assert node.name == "processing_node"
    
    # Check that the node has CUDA kernels
    assert len(node.content.cuda_kernels) == 2
    
    # Check first kernel
    kernel1 = node.content.cuda_kernels[0]
    assert kernel1.name == "process_kernel"
    assert kernel1.content.block_size == (256, 1, 1)
    assert kernel1.content.grid_size == (4, 1, 1)
    assert kernel1.content.shared_memory == 1024
    assert kernel1.content.use_thrust is True
    assert len(kernel1.content.parameters) == 2
    
    # Check second kernel
    kernel2 = node.content.cuda_kernels[1]
    assert kernel2.name == "filter_kernel"
    assert kernel2.content.block_size == (128, 1, 1)
    assert kernel2.content.grid_size == (8, 1, 1)
    assert kernel2.content.shared_memory == 512
    assert kernel2.content.use_thrust is False
    assert len(kernel2.content.parameters) == 2
    
    # Check that the node also has ROS2 components
    assert len(node.content.publishers) == 1
    assert len(node.content.subscribers) == 1
    assert len(node.content.parameters) == 2


def test_enhanced_cpp_method_parsing():
    """Test enhanced C++ method parsing with input/output parameters.
    NOTE: Pointer types must be written with a space, e.g., 'float * input_data', not 'float* input_data'.
    """
    config = """
node test_node {
    method process_data {
        input: int data_size
        input: float * input_data (data_size)
        output: float * output_data (data_size)
        code: {
            for (int i = 0; i < data_size; i++) { 
                output_data[i] = input_data[i] * 2.0f; 
            }
        }
    }
    
    method calculate_stats {
        input: std::vector<float> values
        output: float mean
        output: float variance
        code: {
            float sum = 0.0f; 
            for (auto v : values) sum += v; 
            mean = sum / values.size();
        }
    }
}
"""
    
    ast = parse_robodsl(config)
    assert len(ast.nodes) == 1
    
    node = ast.nodes[0]
    assert node.name == 'test_node'
    
    cpp_methods = node.content.cpp_methods
    assert len(cpp_methods) == 2
    
    # Check first method
    method1 = cpp_methods[0]
    assert method1.name == 'process_data'
    assert len(method1.inputs) == 2
    assert len(method1.outputs) == 1
    
    # Check input parameters
    assert method1.inputs[0].param_type == 'int'
    assert method1.inputs[0].param_name == 'data_size'
    assert method1.inputs[0].size_expr is None
    
    assert method1.inputs[1].param_type == 'float*'
    assert method1.inputs[1].param_name == 'input_data'
    assert method1.inputs[1].size_expr == 'data_size'
    
    # Check output parameters
    assert method1.outputs[0].param_type == 'float*'
    assert method1.outputs[0].param_name == 'output_data'
    assert method1.outputs[0].size_expr == 'data_size'
    
    # Check code
    assert 'for (int i = 0; i < data_size; i++)' in method1.code
    
    # Check second method
    method2 = cpp_methods[1]
    assert method2.name == 'calculate_stats'
    assert len(method2.inputs) == 1
    assert len(method2.outputs) == 2
    
    # Check input parameters
    assert method2.inputs[0].param_type == 'std::vector<float>'
    assert method2.inputs[0].param_name == 'values'
    assert method2.inputs[0].size_expr is None
    
    # Check output parameters
    assert method2.outputs[0].param_type == 'float'
    assert method2.outputs[0].param_name == 'mean'
    assert method2.outputs[0].size_expr is None
    
    assert method2.outputs[1].param_type == 'float'
    assert method2.outputs[1].param_name == 'variance'
    assert method2.outputs[1].size_expr is None
    
    # Check code
    assert 'float sum = 0.0f;' in method2.code


def test_enhanced_cpp_method_semantic_validation():
    """Test semantic validation of enhanced C++ methods."""
    config = """
node test_node {
    method valid_method {
        input: int data_size
        output: float result
        code: {
            result = data_size * 2.0f;
        }
    }
    
    method duplicate_input {
        input: int data_size
        input: int data_size  // Duplicate name
        output: float result
        code: {
            result = data_size * 2.0f;
        }
    }
    
    method input_output_conflict {
        input: int data_size
        output: int data_size  // Conflict with input
        code: {
            data_size = 42;
        }
    }
}
"""
    
    with pytest.raises(SemanticError) as exc_info:
        ast = parse_robodsl(config)
    
    # Should have errors due to duplicate and conflicting parameter names
    error_msg = str(exc_info.value)
    
    # Check for expected errors
    duplicate_error = "duplicate input parameter name: data_size" in error_msg
    conflict_error = "parameter name conflicts between inputs and outputs: data_size" in error_msg
    
    assert duplicate_error, "Should detect duplicate input parameter names"
    assert conflict_error, "Should detect parameter name conflicts between inputs and outputs"


if __name__ == "__main__":
    # Run tests
    test_basic_node_parsing()
    test_cuda_kernel_parsing()
    test_qos_configuration()
    test_lifecycle_configuration()
    test_timer_configuration()
    test_include_statements()
    test_comments()
    test_semantic_errors()
    test_parse_errors()
    test_complex_value_types()
    test_remapping()
    test_namespace()
    test_cpp_method_parsing()
    test_node_with_cuda_kernels()
    test_enhanced_cpp_method_parsing()
    test_enhanced_cpp_method_semantic_validation()
    
    print("All tests passed!") 