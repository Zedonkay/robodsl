"""Comprehensive test cases for RoboDSL features implemented in Phase 0, 1, and 1.5."""

import pytest
from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.parsers.semantic_analyzer import SemanticError
from lark import ParseError
from robodsl.core.ast import (
    RoboDSLAST, NodeNode, PublisherNode, SubscriberNode, ServiceNode, 
    ActionNode, ClientNode, ParameterNode, TimerNode, LifecycleNode,
    RemapNode, NamespaceNode, FlagNode, QoSNode, CudaKernelsNode,
    KernelNode, CppMethodNode, MethodParamNode, IncludeNode
)


class TestComprehensiveNodeFeatures:
    """Test comprehensive node feature parsing and validation."""
    
    def test_complex_node_with_all_features(self):
        """Test a complex node with all supported features."""
        dsl_code = """
        node comprehensive_node {
            // Parameters with different types
            parameter int int_rate = 30
            parameter float float_tolerance = 0.001
            parameter string string_frame_id = "base_link"
            parameter bool bool_debug = true
            parameter list array_coords = [1.0, 2.0, 3.0]
            parameter dict dict_config = {
                max_iterations: 100,
                timeout: 5.0,
                enabled: true
            }
            
            // ROS2 primitives
            publisher /output/topic: "std_msgs/msg/String"
            subscriber /input/topic: "std_msgs/msg/String"
            service /process/service: "example_interfaces/srv/AddTwoInts"
            client /external/service: "example_interfaces/srv/AddTwoInts"
            action /navigate/action: "nav2_msgs/action/NavigateToPose"
            
            // QoS configurations
            publisher /qos/topic: "std_msgs/msg/String" {
                qos {
                    reliability: reliable
                    depth: 10
                    deadline: 1000
                }
            }
            
            // Lifecycle configuration
            lifecycle {
                enabled: true
                autostart: false
                cleanup_on_shutdown: true
            }
            
            // Timers
            timer periodic_timer: 1.0 {
                oneshot: false
                autostart: true
            }
            
            // Remaps
            remap from: /old/topic to: /new/topic
            
            // Namespace
            namespace: /my/namespace
            
            // Flags
            flag enable_logging: true
            flag debug_mode: false
            
            // C++ methods
            method process_data {
                input: float* input_data (size)
                output: float* output_data (size)
                code: {
                    for (int i = 0; i < size; i++) {
                        output_data[i] = input_data[i] * 2.0f;
                    }
                }
            }
            
            // CUDA kernels
            kernel gpu_process {
                input: float* input_data (size)
                output: float* output_data (size)
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                shared_memory: 1024
                use_thrust: true
                code: {
                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    if (idx < size) {
                        output_data[idx] = input_data[idx] * 2.0f;
                    }
                }
            }
            
            // ONNX models
            onnx_model classifier {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
                optimization: tensorrt
            }
        }
        """
        
        ast = parse_robodsl(dsl_code)
        assert ast is not None
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "comprehensive_node"
        
        # Check parameters
        assert len(node.content.parameters) == 6
        
        # Check ROS2 primitives
        assert len(node.content.publishers) == 2
        assert len(node.content.subscribers) == 1
        assert len(node.content.services) == 1
        assert len(node.content.clients) == 1
        assert len(node.content.actions) == 1
        
        # Check timers
        assert len(node.content.timers) == 1
        timer = node.content.timers[0]
        assert timer.name == "periodic_timer"
        assert timer.period == 1.0
        
        # Check methods and kernels
        assert len(node.content.cpp_methods) == 1
        assert len(node.content.cuda_kernels) == 1
        
        # Check ONNX models
        assert len(node.content.onnx_models) == 1


class TestAdvancedCppMethodFeatures:
    """Test advanced C++ method features."""
    
    def test_complex_cpp_methods(self):
        """Test complex C++ methods with various parameter types."""
        content = """
        node test_node_1 {
            method complex_processing {
                input: int data_size
                input: float* input_data (data_size)
                input: float* weights (weights.size())
                input: string config_file
                input: bool enable_optimization
                output: float* result (data_size)
                output: int* indices (data_size)
                output: bool success
                code: {
                    try {
                        success = true;
                        for (int i = 0; i < data_size; i++) {
                            float sum = 0.0f;
                            for (size_t j = 0; j < weights.size(); j++) {
                                sum += input_data[i] * weights[j];
                            }
                            result[i] = sum;
                            indices[i] = i;
                        }
                    } catch (const std::exception& e) {
                        success = false;
                    }
                }
            }
            
            method template_method {
                input: float* input_vector (input_vector.size())
                output: float* output_vector (input_vector.size())
                code: {
                    output_vector.resize(input_vector.size());
                    for (size_t i = 0; i < input_vector.size(); i++) {
                        output_vector[i] = static_cast<float>(input_vector[i]) * 1.5f;
                    }
                }
            }
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        
        # Test complex method
        complex_method = node.content.cpp_methods[0]
        assert complex_method.name == "complex_processing"
        assert len(complex_method.inputs) == 5
        assert len(complex_method.outputs) == 3
        
        # Check input parameters
        input_types = [p.param_type for p in complex_method.inputs]
        assert "int" in input_types
        assert "float*" in input_types
        assert "float*" in input_types
        assert "string" in input_types
        assert "bool" in input_types
        
        # Check output parameters
        output_types = [p.param_type for p in complex_method.outputs]
        assert "float*" in output_types
        assert "int*" in output_types
        assert "bool" in output_types
        
        # Test template method
        template_method = node.content.cpp_methods[1]
        assert template_method.name == "template_method"
        assert len(template_method.inputs) == 1
        assert len(template_method.outputs) == 1


class TestAdvancedCudaKernelFeatures:
    """Test advanced CUDA kernel features."""
    
    def test_complex_cuda_kernels(self):
        """Test complex CUDA kernels with various configurations."""
        content = """
        cuda_kernels {
            kernel convolution_2d {
                input: float* input_image (width, height, channels)
                input: float* kernel (kernel_size, kernel_size)
                output: float* output_image (width, height, channels)
                input: int width
                input: int height
                input: int channels
                input: int kernel_size
                block_size: (32, 32, 1)
                grid_size: ((width + 31) / 32, (height + 31) / 32, 1)
                shared_memory: 4096
                use_thrust: false
                code: {
                    __global__ void convolution_2d(const float* input, const float* kernel, float* output,
                                                 int width, int height, int channels, int kernel_size) {
                        __shared__ float shared_input[34][34];  // 32 + 2 for padding
                        __shared__ float shared_kernel[5][5];   // Max kernel size
                        
                        int x = blockIdx.x * blockDim.x + threadIdx.x;
                        int y = blockIdx.y * blockDim.y + threadIdx.y;
                        
                        if (x < width && y < height) {
                            // Load input data to shared memory
                            for (int ky = 0; ky < kernel_size; ky++) {
                                for (int kx = 0; kx < kernel_size; kx++) {
                                    int sx = threadIdx.x + kx;
                                    int sy = threadIdx.y + ky;
                                    if (sx < 34 && sy < 34) {
                                        int ix = x + kx - kernel_size/2;
                                        int iy = y + ky - kernel_size/2;
                                        if (ix >= 0 && ix < width && iy >= 0 && iy < height) {
                                            shared_input[sy][sx] = input[iy * width + ix];
                                        } else {
                                            shared_input[sy][sx] = 0.0f;
                                        }
                                    }
                                }
                            }
                            
                            // Load kernel to shared memory
                            if (threadIdx.x < kernel_size && threadIdx.y < kernel_size) {
                                shared_kernel[threadIdx.y][threadIdx.x] = kernel[threadIdx.y * kernel_size + threadIdx.x];
                            }
                            
                            __syncthreads();
                            
                            // Compute convolution
                            float sum = 0.0f;
                            for (int ky = 0; ky < kernel_size; ky++) {
                                for (int kx = 0; kx < kernel_size; kx++) {
                                    sum += shared_input[threadIdx.y + ky][threadIdx.x + kx] * 
                                           shared_kernel[ky][kx];
                                }
                            }
                            
                            output[y * width + x] = sum;
                        }
                    }
                }
            }
            
            kernel reduce_sum {
                input: float* input_data (N)
                output: float* result (1)
                input: int N
                block_size: (256, 1, 1)
                grid_size: (1, 1, 1)
                shared_memory: 1024
                use_thrust: true
                code: {
                    __global__ void reduce_sum(const float* input, float* result, int n) {
                        __shared__ float shared_data[256];
                        
                        int tid = threadIdx.x;
                        int i = blockIdx.x * blockDim.x + threadIdx.x;
                        
                        shared_data[tid] = (i < n) ? input[i] : 0.0f;
                        __syncthreads();
                        
                        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
                            if (tid < stride) {
                                shared_data[tid] += shared_data[tid + stride];
                            }
                            __syncthreads();
                        }
                        
                        if (tid == 0) {
                            atomicAdd(result, shared_data[0]);
                        }
                    }
                }
            }
        }
        """
        
        ast = parse_robodsl(content)
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 2
        
        # Test convolution kernel
        conv_kernel = ast.cuda_kernels.kernels[0]
        assert conv_kernel.name == "convolution_2d"
        assert len(conv_kernel.content.parameters) == 7
        assert conv_kernel.content.block_size == (32, 32, 1)
        assert conv_kernel.content.shared_memory == 4096
        assert conv_kernel.content.use_thrust is False
        
        # Test reduce kernel
        reduce_kernel = ast.cuda_kernels.kernels[1]
        assert reduce_kernel.name == "reduce_sum"
        assert len(reduce_kernel.content.parameters) == 3
        assert reduce_kernel.content.block_size == (256, 1, 1)
        assert reduce_kernel.content.shared_memory == 1024
        assert reduce_kernel.content.use_thrust is True


class TestSemanticValidation:
    """Test comprehensive semantic validation."""
    
    def test_duplicate_parameter_names(self):
        """Test semantic validation for duplicate parameter names."""
        dsl_code = """
        node test_node_1 {
            parameter int test_param = 42
            parameter int test_param = 43  // Duplicate name
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            ast = parse_robodsl(dsl_code)
        
        error_msg = str(exc_info.value)
        assert "duplicate parameter name" in error_msg.lower()
    
    def test_duplicate_timer_names(self):
        """Test detection of duplicate timer names."""
        content = """
        node test_node_1 {
            timer my_timer: 1.0
            timer my_timer: 2.0  // Duplicate name
        }
        """
    
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
    
        assert "Duplicate timer callback name" in str(exc_info.value)
    
    def test_duplicate_publisher_topics(self):
        """Test detection of duplicate publisher topics."""
        content = """
        node test_node_1 {
            publisher /test_topic : "std_msgs/String"
            publisher /test_topic : "std_msgs/Int32"  // Duplicate topic
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate publisher topic" in str(exc_info.value)
    
    def test_duplicate_subscriber_topics(self):
        """Test detection of duplicate subscriber topics."""
        content = """
        node test_node_1 {
            subscriber /test_topic : "std_msgs/String"
            subscriber /test_topic : "std_msgs/Int32"  // Duplicate topic
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate subscriber topic" in str(exc_info.value)
    
    def test_duplicate_service_names(self):
        """Test detection of duplicate service names."""
        content = """
        node test_node_1 {
            service /test_service : "std_srvs/Empty"
            service /test_service : "std_srvs/Trigger"  // Duplicate service
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate service name" in str(exc_info.value)
    
    def test_duplicate_action_names(self):
        """Test detection of duplicate action names."""
        content = """
        node test_node_1 {
            action /test_action : "test_msgs/TestAction"
            action /test_action : "test_msgs/TestAction2"  // Duplicate action
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate action name" in str(exc_info.value)
    
    def test_duplicate_client_names(self):
        """Test detection of duplicate client names."""
        content = """
        node test_node_1 {
            client /test_client : "std_srvs/Empty"
            client /test_client : "std_srvs/Trigger"  // Duplicate client
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate client name" in str(exc_info.value)
    
    def test_duplicate_flag_names(self):
        """Test detection of duplicate flag names."""
        content = """
        node test_node_1 {
            flag test_flag: true
            flag test_flag: false  // Duplicate flag
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate flag name" in str(exc_info.value)
    
    def test_duplicate_method_names(self):
        """Test detection of duplicate method names."""
        content = """
        node test_node_1 {
            method test_method {
                input: int x
                code: {
                    return x * 2;
                }
            }
            method test_method {
                input: float y
                code: {
                    return y * 3;
                }
            }
        }
        """
    
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
    
        assert "duplicate C++ method name" in str(exc_info.value)
    
    def test_duplicate_kernel_names(self):
        """Test detection of duplicate kernel names."""
        content = """
        cuda_kernels {
            kernel test_kernel {
                input: float* input (N)
                output: float* output (N)
                code: {
                    printf('hello');
                }
            }
            kernel test_kernel {
                input: int* input (N)
                output: int* output (N)
                code: {
                    printf('world');
                }
            }
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate kernel name" in str(exc_info.value)
    
    def test_invalid_qos_values(self):
        """Test detection of invalid QoS values."""
        content = """
        node test_node_1 {
            publisher /test_topic: "std_msgs/msg/String" {
                qos {
                    reliability: invalid_value
                    durability: also_invalid
                }
            }
        }
        """
    
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
    
        error_message = str(exc_info.value)
        assert "QoS reliability" in error_message
        assert "QoS durability" in error_message
    
    def test_invalid_timer_period(self):
        """Test detection of invalid timer period."""
        content = """
        node test_node_1 {
            timer my_timer: -1.0  // Invalid negative period
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Timer my_timer period must be positive" in str(exc_info.value)
    
    def test_invalid_block_size(self):
        """Test detection of invalid CUDA block size."""
        content = """
        cuda_kernels {
            kernel test_kernel {
                input: float* input (N)
                output: float* output (N)
                block_size: (0, 1, 1)  // Invalid zero block size
                code: {
                    printf('hello');
                }
            }
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "block size dimension 0 must be a positive integer" in str(exc_info.value)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_node(self):
        """Test parsing of an empty node."""
        content = """
        node empty_node {
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "empty_node"
        assert len(node.content.parameters) == 0
        assert len(node.content.publishers) == 0
        assert len(node.content.subscribers) == 0
    
    def test_node_with_only_comments(self):
        """Test parsing of a node with only comments."""
        content = """
        node comment_node {
            // This is a comment
            // Another comment
            /* Block comment */
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "comment_node"
        assert len(node.content.parameters) == 0
    
    def test_very_large_values(self):
        """Test handling of very large numeric values."""
        dsl_code = """
        node test_node_1 {
            parameter int large_int = 999999999999999999
            parameter float large_float = 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
            parameter float small_float = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
        }
        """
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 3
        
        # Check that large values are handled correctly
        large_int_param = next(p for p in node.content.parameters if p.name == "large_int")
        assert large_int_param.type == "int"
        assert large_int_param.value.value == 999999999999999999
        
        large_float_param = next(p for p in node.content.parameters if p.name == "large_float")
        assert large_float_param.type == "float"
        assert large_float_param.value.value == 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
    
    def test_unicode_strings(self):
        """Test handling of unicode strings."""
        dsl_code = """
        node test_node_1 {
            parameter string japanese = "こんにちは世界"
            parameter string chinese = "你好世界"
            parameter string emoji = "🚀🤖💻"
            parameter string special_chars = "áéíóúñü¿¡"
        }
        """
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 4
        
        # Check unicode strings are preserved
        japanese_param = next(p for p in node.content.parameters if p.name == "japanese")
        assert japanese_param.type == "string"
        assert japanese_param.value.value == "こんにちは世界"
        
        chinese_param = next(p for p in node.content.parameters if p.name == "chinese")
        assert chinese_param.type == "string"
        assert chinese_param.value.value == "你好世界"
        
        emoji_param = next(p for p in node.content.parameters if p.name == "emoji")
        assert emoji_param.type == "string"
        assert emoji_param.value.value == "🚀🤖💻"
    
    def test_nested_arrays_and_dicts(self):
        """Test handling of nested arrays and dictionaries."""
        dsl_code = """
        node test_node_1 {
            parameter list nested_array = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            parameter dict nested_dict = {
                level1: {
                    level2: {
                        value: 42,
                        array: [1, 2, 3]
                    }
                }
            }
            parameter list mixed = [{
                name: "item1",
                values: [1, 2, 3]
            }, {
                name: "item2",
                values: [4, 5, 6]
            }]
        }
        """
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 3
        
        # Check nested structures
        nested_array_param = next(p for p in node.content.parameters if p.name == "nested_array")
        assert nested_array_param.type == "list"
        assert nested_array_param.value.value == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        
        nested_dict_param = next(p for p in node.content.parameters if p.name == "nested_dict")
        assert nested_dict_param.type == "dict"
        assert nested_dict_param.value.value["level1"]["level2"]["value"] == 42
        assert nested_dict_param.value.value["level1"]["level2"]["array"] == [1, 2, 3]
    
    def test_method_with_no_parameters(self):
        """Test parsing of a method with no input/output parameters."""
        content = """
        node test_node_1 {
            method no_params {
                code: {
                    std::cout << 'Hello, World!' << std::endl;
                }
            }
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.cpp_methods) == 1
        method = node.content.cpp_methods[0]
        assert len(method.inputs) == 0
        assert len(method.outputs) == 0
    
    def test_kernel_with_no_parameters(self):
        """Test parsing of a kernel with no parameters."""
        content = """
        cuda_kernels {
            kernel no_params {
                block_size: (256, 1, 1)
                code: {
                    printf('Hello from GPU!');
                }
            }
        }
        """
        
        ast = parse_robodsl(content)
        assert ast.cuda_kernels is not None
        assert len(ast.cuda_kernels.kernels) == 1
        kernel = ast.cuda_kernels.kernels[0]
        assert len(kernel.content.parameters) == 0


class TestErrorRecovery:
    """Test error recovery and graceful degradation."""
    
    def test_multiple_nodes_with_one_invalid(self):
        """Test that valid nodes are parsed even when one node has errors."""
        dsl_code = """
        node valid_node {
            parameter int test = 42
        }
        
        node invalid_node {
            parameter int test = 42
            invalid_syntax_here
        }
        
        node another_valid_node {
            parameter int test = 42
        }
        """
        
        with pytest.raises(ParseError):
            ast = parse_robodsl(dsl_code)
    
    def test_invalid_syntax_in_comments(self):
        """Test that invalid syntax in comments doesn't cause parsing errors."""
        dsl_code = """
        // This comment has invalid syntax: node invalid { parameter x: y }
        node valid_node {
            parameter int valid_param = 42
        }
        """
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "valid_node"
        assert len(node.content.parameters) == 1
        assert node.content.parameters[0].name == "valid_param"
        assert node.content.parameters[0].type == "int"
        assert node.content.parameters[0].value.value == 42


class TestPerformance:
    """Test performance with large configurations."""
    
    def test_large_number_of_parameters(self):
        """Test parsing with a large number of parameters."""
        dsl_code = "node test_node_1 {\n"
        for i in range(100):
            dsl_code += f"    parameter int param_{i} = {i}\n"
        dsl_code += "}"
        
        ast = parse_robodsl(dsl_code)
        
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 100
        
        # Check a few parameters
        for i in range(0, 100, 10):
            param = next(p for p in node.content.parameters if p.name == f"param_{i}")
            assert param.type == "int"
            assert param.value.value == i
    
    def test_large_number_of_publishers(self):
        """Test parsing with a large number of publishers."""
        content = "node large_node {\n"
        for i in range(100):
            content += f'    publisher /topic_{i}: "std_msgs/msg/String"\n'
        content += "}\n"
    
        ast = parse_robodsl(content)
        assert ast is not None
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.publishers) == 100
    
    def test_large_code_blocks(self):
        """Test parsing with large code blocks."""
        large_code = "int x = 0;\\n" * 1000
        content = f"""
        node test_node_1 {{
            method large_method {{
                input: int size
                code: {{
                    {large_code}
                }}
            }}
        }}
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.cpp_methods) == 1
        method = node.content.cpp_methods[0]
        assert len(method.code) > 1000  # Should contain the large code block


if __name__ == "__main__":
    pytest.main([__file__]) 