"""Comprehensive test cases for RoboDSL features implemented in Phase 0, 1, and 1.5."""

import pytest
from robodsl.parser import parse_robodsl, SemanticError
from lark import ParseError
from robodsl.ast import (
    RoboDSLAST, NodeNode, PublisherNode, SubscriberNode, ServiceNode, 
    ActionNode, ClientNode, ParameterNode, TimerNode, LifecycleNode,
    RemapNode, NamespaceNode, FlagNode, QoSNode, CudaKernelsNode,
    KernelNode, CppMethodNode, MethodParamNode, IncludeNode
)


class TestComprehensiveNodeFeatures:
    """Test comprehensive node feature parsing and validation."""
    
    def test_complex_node_with_all_features(self):
        """Test a node with all possible features."""
        content = """
        include <ros/ros.h>
        include "custom_header.h"
        
        node complex_node {
            // Parameters with various types
            parameter int_rate: 30
            parameter float_tolerance: 0.001
            parameter string_frame_id: "base_link"
            parameter bool_debug: true
            parameter array_coords: [1.0, 2.0, 3.0]
            parameter dict_config: {
                max_velocity: 2.0
                min_distance: 0.5
                timeout: 5.0
            }
            
            // Lifecycle configuration
            lifecycle {
                autostart: true
                cleanup_on_shutdown: true
            }
            
            // Timers with settings
            timer main_timer: 0.1 {
                oneshot: false
                autostart: true
            }
            timer debug_timer: 1.0 {
                oneshot: true
                autostart: false
            }
            
            // ROS primitives with QoS
            publisher /cmd_vel : "geometry_msgs/Twist" {
                qos {
                    reliability: reliable
                    durability: transient_local
                    history: keep_last
                    depth: 10
                }
            }
            
            subscriber /scan : "sensor_msgs/LaserScan" {
                qos {
                    reliability: best_effort
                    durability: volatile
                    history: keep_last
                    depth: 5
                }
            }
            
            service /set_goal : "nav_msgs/SetGoal" {
                qos {
                    reliability: reliable
                    durability: volatile
                }
            }
            
            client /get_map : "nav_msgs/GetMap" {
                qos {
                    reliability: reliable
                    durability: volatile
                }
            }
            
            action /navigate : "nav2_msgs/NavigateToPose" {
                qos {
                    reliability: reliable
                    durability: transient_local
                }
            }
            
            // Remapping
            remap from: /cmd_vel to: /robot/cmd_vel
            remap /scan: /robot/scan
            remap from: /odom to: /robot/odom
            
            // Namespace
            namespace: /robot
            
            // Flags
            flag enable_debug: true
            flag use_sim_time: false
            flag enable_logging: true
            
            // C++ methods
            method process_data {
                input: std::vector<float> input_data
                output: std::vector<float> output_data
                code: {
                    // Process input data
                    output_data.resize(input_data.size());
                    for (size_t i = 0; i < input_data.size(); ++i) {
                        output_data[i] = input_data[i] * 2.0f;
                    }
                }
            }
            
            method validate_input {
                input: std::string input_string
                output: bool is_valid
                code: {
                    is_valid = !input_string.empty() && input_string.length() > 0;
                }
            }
            
            // CUDA kernels
            kernel vector_add {
                param in float* a (N)
                param in float* b (N)
                param out float* c (N)
                param in int N
                block_size: (256, 1, 1)
                grid_size: ((N + 255) / 256, 1, 1)
                shared_memory: 0
                use_thrust: false
                code: {
                    __global__ void vector_add(const float* a, const float* b, float* c, int n) {
                        int i = blockIdx.x * blockDim.x + threadIdx.x;
                        if (i < n) {
                            c[i] = a[i] + b[i];
                        }
                    }
                }
            }
            
            kernel matrix_multiply {
                param in float* A (M, N)
                param in float* B (N, K)
                param out float* C (M, K)
                param in int M
                param in int N
                param in int K
                block_size: (16, 16, 1)
                grid_size: ((M + 15) / 16, (K + 15) / 16, 1)
                shared_memory: 1024
                use_thrust: true
                code: {
                    __global__ void matrix_multiply(const float* A, const float* B, float* C, int M, int N, int K) {
                        __shared__ float sA[16][16];
                        __shared__ float sB[16][16];
                        
                        int row = blockIdx.y * blockDim.y + threadIdx.y;
                        int col = blockIdx.x * blockDim.x + threadIdx.x;
                        
                        float sum = 0.0f;
                        for (int k = 0; k < N; k += 16) {
                            if (row < M && k + threadIdx.x < N) {
                                sA[threadIdx.y][threadIdx.x] = A[row * N + k + threadIdx.x];
                            } else {
                                sA[threadIdx.y][threadIdx.x] = 0.0f;
                            }
                            
                            if (k + threadIdx.y < N && col < K) {
                                sB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * K + col];
                            } else {
                                sB[threadIdx.y][threadIdx.x] = 0.0f;
                            }
                            
                            __syncthreads();
                            
                            for (int i = 0; i < 16; i++) {
                                sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
                            }
                            
                            __syncthreads();
                        }
                        
                        if (row < M && col < K) {
                            C[row * K + col] = sum;
                        }
                    }
                }
            }
        }
        """
        
        ast = parse_robodsl(content)
        
        # Test includes
        assert len(ast.includes) == 2
        include_paths = [inc.path for inc in ast.includes]
        assert "ros/ros.h" in include_paths
        assert "custom_header.h" in include_paths
        
        # Test node
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "complex_node"
        
        # Test parameters
        assert len(node.content.parameters) == 6
        param_names = [p.name for p in node.content.parameters]
        assert "int_rate" in param_names
        assert "float_tolerance" in param_names
        assert "string_frame_id" in param_names
        assert "bool_debug" in param_names
        assert "array_coords" in param_names
        assert "dict_config" in param_names
        
        # Test lifecycle
        assert node.content.lifecycle is not None
        assert len(node.content.lifecycle.settings) == 2
        
        # Test timers
        assert len(node.content.timers) == 2
        timer_names = [t.name for t in node.content.timers]
        assert "main_timer" in timer_names
        assert "debug_timer" in timer_names
        
        # Test ROS primitives
        assert len(node.content.publishers) == 1
        assert len(node.content.subscribers) == 1
        assert len(node.content.services) == 1
        assert len(node.content.clients) == 1
        assert len(node.content.actions) == 1
        
        # Test remaps
        assert len(node.content.remaps) == 3
        
        # Test namespace
        assert node.content.namespace is not None
        assert node.content.namespace.namespace == "/robot"
        
        # Test flags
        assert len(node.content.flags) == 3
        flag_names = [f.name for f in node.content.flags]
        assert "enable_debug" in flag_names
        assert "use_sim_time" in flag_names
        assert "enable_logging" in flag_names
        
        # Test C++ methods
        assert len(node.content.cpp_methods) == 2
        method_names = [m.name for m in node.content.cpp_methods]
        assert "process_data" in method_names
        assert "validate_input" in method_names
        
        # Test CUDA kernels
        assert len(node.content.cuda_kernels) == 2
        kernel_names = [k.name for k in node.content.cuda_kernels]
        assert "vector_add" in kernel_names
        assert "matrix_multiply" in kernel_names


class TestAdvancedCppMethodFeatures:
    """Test advanced C++ method features."""
    
    def test_complex_cpp_methods(self):
        """Test complex C++ methods with various parameter types."""
        content = """
        node test_node {
            method complex_processing {
                input: int data_size
                input: float* input_data (data_size)
                input: std::vector<float> weights (weights.size())
                input: std::string config_file
                input: bool enable_optimization
                output: float* result (data_size)
                output: std::vector<int> indices (data_size)
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
                input: std::vector<int> input_vector (input_vector.size())
                output: std::vector<float> output_vector (input_vector.size())
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
        assert "std::vector<float>" in input_types
        assert "std::string" in input_types
        assert "bool" in input_types
        
        # Check output parameters
        output_types = [p.param_type for p in complex_method.outputs]
        assert "float*" in output_types
        assert "std::vector<int>" in output_types
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
                param in float* input_image (width, height, channels)
                param in float* kernel (kernel_size, kernel_size)
                param out float* output_image (width, height, channels)
                param in int width
                param in int height
                param in int channels
                param in int kernel_size
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
                param in float* input_data (N)
                param out float* result (1)
                param in int N
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
        """Test detection of duplicate parameter names."""
        content = """
        node test_node {
            parameter test_param: 42
            parameter test_param: 43  // Duplicate name
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "Duplicate parameter name" in str(exc_info.value)
    
    def test_duplicate_timer_names(self):
        """Test detection of duplicate timer names."""
        content = """
        node test_node {
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
        node test_node {
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
        node test_node {
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
        node test_node {
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
        node test_node {
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
        node test_node {
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
        node test_node {
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
        node test_node {
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
                param in float* input (N)
                param out float* output (N)
                code: {
                    printf('hello');
                }
            }
            kernel test_kernel {
                param in int* input (N)
                param out int* output (N)
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
        node test_node {
            publisher /test_topic : "std_msgs/String" {
                qos {
                    reliability: invalid_value
                    durability: also_invalid
                }
            }
        }
        """
    
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
    
        assert "Invalid reliability setting" in str(exc_info.value)
        assert "Invalid durability setting" in str(exc_info.value)
    
    def test_invalid_timer_period(self):
        """Test detection of invalid timer period."""
        content = """
        node test_node {
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
                param in float* input (N)
                param out float* output (N)
                block_size: (0, 1, 1)  // Invalid zero block size
                code: {
                    printf('hello');
                }
            }
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        assert "block size dimension 0 cannot be zero" in str(exc_info.value)


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
        """Test parsing of very large numeric values."""
        content = """
        node large_values_node {
            parameter large_int: 999999999999999999
            parameter large_float: 3.1415926535897932384626433832795028841971693993751058209749445923078164062862089986280348253421170679
            parameter small_float: 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 3
    
    def test_unicode_strings(self):
        """Test parsing of Unicode strings."""
        content = """
        node unicode_node {
            parameter japanese: "こんにちは世界"
            parameter chinese: "你好世界"
            parameter emoji: "🚀🤖💻"
            parameter special_chars: "áéíóúñü¿¡"
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 4
    
    def test_nested_arrays_and_dicts(self):
        """Test parsing of deeply nested arrays and dictionaries."""
        content = """
        node nested_node {
            parameter nested_array: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            parameter nested_dict: {
                level1: {
                    level2: {
                        level3: {
                            value: 42
                            array: [1, 2, 3, 4, 5]
                        }
                    }
                }
            }
            parameter mixed: [{
                name: "item1"
                values: [1, 2, 3]
                config: {
                    enabled: true
                    timeout: 5.0
                }
            }, {
                name: "item2"
                values: [4, 5, 6]
                config: {
                    enabled: false
                    timeout: 10.0
                }
            }]
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 3
    
    def test_method_with_no_parameters(self):
        """Test parsing of a method with no input/output parameters."""
        content = """
        node test_node {
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
        """Test that valid nodes are still parsed even if one is invalid."""
        content = """
        node valid_node1 {
            parameter test: 42
        }
        
        node invalid_node {
            parameter test: 42
            parameter test: 43  // Duplicate - should cause error
        }
        
        node valid_node2 {
            parameter test: 42
        }
        """
        
        with pytest.raises(SemanticError) as exc_info:
            parse_robodsl(content)
        
        # The error should be caught, but we should still have parsed the valid nodes
        assert "Duplicate parameter name" in str(exc_info.value)
    
    def test_invalid_syntax_in_comments(self):
        """Test that invalid syntax in comments doesn't cause parsing errors."""
        content = """
        node test_node {
            // This comment has invalid syntax: node invalid { parameter x: y }
            parameter valid_param: 42
            /* Another comment with invalid syntax:
               publisher /topic "invalid"
               subscriber /topic "also invalid"
            */
        }
        """
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 1
        assert node.content.parameters[0].name == "valid_param"


class TestPerformance:
    """Test performance with large configurations."""
    
    def test_large_number_of_parameters(self):
        """Test parsing with a large number of parameters."""
        content = "node large_node {\n"
        for i in range(1000):
            content += f"    parameter param_{i}: {i}\n"
        content += "}\n"
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.parameters) == 1000
    
    def test_large_number_of_publishers(self):
        """Test parsing with a large number of publishers."""
        content = "node large_node {\n"
        for i in range(100):
            content += f'    publisher /topic_{i} : "std_msgs/String"\n'
        content += "}\n"
        
        ast = parse_robodsl(content)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.publishers) == 100
    
    def test_large_code_blocks(self):
        """Test parsing with large code blocks."""
        large_code = "int x = 0;\\n" * 1000
        content = f"""
        node test_node {{
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