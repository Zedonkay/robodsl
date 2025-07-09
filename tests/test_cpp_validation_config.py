"""C++ Code Validation Test Configuration.

This file defines comprehensive test cases and validation scenarios for ensuring
that generated C++ code is correct, efficient, and follows best practices.
"""

import pytest
from conftest import skip_if_no_ros2, skip_if_no_cuda
from typing import List, Dict, Any

# Test case definitions for comprehensive C++ validation

BASIC_NODE_TEST_CASES = [
    {
        "name": "simple_node",
        "source": """
        node simple_node {
            parameter int max_speed = 10
            publisher /test: std_msgs/String
            subscriber /input: std_msgs/String
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 1,  # Allow some performance suggestions
            "practices": 1     # Allow some practice suggestions
        }
    },
    {
        "name": "node_with_services",
        "source": """
        node service_node {
            parameter float threshold = 0.5
            service /process: std_srvs/Trigger
            publisher /result: std_msgs/Float32
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "lifecycle_node",
        "source": """
        node lifecycle_node: lifecycle {
            parameter int max_iterations = 100
            publisher /status: std_msgs/String
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 1,
            "practices": 1
        }
    }
]

CUDA_KERNEL_TEST_CASES = [
    {
        "name": "basic_kernel",
        "source": """
        cuda_kernel basic_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 1,  # Allow some memory management suggestions
            "cuda": 1,    # Allow some CUDA best practice suggestions
            "performance": 1
        }
    },
    {
        "name": "optimized_kernel",
        "source": """
        cuda_kernel optimized_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            shared_memory: 1024
            use_thrust: true
            input: float data[1000]
            output: float result[1000]
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 1,
            "cuda": 1,
            "performance": 1
        }
    },
    {
        "name": "multi_param_kernel",
        "source": """
        cuda_kernel multi_param_kernel {
            block_size: (32, 32, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
            parameter: float threshold
            parameter: int iterations
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 1,
            "cuda": 1,
            "performance": 1
        }
    }
]

ADVANCED_CPP_TEST_CASES = [
    {
        "name": "templates",
        "source": """
        template<typename T> struct Vector {
            T* data;
            size_t size;
        }
        
        template<typename T> T sqr(T x) {
            return x * x;
        }
        
        node template_node {
            parameter int size = 100
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 1,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "operators",
        "source": """
        def operator<<(stream: std::ostream&, vec: std::vector<int>&) -> std::ostream& {
            stream << "Vector";
            return stream;
        }
        
        node operator_node {
            parameter int value = 42
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "constexpr",
        "source": """
        global PI: constexpr float = 3.14159;
        global MAX_SIZE: constexpr int = 1000;
        
        node constexpr_node {
            parameter int count = 10
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    }
]

EFFICIENCY_TEST_CASES = [
    {
        "name": "memory_efficient",
        "source": """
        node memory_efficient_node {
            parameter int buffer_size = 1000
            
            def efficient_process(input: const std::vector<float>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(input.size());  // Efficient allocation
                
                for (const auto& value : input) {  // Range-based loop
                    result.push_back(value);
                }
                return result;
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    },
    {
        "name": "smart_pointers",
        "source": """
        node smart_pointer_node {
            def smart_pointer_function() -> std::unique_ptr<std::vector<float>> {
                auto result = std::make_unique<std::vector<float>>();
                result->reserve(1000);
                return result;  // RAII ensures cleanup
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    },
    {
        "name": "move_semantics",
        "source": """
        node move_semantics_node {
            def move_function(input: std::vector<float>&&) -> std::vector<float> {
                return std::move(input);  // Use move semantics
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    }
]

CORRECTNESS_TEST_CASES = [
    {
        "name": "type_safe",
        "source": """
        node type_safe_node {
            def type_safe_function(input: const std::vector<float>&) -> std::vector<float> {
                if (input.empty()) {
                    return {};
                }
                
                std::vector<float> result;
                result.reserve(input.size());
                
                for (const auto& value : input) {
                    if (value > 0.0f) {
                        result.push_back(value);
                    }
                }
                return result;
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    },
    {
        "name": "exception_safe",
        "source": """
        node exception_safe_node {
            def exception_safe_function() -> std::unique_ptr<std::vector<float>> {
                try {
                    auto result = std::make_unique<std::vector<float>>();
                    result->reserve(1000);
                    return result;
                } catch (const std::exception& e) {
                    return nullptr;
                }
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    },
    {
        "name": "error_handling",
        "source": """
        node error_handling_node {
            def error_handling_function(input: const std::vector<float>&) -> bool {
                if (input.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Empty input");
                    return false;
                }
                
                if (input.size() > 1000) {
                    RCLCPP_ERROR(this->get_logger(), "Input too large");
                    return false;
                }
                
                return true;
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 0,
            "practices": 0
        }
    }
]

COMPREHENSIVE_TEST_CASES = [
    {
        "name": "full_feature_node",
        "source": """
        include <iostream>
        include <vector>
        include <memory>
        
        template<typename T> struct SafeContainer {
            std::unique_ptr<T[]> data;
            size_t size;
            
            SafeContainer(size_t n) : size(n) {
                data = std::make_unique<T[]>(n);
            }
        }
        
        global MAX_SIZE: constexpr int = 1000;
        
        def operator<<(stream: std::ostream&, container: SafeContainer<int>&) -> std::ostream& {
            stream << "SafeContainer[" << container.size << "]";
            return stream;
        }
        
        cuda_kernel efficient_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        
        node comprehensive_node {
            parameter int max_speed = 10
            parameter float safety_distance = 1.5
            
            publisher /robot/position: geometry_msgs/Point
            subscriber /robot/command: geometry_msgs/Twist
            service /robot/status: std_srvs/Trigger
            
            def efficient_process(input: const std::vector<float>&) -> std::vector<float> {
                if (input.empty()) {
                    RCLCPP_WARN(this->get_logger(), "Empty input");
                    return {};
                }
                
                std::vector<float> result;
                result.reserve(input.size());
                
                for (const auto& value : input) {
                    if (value > safety_distance) {
                        result.push_back(value);
                    }
                }
                
                RCLCPP_INFO(this->get_logger(), "Processed %zu values", result.size());
                return result;
            }
            
            def __init__() {
                // Constructor with proper initialization
            }
            
            def __del__() {
                // Destructor with proper cleanup
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 1,  # Allow some suggestions
            "performance": 1,
            "practices": 1,
            "cuda": 1
        }
    }
]

EDGE_CASE_TEST_CASES = [
    {
        "name": "zero_values",
        "source": """
        node edge_case_node {
            parameter int zero_param = 0
            parameter float negative_param = -1.0
            parameter std::string empty_string = ""
            
            def edge_case_function() -> int {
                return 0;
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 1,
            "practices": 1
        }
    },
    {
        "name": "large_values",
        "source": """
        node large_value_node {
            parameter int large_int = 2147483647
            parameter float large_float = 3.402823e+38
            
            def large_value_function() -> float {
                return large_float;
            }
        }
        """,
        "expected_issues": {
            "syntax": 0,
            "memory": 0,
            "performance": 1,
            "practices": 1
        }
    }
]

# Test case generators for parameterized testing

def generate_basic_node_test_cases():
    """Generate test cases for basic node validation."""
    for test_case in BASIC_NODE_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_cuda_kernel_test_cases():
    """Generate test cases for CUDA kernel validation."""
    for test_case in CUDA_KERNEL_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_advanced_cpp_test_cases():
    """Generate test cases for advanced C++ features validation."""
    for test_case in ADVANCED_CPP_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_efficiency_test_cases():
    """Generate test cases for efficiency validation."""
    for test_case in EFFICIENCY_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_correctness_test_cases():
    """Generate test cases for correctness validation."""
    for test_case in CORRECTNESS_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_comprehensive_test_cases():
    """Generate test cases for comprehensive validation."""
    for test_case in COMPREHENSIVE_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

def generate_edge_case_test_cases():
    """Generate test cases for edge case validation."""
    for test_case in EDGE_CASE_TEST_CASES:
        yield pytest.param(
            test_case["source"],
            test_case["expected_issues"],
            id=test_case["name"]
        )

# Validation configuration

VALIDATION_CONFIG = {
    "syntax_tolerance": 0,      # No syntax errors allowed
    "memory_tolerance": 1,      # Allow 1 memory suggestion
    "performance_tolerance": 2, # Allow 2 performance suggestions
    "practice_tolerance": 2,    # Allow 2 practice suggestions
    "cuda_tolerance": 1,        # Allow 1 CUDA suggestion
    
    "compiler_flags": [
        '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2',
        '-fno-exceptions', '-fno-rtti', '-DNDEBUG'
    ],
    
    "cuda_flags": [
        '-std=c++17', '-Wall', '-Wextra', '-O2',
        '-arch=sm_60', '-DNDEBUG'
    ],
    
    "timeout_seconds": 30,
    "max_file_size_kb": 100
}

# Test categories for organization

TEST_CATEGORIES = {
    "basic": {
        "description": "Basic node generation tests",
        "test_cases": BASIC_NODE_TEST_CASES,
        "priority": "high"
    },
    "cuda": {
        "description": "CUDA kernel generation tests",
        "test_cases": CUDA_KERNEL_TEST_CASES,
        "priority": "high"
    },
    "advanced": {
        "description": "Advanced C++ features tests",
        "test_cases": ADVANCED_CPP_TEST_CASES,
        "priority": "medium"
    },
    "efficiency": {
        "description": "Code efficiency tests",
        "test_cases": EFFICIENCY_TEST_CASES,
        "priority": "medium"
    },
    "correctness": {
        "description": "Code correctness tests",
        "test_cases": CORRECTNESS_TEST_CASES,
        "priority": "high"
    },
    "comprehensive": {
        "description": "Comprehensive feature tests",
        "test_cases": COMPREHENSIVE_TEST_CASES,
        "priority": "medium"
    },
    "edge_cases": {
        "description": "Edge case tests",
        "test_cases": EDGE_CASE_TEST_CASES,
        "priority": "low"
    }
} 