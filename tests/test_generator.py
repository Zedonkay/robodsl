"""Tests for the RoboDSL code generator."""

import os
import tempfile
from pathlib import Path
import pytest

from robodsl.parser import (
    parse_robodsl, 
    RoboDSLConfig, 
    NodeConfig, 
    CudaKernelConfig
)
from robodsl.generator import CodeGenerator


def test_generate_node_header():
    """Test generating a C++ header file for a simple node."""
    # Create a simple configuration
    config = RoboDSLConfig()
    node = NodeConfig("test_node")
    node.publishers = [{"topic": "/odom", "msg_type": "nav_msgs/msg/Odometry"}]
    node.subscribers = [{"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist"}]
    node.services = [{"service": "/get_status", "srv_type": "std_srvs/srv/Trigger"}]
    node.parameters = {"max_speed": "1.5", "use_gpu": "true"}
    config.nodes.append(node)
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = CodeGenerator(config, output_dir=tmpdir)
        generator._generate_node_header(node)  # Call the internal method directly for testing
        
        # Check that the header file was created
        header_path = Path(tmpdir) / 'include' / 'robodsl' / 'test_node_node.hpp'
        assert header_path.exists(), f"Header file not found at {header_path}"
        
        # Check the content of the header file
        content = header_path.read_text()
        assert '#ifndef TEST_NODE_NODE_HPP_' in content
        assert 'class TestNode : public rclcpp::Node' in content
        assert 'rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;' in content
        assert 'rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub_;' in content
        assert 'rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr get_status_srv_;' in content
        assert 'void cmd_vel_callback(const geometry_msgs::msg::Twist::SharedPtr msg) const;' in content
        assert 'void get_status_callback(' in content
        assert 'const std::shared_ptr<std_srvs::srv::Trigger::Request> request,' in content
        assert 'std::shared_ptr<std_srvs::srv::Trigger::Response> response);' in content
        assert 'double max_speed_;' in content
        assert 'bool use_gpu_;' in content


def test_generate_node_with_cuda_kernel():
    """Test generating a node with CUDA kernel support."""
    # Create a configuration with a node and a CUDA kernel
    config = RoboDSLConfig()
    
    # Add a node
    node = NodeConfig("image_processor")
    node.publishers = [{"topic": "/processed_image", "msg_type": "sensor_msgs/msg/Image"}]
    node.subscribers = [{"topic": "/camera/image_raw", "msg_type": "sensor_msgs/msg/Image"}]
    config.nodes.append(node)
    
    # Add a CUDA kernel
    kernel = CudaKernelConfig("image_processor_filter")
    kernel.inputs = [{"type": "GpuImage", "width": "1920", "height": "1080"}]
    kernel.outputs = [{"type": "GpuImage"}]
    kernel.block_size = (16, 16, 1)
    config.cuda_kernels.append(kernel)
    
    # Create a temporary directory for output
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = CodeGenerator(config, output_dir=tmpdir)
        generator._generate_node_header(node)  # Call the internal method directly for testing
        
        # Check that the header file was created
        header_path = Path(tmpdir) / 'include' / 'robodsl' / 'image_processor_node.hpp'
        assert header_path.exists(), f"Header file not found at {header_path}"
        
        # Check the content of the header file
        content = header_path.read_text()
        assert 'class ImageProcessor : public rclcpp::Node' in content
        assert 'class Image_processor_filterKernel' in content
        assert 'void configure(const std::map<std::string, std::string>& params);' in content
        assert 'void cleanup();' in content
        assert 'void process();' in content


def test_generate_empty_config():
    """Test generating files from an empty configuration."""
    config = RoboDSLConfig()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = CodeGenerator(config, output_dir=tmpdir)
        generator.generate()
        
        # Verify that the output directory structure was created
        assert (Path(tmpdir) / 'include').exists()
        assert (Path(tmpdir) / 'src').exists()
        assert (Path(tmpdir) / 'CMakeLists.txt').exists()


def test_generate_cmakelists():
    """Test generating CMakeLists.txt with nodes and CUDA kernels."""
    # Create a configuration with nodes and CUDA kernels
    config = RoboDSLConfig()
    config.project_name = "test_project"
    
    # Add a node with publishers, subscribers, and services
    node1 = NodeConfig("test_node")
    node1.publishers = [{"topic": "/odom", "msg_type": "nav_msgs/msg/Odometry"}]
    node1.subscribers = [{"topic": "/cmd_vel", "msg_type": "geometry_msgs/msg/Twist"}]
    node1.services = [{"service": "/get_status", "srv_type": "std_srvs/srv/Trigger"}]
    node1.parameters = {"max_speed": "1.5", "use_gpu": "true"}
    config.nodes.append(node1)
    
    # Add a CUDA kernel
    kernel = CudaKernelConfig("test_kernel")
    kernel.inputs = [{"type": "float*"}]
    kernel.outputs = [{"type": "float*"}]
    kernel.block_size = [16, 1, 1]
    config.cuda_kernels.append(kernel)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        generator = CodeGenerator(config, output_dir=tmpdir)
        
        # Call the internal method directly for testing
        generator._generate_cmakelists()
        
        # Check that CMakeLists.txt was created
        cmake_path = Path(tmpdir) / 'CMakeLists.txt'
        assert cmake_path.exists(), f"CMakeLists.txt not found at {cmake_path}"
        
        # Check the content of CMakeLists.txt
        content = cmake_path.read_text()
        
        # Check project name
        assert f"project({config.project_name})" in content
        
        # Check CUDA configuration
        assert "find_package(CUDA REQUIRED)" in content
        assert "set(CUDA_NVCC_FLAGS" in content
        
        # Check CUDA kernel target
        assert "cuda_add_library(test_kernel_kernel" in content
        
        # Check node target
        assert "add_library(test_node_node" in content
        assert "add_executable(test_node_node_exe" in content
        
        # Check installation rules
        assert "ament_package()" in content


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
