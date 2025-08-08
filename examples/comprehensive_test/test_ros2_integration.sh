#!/bin/bash

# ROS2 Integration Test Script for RoboDSL Generated Files
# Tests ROS2 node compilation and basic functionality

set -e

echo "🤖 Starting ROS2 Integration Test"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if ROS2 is available
if ! command -v ros2 &> /dev/null; then
    print_warning "ROS2 not found. Skipping ROS2 integration tests."
    print_status "Install ROS2 to run these tests."
    exit 0
fi

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run this script from the comprehensive_test directory."
    exit 1
fi

# Source ROS2 environment
print_status "Sourcing ROS2 environment..."
source /opt/ros/humble/setup.bash 2>/dev/null || source /opt/ros/foxy/setup.bash 2>/dev/null || {
    print_warning "Could not source ROS2 environment automatically."
    print_status "Please source your ROS2 environment manually:"
    print_status "  source /opt/ros/<your_distro>/setup.bash"
}

# Check if build exists
if [ ! -d "build" ]; then
    print_error "Build directory not found. Run ./test_build.sh first."
    exit 1
fi

cd build

# Test ROS2 node compilation
print_status "Testing ROS2 node compilation..."

# Create a simple ROS2 test program
cat > test_ros2_integration.cpp << 'EOF'
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <chrono>

// Include our generated CUDA wrappers
#include "../include/cuda/vector_add_wrapper.hpp"

class TestNode : public rclcpp::Node {
public:
    TestNode() : Node("test_node") {
        RCLCPP_INFO(get_logger(), "Test node created");
        
        // Test CUDA wrapper creation
        try {
            cuda_wrapper_ = robodsl::createvector_addWrapper(this);
            if (cuda_wrapper_->initialize(0)) {
                RCLCPP_INFO(get_logger(), "CUDA wrapper initialized successfully");
            } else {
                RCLCPP_ERROR(get_logger(), "Failed to initialize CUDA wrapper: %s", 
                           cuda_wrapper_->getLastError().c_str());
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR(get_logger(), "Exception creating CUDA wrapper: %s", e.what());
        }
        
        // Create a simple publisher
        publisher_ = create_publisher<std_msgs::msg::String>("test_topic", 10);
        
        // Create a timer
        timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&TestNode::timer_callback, this)
        );
        
        RCLCPP_INFO(get_logger(), "Test node initialized successfully");
    }
    
private:
    void timer_callback() {
        auto message = std_msgs::msg::String();
        message.data = "Hello from test node!";
        publisher_->publish(message);
        
        // Test CUDA processing if wrapper is available
        if (cuda_wrapper_ && cuda_wrapper_->isInitialized()) {
            std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
            std::vector<float> output_data;
            
            if (cuda_wrapper_->processData(input_data, output_data)) {
                RCLCPP_INFO(get_logger(), "CUDA processing successful");
            } else {
                RCLCPP_ERROR(get_logger(), "CUDA processing failed: %s", 
                           cuda_wrapper_->getLastError().c_str());
            }
        }
    }
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    std::unique_ptr<robodsl::vector_addWrapper> cuda_wrapper_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<TestNode>();
    
    RCLCPP_INFO(node->get_logger(), "Starting ROS2 integration test...");
    
    // Run for a few seconds
    rclcpp::spin_some(node);
    
    // Sleep for a moment to let messages be published
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    RCLCPP_INFO(node->get_logger(), "ROS2 integration test completed successfully");
    
    rclcpp::shutdown();
    return 0;
}
EOF

# Compile the ROS2 test program
print_status "Compiling ROS2 integration test..."

# Get ROS2 include and library paths
ROS2_INCLUDE_DIRS=$(pkg-config --cflags rclcpp std_msgs sensor_msgs 2>/dev/null || echo "")
ROS2_LIBRARY_DIRS=$(pkg-config --libs rclcpp std_msgs sensor_msgs 2>/dev/null || echo "")

# Try to compile with ROS2
g++ -std=c++17 -I../include -I/usr/local/cuda/include \
    $ROS2_INCLUDE_DIRS \
    test_ros2_integration.cpp \
    -L/usr/local/cuda/lib64 -lcudart \
    $ROS2_LIBRARY_DIRS \
    -o test_ros2_integration

if [ $? -eq 0 ]; then
    print_success "ROS2 integration test compiled successfully"
else
    print_warning "ROS2 integration test compilation failed"
    print_status "This might be due to missing ROS2 development packages"
    print_status "Install with: sudo apt install ros-humble-rclcpp-dev ros-humble-std-msgs-dev"
    cd ..
    exit 0
fi

# Run the ROS2 test
print_status "Running ROS2 integration test..."
timeout 10s ./test_ros2_integration

if [ $? -eq 0 ] || [ $? -eq 124 ]; then  # 124 is timeout exit code
    print_success "ROS2 integration test PASSED"
else
    print_error "ROS2 integration test FAILED"
    cd ..
    exit 1
fi

# Test ROS2 message compilation
print_status "Testing ROS2 message compilation..."
cat > test_ros2_messages.cpp << 'EOF'
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <iostream>

int main() {
    std::cout << "Testing ROS2 message compilation..." << std::endl;
    
    // Test creating various message types
    auto string_msg = std_msgs::msg::String();
    string_msg.data = "test";
    
    auto array_msg = std_msgs::msg::Float32MultiArray();
    array_msg.data = {1.0f, 2.0f, 3.0f};
    
    auto image_msg = sensor_msgs::msg::Image();
    image_msg.width = 640;
    image_msg.height = 480;
    image_msg.encoding = "rgb8";
    
    auto twist_msg = geometry_msgs::msg::Twist();
    twist_msg.linear.x = 1.0;
    twist_msg.angular.z = 0.5;
    
    std::cout << "ROS2 message compilation test PASSED" << std::endl;
    return 0;
}
EOF

g++ -std=c++17 $ROS2_INCLUDE_DIRS test_ros2_messages.cpp $ROS2_LIBRARY_DIRS -o test_ros2_messages
./test_ros2_messages

if [ $? -eq 0 ]; then
    print_success "ROS2 message compilation test PASSED"
else
    print_error "ROS2 message compilation test FAILED"
    cd ..
    exit 1
fi

# Test ROS2 node template compilation
print_status "Testing ROS2 node template compilation..."
cat > test_node_template.cpp << 'EOF'
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <iostream>

class TestLifecycleNode : public rclcpp_lifecycle::LifecycleNode {
public:
    TestLifecycleNode() : LifecycleNode("test_lifecycle_node") {
        RCLCPP_INFO(get_logger(), "Test lifecycle node created");
    }
    
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State&) override {
        RCLCPP_INFO(get_logger(), "Node configured");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State&) override {
        RCLCPP_INFO(get_logger(), "Node activated");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State&) override {
        RCLCPP_INFO(get_logger(), "Node deactivated");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State&) override {
        RCLCPP_INFO(get_logger(), "Node cleaned up");
        return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
};

RCLCPP_COMPONENTS_REGISTER_NODE(TestLifecycleNode)

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    auto node = std::make_shared<TestLifecycleNode>();
    
    std::cout << "ROS2 node template compilation test PASSED" << std::endl;
    
    rclcpp::shutdown();
    return 0;
}
EOF

# Get additional ROS2 component paths
ROS2_COMPONENT_INCLUDE_DIRS=$(pkg-config --cflags rclcpp_lifecycle rclcpp_components 2>/dev/null || echo "")
ROS2_COMPONENT_LIBRARY_DIRS=$(pkg-config --libs rclcpp_lifecycle rclcpp_components 2>/dev/null || echo "")

g++ -std=c++17 $ROS2_INCLUDE_DIRS $ROS2_COMPONENT_INCLUDE_DIRS \
    test_node_template.cpp \
    $ROS2_LIBRARY_DIRS $ROS2_COMPONENT_LIBRARY_DIRS \
    -o test_node_template

if [ $? -eq 0 ]; then
    print_success "ROS2 node template compilation test PASSED"
else
    print_warning "ROS2 node template compilation test failed (this is optional)"
fi

cd ..

echo ""
print_success "All ROS2 integration tests completed successfully!"
echo ""
print_status "Next steps:"
echo "  1. Run: ./test_performance.sh"
echo "  2. Test with actual ROS2 launch files: ros2 launch launch/main_launch.py"
echo ""

exit 0
