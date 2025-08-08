#include "robodsl_project/obstacle_avoidance_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

obstacle_avoidanceNode::obstacle_avoidanceNode() 
    : Node("obstacle_avoidance_node", "/navigation_pipeline/obstacle_avoidance") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing obstacle_avoidance stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "obstacle_avoidance stage node initialized successfully");
}

obstacle_avoidanceNode::~obstacle_avoidanceNode() {
    
}

void obstacle_avoidanceNode::initialize_ros_components() {
    // Initialize subscribers
    velocity_command,sensor_data_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/navigation_pipeline/obstacle_avoidance/velocity_command,sensor_data", 10,
        std::bind(&obstacle_avoidanceNode::velocity_command,sensor_data_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    safe_velocity_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/navigation_pipeline/obstacle_avoidance/safe_velocity", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&obstacle_avoidanceNode::timer_callback, this)
    );
}



// Callback methods
void obstacle_avoidanceNode::velocity_command,sensor_data_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received velocity_command,sensor_data: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    safe_velocity_pub_->publish(output_msg);
}

void obstacle_avoidanceNode::timer_callback() {
    // Periodic processing tasks
    avoid_obstacles();
    
}

// Processing methods
void obstacle_avoidanceNode::avoid_obstacles() {
    RCLCPP_DEBUG(this->get_logger(), "Executing avoid_obstacles");
    // TODO: Implement avoid_obstacles logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::obstacle_avoidanceNode) 