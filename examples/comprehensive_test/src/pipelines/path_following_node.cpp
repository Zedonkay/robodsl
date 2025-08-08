#include "robodsl_project/path_following_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

path_followingNode::path_followingNode() 
    : Node("path_following_node", "/navigation_pipeline/path_following") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing path_following stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "path_following stage node initialized successfully");
}

path_followingNode::~path_followingNode() {
    
}

void path_followingNode::initialize_ros_components() {
    // Initialize subscribers
    planned_path,current_pose_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/navigation_pipeline/path_following/planned_path,current_pose", 10,
        std::bind(&path_followingNode::planned_path,current_pose_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    velocity_command_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/navigation_pipeline/path_following/velocity_command", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&path_followingNode::timer_callback, this)
    );
}



// Callback methods
void path_followingNode::planned_path,current_pose_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received planned_path,current_pose: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    velocity_command_pub_->publish(output_msg);
}

void path_followingNode::timer_callback() {
    // Periodic processing tasks
    follow_path();
    
}

// Processing methods
void path_followingNode::follow_path() {
    RCLCPP_DEBUG(this->get_logger(), "Executing follow_path");
    // TODO: Implement follow_path logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::path_followingNode) 