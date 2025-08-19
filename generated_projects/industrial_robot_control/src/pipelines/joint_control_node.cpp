#include "robodsl_project/joint_control_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

joint_controlNode::joint_controlNode() 
    : Node("joint_control_node", "/industrial_robot_pipeline/joint_control") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing joint_control stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "joint_control stage node initialized successfully");
}

joint_controlNode::~joint_controlNode() {
    
}

void joint_controlNode::initialize_ros_components() {
    // Initialize subscribers
    optimized_trajectory_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/joint_control/optimized_trajectory", 10,
        std::bind(&joint_controlNode::optimized_trajectory_callback, this, std::placeholders::_1)
    );
    force_correction_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/joint_control/force_correction", 10,
        std::bind(&joint_controlNode::force_correction_callback, this, std::placeholders::_1)
    );
    collision_status_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/joint_control/collision_status", 10,
        std::bind(&joint_controlNode::collision_status_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    joint_commands_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/industrial_robot_pipeline/joint_control/joint_commands", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&joint_controlNode::timer_callback, this)
    );
}



// Callback methods
void joint_controlNode::optimized_trajectory_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received optimized_trajectory: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    joint_commands_pub_->publish(output_msg);
}
void joint_controlNode::force_correction_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received force_correction: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    joint_commands_pub_->publish(output_msg);
}
void joint_controlNode::collision_status_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received collision_status: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    joint_commands_pub_->publish(output_msg);
}

void joint_controlNode::timer_callback() {
    // Periodic processing tasks
    compute_joint_commands();
    
}

// Processing methods
void joint_controlNode::compute_joint_commands() {
    RCLCPP_DEBUG(this->get_logger(), "Executing compute_joint_commands");
    // TODO: Implement compute_joint_commands logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::joint_controlNode) 