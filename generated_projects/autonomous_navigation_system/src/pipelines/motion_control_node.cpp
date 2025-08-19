#include "robodsl_project/motion_control_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

motion_controlNode::motion_controlNode() 
    : Node("motion_control_node", "/autonomous_navigation_pipeline/motion_control") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing motion_control stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "motion_control stage node initialized successfully");
}

motion_controlNode::~motion_controlNode() {
    
}

void motion_controlNode::initialize_ros_components() {
    // Initialize subscribers
    planned_path_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/motion_control/planned_path", 10,
        std::bind(&motion_controlNode::planned_path_callback, this, std::placeholders::_1)
    );
    robot_pose_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/motion_control/robot_pose", 10,
        std::bind(&motion_controlNode::robot_pose_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    velocity_command_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/motion_control/velocity_command", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&motion_controlNode::timer_callback, this)
    );
}



// Callback methods
void motion_controlNode::planned_path_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received planned_path: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    velocity_command_pub_->publish(output_msg);
}
void motion_controlNode::robot_pose_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received robot_pose: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    velocity_command_pub_->publish(output_msg);
}

void motion_controlNode::timer_callback() {
    // Periodic processing tasks
    compute_velocity();
    
}

// Processing methods
void motion_controlNode::compute_velocity() {
    RCLCPP_DEBUG(this->get_logger(), "Executing compute_velocity");
    // TODO: Implement compute_velocity logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::motion_controlNode) 