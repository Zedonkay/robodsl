#include "robodsl_project/path_planning_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

path_planningNode::path_planningNode() 
    : Node("path_planning_node", "/navigation_pipeline/path_planning") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing path_planning stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "path_planning stage node initialized successfully");
}

path_planningNode::~path_planningNode() {
    
}

void path_planningNode::initialize_ros_components() {
    // Initialize subscribers
    goal_pose,current_pose,map_data_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/navigation_pipeline/path_planning/goal_pose,current_pose,map_data", 10,
        std::bind(&path_planningNode::goal_pose,current_pose,map_data_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    planned_path_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/navigation_pipeline/path_planning/planned_path", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&path_planningNode::timer_callback, this)
    );
}



// Callback methods
void path_planningNode::goal_pose,current_pose,map_data_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received goal_pose,current_pose,map_data: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    planned_path_pub_->publish(output_msg);
}

void path_planningNode::timer_callback() {
    // Periodic processing tasks
    plan_path();
    
}

// Processing methods
void path_planningNode::plan_path() {
    RCLCPP_DEBUG(this->get_logger(), "Executing plan_path");
    // TODO: Implement plan_path logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::path_planningNode) 