#include "robodsl_project/fusion_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

fusionNode::fusionNode() 
    : Node("fusion_node", "/perception_pipeline/fusion") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing fusion stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "fusion stage node initialized successfully");
}

fusionNode::~fusionNode() {
    
}

void fusionNode::initialize_ros_components() {
    // Initialize subscribers
    classification_result,detection_result,segmentation_result_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/perception_pipeline/fusion/classification_result,detection_result,segmentation_result", 10,
        std::bind(&fusionNode::classification_result,detection_result,segmentation_result_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    final_result_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/perception_pipeline/fusion/final_result", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&fusionNode::timer_callback, this)
    );
}



// Callback methods
void fusionNode::classification_result,detection_result,segmentation_result_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received classification_result,detection_result,segmentation_result: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    final_result_pub_->publish(output_msg);
}

void fusionNode::timer_callback() {
    // Periodic processing tasks
    fuse_results();
    
}

// Processing methods
void fusionNode::fuse_results() {
    RCLCPP_DEBUG(this->get_logger(), "Executing fuse_results");
    // TODO: Implement fuse_results logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::fusionNode) 