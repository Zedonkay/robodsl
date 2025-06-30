#include "robodsl_project/image_preprocessing_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

image_preprocessingNode::image_preprocessingNode() 
    : Node("image_preprocessing_node", "/vision_pipeline/image_preprocessing") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing image_preprocessing stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "image_preprocessing stage node initialized successfully");
}

image_preprocessingNode::~image_preprocessingNode() {
    
}

void image_preprocessingNode::initialize_ros_components() {
    // Initialize subscribers
    /camera/image_raw_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/vision_pipeline/image_preprocessing//camera/image_raw", 10,
        std::bind(&image_preprocessingNode::/camera/image_raw_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    /pipeline/preprocessed_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/vision_pipeline/image_preprocessing//pipeline/preprocessed", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&image_preprocessingNode::timer_callback, this)
    );
}



// Callback methods
void image_preprocessingNode::/camera/image_raw_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received /camera/image_raw: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    /pipeline/preprocessed_pub_->publish(output_msg);
}

void image_preprocessingNode::timer_callback() {
    // Periodic processing tasks
    preprocess_image();
    
}

// Processing methods
void image_preprocessingNode::preprocess_image() {
    RCLCPP_DEBUG(this->get_logger(), "Executing preprocess_image");
    // TODO: Implement preprocess_image logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::image_preprocessingNode) 