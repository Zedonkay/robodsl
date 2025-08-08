#include "robodsl_project/preprocessing_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

preprocessingNode::preprocessingNode() 
    : Node("preprocessing_node", "/perception_pipeline/preprocessing") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing preprocessing stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "preprocessing stage node initialized successfully");
}

preprocessingNode::~preprocessingNode() {
    
}

void preprocessingNode::initialize_ros_components() {
    // Initialize subscribers
    raw_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/perception_pipeline/preprocessing/raw_image", 10,
        std::bind(&preprocessingNode::raw_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    preprocessed_image_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/perception_pipeline/preprocessing/preprocessed_image", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&preprocessingNode::timer_callback, this)
    );
}



// Callback methods
void preprocessingNode::raw_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received raw_image: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    preprocessed_image_pub_->publish(output_msg);
}

void preprocessingNode::timer_callback() {
    // Periodic processing tasks
    preprocess_image();
    
}

// Processing methods
void preprocessingNode::preprocess_image() {
    RCLCPP_DEBUG(this->get_logger(), "Executing preprocess_image");
    // TODO: Implement preprocess_image logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::preprocessingNode) 