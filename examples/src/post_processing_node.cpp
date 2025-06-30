#include "robodsl_project/post_processing_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

post_processingNode::post_processingNode() 
    : Node("post_processing_node", "/vision_pipeline/post_processing") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing post_processing stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    
    RCLCPP_INFO(this->get_logger(), "post_processing stage node initialized successfully");
}

post_processingNode::~post_processingNode() {
    
}

void post_processingNode::initialize_ros_components() {
    // Initialize subscribers
    /pipeline/detections_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/vision_pipeline/post_processing//pipeline/detections", 10,
        std::bind(&post_processingNode::/pipeline/detections_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    /pipeline/final_results_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/vision_pipeline/post_processing//pipeline/final_results", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&post_processingNode::timer_callback, this)
    );
}



// Callback methods
void post_processingNode::/pipeline/detections_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received /pipeline/detections: %s", msg->data.c_str());
    
    // Process the received data
    // Simple processing without CUDA/ONNX
    auto output_msg = std_msgs::msg::String();
    output_msg.data = "Processed: " + msg->data;
    /pipeline/final_results_pub_->publish(output_msg);
}

void post_processingNode::timer_callback() {
    // Periodic processing tasks
    post_process();
    
}

// Processing methods
void post_processingNode::post_process() {
    RCLCPP_DEBUG(this->get_logger(), "Executing post_process");
    // TODO: Implement post_process logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::post_processingNode) 