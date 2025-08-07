#include "robodsl_project/inference_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

inferenceNode::inferenceNode() 
    : Node("inference_node", "/onnx_pipeline/inference") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing inference stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "inference stage node initialized successfully");
}

inferenceNode::~inferenceNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void inferenceNode::initialize_ros_components() {
    // Initialize subscribers
    input_data_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/onnx_pipeline/inference/input_data", 10,
        std::bind(&inferenceNode::input_data_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    output_data_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/onnx_pipeline/inference/output_data", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&inferenceNode::timer_callback, this)
    );
}


bool inferenceNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<inferenceOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void inferenceNode::input_data_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received input_data: %s", msg->data.c_str());
    
    // Process the received data
    std::vector<float> input_data;
    // TODO: Convert msg->data to input_data vector
    // For now, create dummy data
    input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    std::vector<float> output_data;
    
    
    // Process with ONNX models
    if (onnx_manager_) {
        if (onnx_manager_->run_inference(output_data.empty() ? input_data : output_data, output_data)) {
            RCLCPP_INFO(this->get_logger(), "ONNX inference completed");
        } else {
            RCLCPP_ERROR(this->get_logger(), "ONNX inference failed");
            return;
        }
    }
    
    // Publish results
    auto output_msg = std_msgs::msg::String();
    // TODO: Convert output_data to string format
    output_msg.data = "Processed data from input_data";
    output_data_pub_->publish(output_msg);
}

void inferenceNode::timer_callback() {
    // Periodic processing tasks
    run_inference();
    
}

// Processing methods
void inferenceNode::run_inference() {
    RCLCPP_DEBUG(this->get_logger(), "Executing run_inference");
    // TODO: Implement run_inference logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::inferenceNode) 