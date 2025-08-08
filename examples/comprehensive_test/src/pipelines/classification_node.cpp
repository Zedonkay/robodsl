#include "robodsl_project/classification_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

classificationNode::classificationNode() 
    : Node("classification_node", "/perception_pipeline/classification") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing classification stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "classification stage node initialized successfully");
}

classificationNode::~classificationNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void classificationNode::initialize_ros_components() {
    // Initialize subscribers
    preprocessed_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/perception_pipeline/classification/preprocessed_image", 10,
        std::bind(&classificationNode::preprocessed_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    classification_result_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/perception_pipeline/classification/classification_result", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&classificationNode::timer_callback, this)
    );
}


bool classificationNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<classificationOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void classificationNode::preprocessed_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received preprocessed_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from preprocessed_image";
    classification_result_pub_->publish(output_msg);
}

void classificationNode::timer_callback() {
    // Periodic processing tasks
    run_classification();
    
}

// Processing methods
void classificationNode::run_classification() {
    RCLCPP_DEBUG(this->get_logger(), "Executing run_classification");
    // TODO: Implement run_classification logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::classificationNode) 