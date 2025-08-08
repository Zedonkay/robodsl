#include "robodsl_project/segmentation_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

segmentationNode::segmentationNode() 
    : Node("segmentation_node", "/perception_pipeline/segmentation") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing segmentation stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "segmentation stage node initialized successfully");
}

segmentationNode::~segmentationNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void segmentationNode::initialize_ros_components() {
    // Initialize subscribers
    preprocessed_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/perception_pipeline/segmentation/preprocessed_image", 10,
        std::bind(&segmentationNode::preprocessed_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    segmentation_result_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/perception_pipeline/segmentation/segmentation_result", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&segmentationNode::timer_callback, this)
    );
}


bool segmentationNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<segmentationOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void segmentationNode::preprocessed_image_callback(const std_msgs::msg::String::SharedPtr msg) {
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
    segmentation_result_pub_->publish(output_msg);
}

void segmentationNode::timer_callback() {
    // Periodic processing tasks
    run_segmentation();
    
}

// Processing methods
void segmentationNode::run_segmentation() {
    RCLCPP_DEBUG(this->get_logger(), "Executing run_segmentation");
    // TODO: Implement run_segmentation logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::segmentationNode) 