#include "robodsl_project/depth_estimation_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

depth_estimationNode::depth_estimationNode() 
    : Node("depth_estimation_node", "/computer_vision_pipeline/depth_estimation") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing depth_estimation stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "depth_estimation stage node initialized successfully");
}

depth_estimationNode::~depth_estimationNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void depth_estimationNode::initialize_ros_components() {
    // Initialize subscribers
    enhanced_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/depth_estimation/enhanced_image", 10,
        std::bind(&depth_estimationNode::enhanced_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    depth_map_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/computer_vision_pipeline/depth_estimation/depth_map", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&depth_estimationNode::timer_callback, this)
    );
}


bool depth_estimationNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<depth_estimationOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void depth_estimationNode::enhanced_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received enhanced_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from enhanced_image";
    depth_map_pub_->publish(output_msg);
}

void depth_estimationNode::timer_callback() {
    // Periodic processing tasks
    estimate_depth();
    
}

// Processing methods
void depth_estimationNode::estimate_depth() {
    RCLCPP_DEBUG(this->get_logger(), "Executing estimate_depth");
    // TODO: Implement estimate_depth logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::depth_estimationNode) 