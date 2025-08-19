#include "robodsl_project/optical_flow_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

optical_flowNode::optical_flowNode() 
    : Node("optical_flow_node", "/computer_vision_pipeline/optical_flow") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing optical_flow stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "optical_flow stage node initialized successfully");
}

optical_flowNode::~optical_flowNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void optical_flowNode::initialize_ros_components() {
    // Initialize subscribers
    current_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/optical_flow/current_image", 10,
        std::bind(&optical_flowNode::current_image_callback, this, std::placeholders::_1)
    );
    previous_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/optical_flow/previous_image", 10,
        std::bind(&optical_flowNode::previous_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    flow_field_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/computer_vision_pipeline/optical_flow/flow_field", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&optical_flowNode::timer_callback, this)
    );
}


bool optical_flowNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<optical_flowOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void optical_flowNode::current_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received current_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from current_image";
    flow_field_pub_->publish(output_msg);
}
void optical_flowNode::previous_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received previous_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from previous_image";
    flow_field_pub_->publish(output_msg);
}

void optical_flowNode::timer_callback() {
    // Periodic processing tasks
    compute_optical_flow();
    
}

// Processing methods
void optical_flowNode::compute_optical_flow() {
    RCLCPP_DEBUG(this->get_logger(), "Executing compute_optical_flow");
    // TODO: Implement compute_optical_flow logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::optical_flowNode) 