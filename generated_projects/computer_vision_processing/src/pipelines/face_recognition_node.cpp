#include "robodsl_project/face_recognition_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

face_recognitionNode::face_recognitionNode() 
    : Node("face_recognition_node", "/computer_vision_pipeline/face_recognition") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing face_recognition stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "face_recognition stage node initialized successfully");
}

face_recognitionNode::~face_recognitionNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void face_recognitionNode::initialize_ros_components() {
    // Initialize subscribers
    enhanced_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/face_recognition/enhanced_image", 10,
        std::bind(&face_recognitionNode::enhanced_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    face_embeddings_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/computer_vision_pipeline/face_recognition/face_embeddings", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&face_recognitionNode::timer_callback, this)
    );
}


bool face_recognitionNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<face_recognitionOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void face_recognitionNode::enhanced_image_callback(const std_msgs::msg::String::SharedPtr msg) {
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
    face_embeddings_pub_->publish(output_msg);
}

void face_recognitionNode::timer_callback() {
    // Periodic processing tasks
    recognize_faces();
    
}

// Processing methods
void face_recognitionNode::recognize_faces() {
    RCLCPP_DEBUG(this->get_logger(), "Executing recognize_faces");
    // TODO: Implement recognize_faces logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::face_recognitionNode) 