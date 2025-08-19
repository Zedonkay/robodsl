#include "robodsl_project/object_detection_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

object_detectionNode::object_detectionNode() 
    : Node("object_detection_node", "/autonomous_navigation_pipeline/object_detection") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing object_detection stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "object_detection stage node initialized successfully");
}

object_detectionNode::~object_detectionNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void object_detectionNode::initialize_ros_components() {
    // Initialize subscribers
    camera_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/object_detection/camera_image", 10,
        std::bind(&object_detectionNode::camera_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    detected_objects_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/object_detection/detected_objects", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&object_detectionNode::timer_callback, this)
    );
}


bool object_detectionNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<object_detectionOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void object_detectionNode::camera_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received camera_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from camera_image";
    detected_objects_pub_->publish(output_msg);
}

void object_detectionNode::timer_callback() {
    // Periodic processing tasks
    detect_objects();
    
}

// Processing methods
void object_detectionNode::detect_objects() {
    RCLCPP_DEBUG(this->get_logger(), "Executing detect_objects");
    // TODO: Implement detect_objects logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::object_detectionNode) 