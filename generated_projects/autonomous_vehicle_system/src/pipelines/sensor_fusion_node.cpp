#include "robodsl_project/sensor_fusion_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

sensor_fusionNode::sensor_fusionNode() 
    : Node("sensor_fusion_node", "/autonomous_driving_pipeline/sensor_fusion") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing sensor_fusion stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "sensor_fusion stage node initialized successfully");
}

sensor_fusionNode::~sensor_fusionNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void sensor_fusionNode::initialize_ros_components() {
    // Initialize subscribers
    lidar_data_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/sensor_fusion/lidar_data", 10,
        std::bind(&sensor_fusionNode::lidar_data_callback, this, std::placeholders::_1)
    );
    camera_data_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/sensor_fusion/camera_data", 10,
        std::bind(&sensor_fusionNode::camera_data_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    fused_objects_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/sensor_fusion/fused_objects", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&sensor_fusionNode::timer_callback, this)
    );
}

bool sensor_fusionNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<sensor_fusionCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void sensor_fusionNode::lidar_data_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received lidar_data: %s", msg->data.c_str());
    
    // Process the received data
    std::vector<float> input_data;
    // TODO: Convert msg->data to input_data vector
    // For now, create dummy data
    input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    std::vector<float> output_data;
    
    // Process with CUDA kernels
    if (cuda_manager_) {
        if (cuda_manager_->process_data(input_data, output_data)) {
            RCLCPP_INFO(this->get_logger(), "CUDA processing completed");
        } else {
            RCLCPP_ERROR(this->get_logger(), "CUDA processing failed");
            return;
        }
    }
    
    
    // Publish results
    auto output_msg = std_msgs::msg::String();
    // TODO: Convert output_data to string format
    output_msg.data = "Processed data from lidar_data";
    fused_objects_pub_->publish(output_msg);
}
void sensor_fusionNode::camera_data_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received camera_data: %s", msg->data.c_str());
    
    // Process the received data
    std::vector<float> input_data;
    // TODO: Convert msg->data to input_data vector
    // For now, create dummy data
    input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    
    std::vector<float> output_data;
    
    // Process with CUDA kernels
    if (cuda_manager_) {
        if (cuda_manager_->process_data(input_data, output_data)) {
            RCLCPP_INFO(this->get_logger(), "CUDA processing completed");
        } else {
            RCLCPP_ERROR(this->get_logger(), "CUDA processing failed");
            return;
        }
    }
    
    
    // Publish results
    auto output_msg = std_msgs::msg::String();
    // TODO: Convert output_data to string format
    output_msg.data = "Processed data from camera_data";
    fused_objects_pub_->publish(output_msg);
}

void sensor_fusionNode::timer_callback() {
    // Periodic processing tasks
    fuse_sensors();
    
}

// Processing methods
void sensor_fusionNode::fuse_sensors() {
    RCLCPP_DEBUG(this->get_logger(), "Executing fuse_sensors");
    // TODO: Implement fuse_sensors logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::sensor_fusionNode) 