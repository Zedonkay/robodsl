#include "robodsl_project/safety_monitoring_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

safety_monitoringNode::safety_monitoringNode() 
    : Node("safety_monitoring_node", "/autonomous_driving_pipeline/safety_monitoring") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing safety_monitoring stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "safety_monitoring stage node initialized successfully");
}

safety_monitoringNode::~safety_monitoringNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void safety_monitoringNode::initialize_ros_components() {
    // Initialize subscribers
    vehicle_state_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/safety_monitoring/vehicle_state", 10,
        std::bind(&safety_monitoringNode::vehicle_state_callback, this, std::placeholders::_1)
    );
    obstacles_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/safety_monitoring/obstacles", 10,
        std::bind(&safety_monitoringNode::obstacles_callback, this, std::placeholders::_1)
    );
    safety_thresholds_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/safety_monitoring/safety_thresholds", 10,
        std::bind(&safety_monitoringNode::safety_thresholds_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    safety_violations_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/safety_monitoring/safety_violations", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&safety_monitoringNode::timer_callback, this)
    );
}

bool safety_monitoringNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<safety_monitoringCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void safety_monitoringNode::vehicle_state_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received vehicle_state: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from vehicle_state";
    safety_violations_pub_->publish(output_msg);
}
void safety_monitoringNode::obstacles_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received obstacles: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from obstacles";
    safety_violations_pub_->publish(output_msg);
}
void safety_monitoringNode::safety_thresholds_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received safety_thresholds: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from safety_thresholds";
    safety_violations_pub_->publish(output_msg);
}

void safety_monitoringNode::timer_callback() {
    // Periodic processing tasks
    monitor_safety();
    
}

// Processing methods
void safety_monitoringNode::monitor_safety() {
    RCLCPP_DEBUG(this->get_logger(), "Executing monitor_safety");
    // TODO: Implement monitor_safety logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::safety_monitoringNode) 