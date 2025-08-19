#include "robodsl_project/point_cloud_processing_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

point_cloud_processingNode::point_cloud_processingNode() 
    : Node("point_cloud_processing_node", "/autonomous_navigation_pipeline/point_cloud_processing") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing point_cloud_processing stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "point_cloud_processing stage node initialized successfully");
}

point_cloud_processingNode::~point_cloud_processingNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void point_cloud_processingNode::initialize_ros_components() {
    // Initialize subscribers
    raw_point_cloud_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/point_cloud_processing/raw_point_cloud", 10,
        std::bind(&point_cloud_processingNode::raw_point_cloud_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    filtered_point_cloud_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/point_cloud_processing/filtered_point_cloud", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&point_cloud_processingNode::timer_callback, this)
    );
}

bool point_cloud_processingNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<point_cloud_processingCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void point_cloud_processingNode::raw_point_cloud_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received raw_point_cloud: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from raw_point_cloud";
    filtered_point_cloud_pub_->publish(output_msg);
}

void point_cloud_processingNode::timer_callback() {
    // Periodic processing tasks
    filter_point_cloud();
    
}

// Processing methods
void point_cloud_processingNode::filter_point_cloud() {
    RCLCPP_DEBUG(this->get_logger(), "Executing filter_point_cloud");
    // TODO: Implement filter_point_cloud logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::point_cloud_processingNode) 