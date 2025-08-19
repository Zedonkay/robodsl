#include "robodsl_project/slam_mapping_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

slam_mappingNode::slam_mappingNode() 
    : Node("slam_mapping_node", "/autonomous_navigation_pipeline/slam_mapping") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing slam_mapping stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "slam_mapping stage node initialized successfully");
}

slam_mappingNode::~slam_mappingNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void slam_mappingNode::initialize_ros_components() {
    // Initialize subscribers
    filtered_point_cloud_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/slam_mapping/filtered_point_cloud", 10,
        std::bind(&slam_mappingNode::filtered_point_cloud_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    occupancy_grid_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/slam_mapping/occupancy_grid", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&slam_mappingNode::timer_callback, this)
    );
}

bool slam_mappingNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<slam_mappingCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void slam_mappingNode::filtered_point_cloud_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received filtered_point_cloud: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from filtered_point_cloud";
    occupancy_grid_pub_->publish(output_msg);
}

void slam_mappingNode::timer_callback() {
    // Periodic processing tasks
    update_occupancy_grid();
    
}

// Processing methods
void slam_mappingNode::update_occupancy_grid() {
    RCLCPP_DEBUG(this->get_logger(), "Executing update_occupancy_grid");
    // TODO: Implement update_occupancy_grid logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::slam_mappingNode) 