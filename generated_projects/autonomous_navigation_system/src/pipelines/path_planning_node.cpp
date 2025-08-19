#include "robodsl_project/path_planning_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

path_planningNode::path_planningNode() 
    : Node("path_planning_node", "/autonomous_navigation_pipeline/path_planning") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing path_planning stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "path_planning stage node initialized successfully");
}

path_planningNode::~path_planningNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void path_planningNode::initialize_ros_components() {
    // Initialize subscribers
    occupancy_grid_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/path_planning/occupancy_grid", 10,
        std::bind(&path_planningNode::occupancy_grid_callback, this, std::placeholders::_1)
    );
    detected_objects_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/path_planning/detected_objects", 10,
        std::bind(&path_planningNode::detected_objects_callback, this, std::placeholders::_1)
    );
    semantic_map_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/path_planning/semantic_map", 10,
        std::bind(&path_planningNode::semantic_map_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    planned_path_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_navigation_pipeline/path_planning/planned_path", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&path_planningNode::timer_callback, this)
    );
}

bool path_planningNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<path_planningCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void path_planningNode::occupancy_grid_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received occupancy_grid: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from occupancy_grid";
    planned_path_pub_->publish(output_msg);
}
void path_planningNode::detected_objects_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received detected_objects: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from detected_objects";
    planned_path_pub_->publish(output_msg);
}
void path_planningNode::semantic_map_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received semantic_map: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from semantic_map";
    planned_path_pub_->publish(output_msg);
}

void path_planningNode::timer_callback() {
    // Periodic processing tasks
    plan_path();
    
}

// Processing methods
void path_planningNode::plan_path() {
    RCLCPP_DEBUG(this->get_logger(), "Executing plan_path");
    // TODO: Implement plan_path logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::path_planningNode) 