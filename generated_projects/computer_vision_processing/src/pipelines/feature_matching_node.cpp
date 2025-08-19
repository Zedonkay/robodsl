#include "robodsl_project/feature_matching_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

feature_matchingNode::feature_matchingNode() 
    : Node("feature_matching_node", "/computer_vision_pipeline/feature_matching") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing feature_matching stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "feature_matching stage node initialized successfully");
}

feature_matchingNode::~feature_matchingNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void feature_matchingNode::initialize_ros_components() {
    // Initialize subscribers
    current_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/feature_matching/current_image", 10,
        std::bind(&feature_matchingNode::current_image_callback, this, std::placeholders::_1)
    );
    reference_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/feature_matching/reference_image", 10,
        std::bind(&feature_matchingNode::reference_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    feature_matches_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/computer_vision_pipeline/feature_matching/feature_matches", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&feature_matchingNode::timer_callback, this)
    );
}

bool feature_matchingNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<feature_matchingCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void feature_matchingNode::current_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received current_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from current_image";
    feature_matches_pub_->publish(output_msg);
}
void feature_matchingNode::reference_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received reference_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from reference_image";
    feature_matches_pub_->publish(output_msg);
}

void feature_matchingNode::timer_callback() {
    // Periodic processing tasks
    match_features();
    
}

// Processing methods
void feature_matchingNode::match_features() {
    RCLCPP_DEBUG(this->get_logger(), "Executing match_features");
    // TODO: Implement match_features logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::feature_matchingNode) 