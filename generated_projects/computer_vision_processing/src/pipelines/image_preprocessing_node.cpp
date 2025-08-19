#include "robodsl_project/image_preprocessing_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

image_preprocessingNode::image_preprocessingNode() 
    : Node("image_preprocessing_node", "/computer_vision_pipeline/image_preprocessing") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing image_preprocessing stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "image_preprocessing stage node initialized successfully");
}

image_preprocessingNode::~image_preprocessingNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void image_preprocessingNode::initialize_ros_components() {
    // Initialize subscribers
    raw_image_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/computer_vision_pipeline/image_preprocessing/raw_image", 10,
        std::bind(&image_preprocessingNode::raw_image_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    preprocessed_image_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/computer_vision_pipeline/image_preprocessing/preprocessed_image", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&image_preprocessingNode::timer_callback, this)
    );
}

bool image_preprocessingNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<image_preprocessingCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void image_preprocessingNode::raw_image_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received raw_image: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from raw_image";
    preprocessed_image_pub_->publish(output_msg);
}

void image_preprocessingNode::timer_callback() {
    // Periodic processing tasks
    preprocess_image();
    
}

// Processing methods
void image_preprocessingNode::preprocess_image() {
    RCLCPP_DEBUG(this->get_logger(), "Executing preprocess_image");
    // TODO: Implement preprocess_image logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::image_preprocessingNode) 