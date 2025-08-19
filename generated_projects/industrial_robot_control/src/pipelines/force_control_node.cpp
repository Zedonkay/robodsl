#include "robodsl_project/force_control_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

force_controlNode::force_controlNode() 
    : Node("force_control_node", "/industrial_robot_pipeline/force_control") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing force_control stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "force_control stage node initialized successfully");
}

force_controlNode::~force_controlNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void force_controlNode::initialize_ros_components() {
    // Initialize subscribers
    desired_force_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_control/desired_force", 10,
        std::bind(&force_controlNode::desired_force_callback, this, std::placeholders::_1)
    );
    measured_force_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_control/measured_force", 10,
        std::bind(&force_controlNode::measured_force_callback, this, std::placeholders::_1)
    );
    current_pose_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_control/current_pose", 10,
        std::bind(&force_controlNode::current_pose_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    force_correction_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_control/force_correction", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&force_controlNode::timer_callback, this)
    );
}

bool force_controlNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<force_controlCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void force_controlNode::desired_force_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received desired_force: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from desired_force";
    force_correction_pub_->publish(output_msg);
}
void force_controlNode::measured_force_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received measured_force: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from measured_force";
    force_correction_pub_->publish(output_msg);
}
void force_controlNode::current_pose_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received current_pose: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from current_pose";
    force_correction_pub_->publish(output_msg);
}

void force_controlNode::timer_callback() {
    // Periodic processing tasks
    compute_force_correction();
    
}

// Processing methods
void force_controlNode::compute_force_correction() {
    RCLCPP_DEBUG(this->get_logger(), "Executing compute_force_correction");
    // TODO: Implement compute_force_correction logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::force_controlNode) 