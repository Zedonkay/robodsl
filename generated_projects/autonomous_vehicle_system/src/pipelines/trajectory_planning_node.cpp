#include "robodsl_project/trajectory_planning_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

trajectory_planningNode::trajectory_planningNode() 
    : Node("trajectory_planning_node", "/autonomous_driving_pipeline/trajectory_planning") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing trajectory_planning stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    // Initialize CUDA integration
    if (!initialize_cuda()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize CUDA integration");
    }
    
    
    RCLCPP_INFO(this->get_logger(), "trajectory_planning stage node initialized successfully");
}

trajectory_planningNode::~trajectory_planningNode() {
    if (cuda_manager_) {
        cuda_manager_->cleanup();
    }
    
}

void trajectory_planningNode::initialize_ros_components() {
    // Initialize subscribers
    current_pose_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/trajectory_planning/current_pose", 10,
        std::bind(&trajectory_planningNode::current_pose_callback, this, std::placeholders::_1)
    );
    target_pose_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/trajectory_planning/target_pose", 10,
        std::bind(&trajectory_planningNode::target_pose_callback, this, std::placeholders::_1)
    );
    obstacles_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/trajectory_planning/obstacles", 10,
        std::bind(&trajectory_planningNode::obstacles_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    planned_trajectory_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/trajectory_planning/planned_trajectory", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&trajectory_planningNode::timer_callback, this)
    );
}

bool trajectory_planningNode::initialize_cuda() {
    cuda_manager_ = std::make_unique<trajectory_planningCudaManager>();
    return cuda_manager_->initialize();
}


// Callback methods
void trajectory_planningNode::current_pose_callback(const std_msgs::msg::String::SharedPtr msg) {
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
    planned_trajectory_pub_->publish(output_msg);
}
void trajectory_planningNode::target_pose_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received target_pose: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from target_pose";
    planned_trajectory_pub_->publish(output_msg);
}
void trajectory_planningNode::obstacles_callback(const std_msgs::msg::String::SharedPtr msg) {
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
    planned_trajectory_pub_->publish(output_msg);
}

void trajectory_planningNode::timer_callback() {
    // Periodic processing tasks
    plan_trajectory();
    
}

// Processing methods
void trajectory_planningNode::plan_trajectory() {
    RCLCPP_DEBUG(this->get_logger(), "Executing plan_trajectory");
    // TODO: Implement plan_trajectory logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::trajectory_planningNode) 