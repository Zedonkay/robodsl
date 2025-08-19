#include "robodsl_project/motion_optimization_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

motion_optimizationNode::motion_optimizationNode() 
    : Node("motion_optimization_node", "/industrial_robot_pipeline/motion_optimization") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing motion_optimization stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "motion_optimization stage node initialized successfully");
}

motion_optimizationNode::~motion_optimizationNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void motion_optimizationNode::initialize_ros_components() {
    // Initialize subscribers
    planned_trajectory_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/motion_optimization/planned_trajectory", 10,
        std::bind(&motion_optimizationNode::planned_trajectory_callback, this, std::placeholders::_1)
    );
    constraints_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/motion_optimization/constraints", 10,
        std::bind(&motion_optimizationNode::constraints_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    optimized_trajectory_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/industrial_robot_pipeline/motion_optimization/optimized_trajectory", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&motion_optimizationNode::timer_callback, this)
    );
}


bool motion_optimizationNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<motion_optimizationOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void motion_optimizationNode::planned_trajectory_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received planned_trajectory: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from planned_trajectory";
    optimized_trajectory_pub_->publish(output_msg);
}
void motion_optimizationNode::constraints_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received constraints: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from constraints";
    optimized_trajectory_pub_->publish(output_msg);
}

void motion_optimizationNode::timer_callback() {
    // Periodic processing tasks
    optimize_motion();
    
}

// Processing methods
void motion_optimizationNode::optimize_motion() {
    RCLCPP_DEBUG(this->get_logger(), "Executing optimize_motion");
    // TODO: Implement optimize_motion logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::motion_optimizationNode) 