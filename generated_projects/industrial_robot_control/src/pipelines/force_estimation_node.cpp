#include "robodsl_project/force_estimation_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

force_estimationNode::force_estimationNode() 
    : Node("force_estimation_node", "/industrial_robot_pipeline/force_estimation") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing force_estimation stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "force_estimation stage node initialized successfully");
}

force_estimationNode::~force_estimationNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void force_estimationNode::initialize_ros_components() {
    // Initialize subscribers
    joint_torques_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_estimation/joint_torques", 10,
        std::bind(&force_estimationNode::joint_torques_callback, this, std::placeholders::_1)
    );
    joint_positions_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_estimation/joint_positions", 10,
        std::bind(&force_estimationNode::joint_positions_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    external_forces_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/industrial_robot_pipeline/force_estimation/external_forces", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&force_estimationNode::timer_callback, this)
    );
}


bool force_estimationNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<force_estimationOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void force_estimationNode::joint_torques_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received joint_torques: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from joint_torques";
    external_forces_pub_->publish(output_msg);
}
void force_estimationNode::joint_positions_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received joint_positions: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from joint_positions";
    external_forces_pub_->publish(output_msg);
}

void force_estimationNode::timer_callback() {
    // Periodic processing tasks
    estimate_forces();
    
}

// Processing methods
void force_estimationNode::estimate_forces() {
    RCLCPP_DEBUG(this->get_logger(), "Executing estimate_forces");
    // TODO: Implement estimate_forces logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::force_estimationNode) 