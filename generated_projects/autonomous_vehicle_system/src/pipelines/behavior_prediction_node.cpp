#include "robodsl_project/behavior_prediction_node.hpp"
#include <chrono>
#include <iostream>

using namespace std::chrono_literals;

namespace robodsl_project {

behavior_predictionNode::behavior_predictionNode() 
    : Node("behavior_prediction_node", "/autonomous_driving_pipeline/behavior_prediction") {
    
    RCLCPP_INFO(this->get_logger(), "Initializing behavior_prediction stage node");
    
    // Initialize ROS components
    initialize_ros_components();
    
    
    // Initialize ONNX integration
    if (!initialize_onnx()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX integration");
    }
    
    RCLCPP_INFO(this->get_logger(), "behavior_prediction stage node initialized successfully");
}

behavior_predictionNode::~behavior_predictionNode() {
    
    if (onnx_manager_) {
        onnx_manager_->cleanup();
    }
}

void behavior_predictionNode::initialize_ros_components() {
    // Initialize subscribers
    vehicle_states_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/behavior_prediction/vehicle_states", 10,
        std::bind(&behavior_predictionNode::vehicle_states_callback, this, std::placeholders::_1)
    );
    road_context_sub_ = this->create_subscription<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/behavior_prediction/road_context", 10,
        std::bind(&behavior_predictionNode::road_context_callback, this, std::placeholders::_1)
    );
    
    // Initialize publishers
    predicted_behaviors_pub_ = this->create_publisher<std_msgs::msg::String>(
        "/autonomous_driving_pipeline/behavior_prediction/predicted_behaviors", 10
    );
    
    // Initialize timer for periodic processing
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(100),
        std::bind(&behavior_predictionNode::timer_callback, this)
    );
}


bool behavior_predictionNode::initialize_onnx() {
    onnx_manager_ = std::make_unique<behavior_predictionOnnxManager>();
    return onnx_manager_->initialize();
}

// Callback methods
void behavior_predictionNode::vehicle_states_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received vehicle_states: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from vehicle_states";
    predicted_behaviors_pub_->publish(output_msg);
}
void behavior_predictionNode::road_context_callback(const std_msgs::msg::String::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Received road_context: %s", msg->data.c_str());
    
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
    output_msg.data = "Processed data from road_context";
    predicted_behaviors_pub_->publish(output_msg);
}

void behavior_predictionNode::timer_callback() {
    // Periodic processing tasks
    predict_behaviors();
    
}

// Processing methods
void behavior_predictionNode::predict_behaviors() {
    RCLCPP_DEBUG(this->get_logger(), "Executing predict_behaviors");
    // TODO: Implement predict_behaviors logic
}

// Model processing methods

} // namespace robodsl_project

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(robodsl_project::behavior_predictionNode) 