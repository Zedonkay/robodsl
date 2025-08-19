#ifndef ROBODSL_PROJECT_BEHAVIOR_PREDICTION_NODE_HPP
#define ROBODSL_PROJECT_BEHAVIOR_PREDICTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/behavior_prediction_onnx.hpp"

namespace robodsl_project {

class behavior_predictionNode : public rclcpp::Node {
public:
    behavior_predictionNode();
    ~behavior_predictionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr vehicle_states_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr road_context_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr predicted_behaviors_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<behavior_predictionOnnxManager> onnx_manager_;
    
    // Callback methods
    void vehicle_states_callback(const std_msgs::msg::String::SharedPtr msg);
    void road_context_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void predict_behaviors();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_BEHAVIOR_PREDICTION_NODE_HPP 