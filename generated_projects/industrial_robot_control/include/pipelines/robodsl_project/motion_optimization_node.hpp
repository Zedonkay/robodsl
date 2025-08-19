#ifndef ROBODSL_PROJECT_MOTION_OPTIMIZATION_NODE_HPP
#define ROBODSL_PROJECT_MOTION_OPTIMIZATION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/motion_optimization_onnx.hpp"

namespace robodsl_project {

class motion_optimizationNode : public rclcpp::Node {
public:
    motion_optimizationNode();
    ~motion_optimizationNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr planned_trajectory_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr constraints_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr optimized_trajectory_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<motion_optimizationOnnxManager> onnx_manager_;
    
    // Callback methods
    void planned_trajectory_callback(const std_msgs::msg::String::SharedPtr msg);
    void constraints_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void optimize_motion();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_MOTION_OPTIMIZATION_NODE_HPP 