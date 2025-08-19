#ifndef ROBODSL_PROJECT_SAFETY_MONITORING_NODE_HPP
#define ROBODSL_PROJECT_SAFETY_MONITORING_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/safety_monitoring_cuda.hpp"


namespace robodsl_project {

class safety_monitoringNode : public rclcpp::Node {
public:
    safety_monitoringNode();
    ~safety_monitoringNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr vehicle_state_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr obstacles_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr safety_thresholds_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr safety_violations_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<safety_monitoringCudaManager> cuda_manager_;
    
    
    // Callback methods
    void vehicle_state_callback(const std_msgs::msg::String::SharedPtr msg);
    void obstacles_callback(const std_msgs::msg::String::SharedPtr msg);
    void safety_thresholds_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void monitor_safety();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_SAFETY_MONITORING_NODE_HPP 