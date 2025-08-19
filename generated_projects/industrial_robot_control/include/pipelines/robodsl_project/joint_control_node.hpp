#ifndef ROBODSL_PROJECT_JOINT_CONTROL_NODE_HPP
#define ROBODSL_PROJECT_JOINT_CONTROL_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>



namespace robodsl_project {

class joint_controlNode : public rclcpp::Node {
public:
    joint_controlNode();
    ~joint_controlNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr optimized_trajectory_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr force_correction_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr collision_status_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr joint_commands_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    
    // Callback methods
    void optimized_trajectory_callback(const std_msgs::msg::String::SharedPtr msg);
    void force_correction_callback(const std_msgs::msg::String::SharedPtr msg);
    void collision_status_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void compute_joint_commands();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_JOINT_CONTROL_NODE_HPP 