#ifndef ROBODSL_PROJECT_MOTION_CONTROL_NODE_HPP
#define ROBODSL_PROJECT_MOTION_CONTROL_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>



namespace robodsl_project {

class motion_controlNode : public rclcpp::Node {
public:
    motion_controlNode();
    ~motion_controlNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr planned_path_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr robot_pose_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr velocity_command_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    
    // Callback methods
    void planned_path_callback(const std_msgs::msg::String::SharedPtr msg);
    void robot_pose_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void compute_velocity();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_MOTION_CONTROL_NODE_HPP 