#ifndef ROBODSL_PROJECT_OBSTACLE_AVOIDANCE_NODE_HPP
#define ROBODSL_PROJECT_OBSTACLE_AVOIDANCE_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>



namespace robodsl_project {

class obstacle_avoidanceNode : public rclcpp::Node {
public:
    obstacle_avoidanceNode();
    ~obstacle_avoidanceNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr velocity_command,sensor_data_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr safe_velocity_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    
    // Callback methods
    void velocity_command,sensor_data_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void avoid_obstacles();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_OBSTACLE_AVOIDANCE_NODE_HPP 