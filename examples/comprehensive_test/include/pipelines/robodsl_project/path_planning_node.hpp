#ifndef ROBODSL_PROJECT_PATH_PLANNING_NODE_HPP
#define ROBODSL_PROJECT_PATH_PLANNING_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>



namespace robodsl_project {

class path_planningNode : public rclcpp::Node {
public:
    path_planningNode();
    ~path_planningNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr goal_pose,current_pose,map_data_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr planned_path_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    
    // Callback methods
    void goal_pose,current_pose,map_data_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void plan_path();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_PATH_PLANNING_NODE_HPP 