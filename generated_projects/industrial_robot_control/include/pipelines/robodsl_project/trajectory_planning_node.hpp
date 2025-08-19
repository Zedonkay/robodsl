#ifndef ROBODSL_PROJECT_TRAJECTORY_PLANNING_NODE_HPP
#define ROBODSL_PROJECT_TRAJECTORY_PLANNING_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/trajectory_planning_cuda.hpp"


namespace robodsl_project {

class trajectory_planningNode : public rclcpp::Node {
public:
    trajectory_planningNode();
    ~trajectory_planningNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr target_pose_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr current_joints_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr planned_trajectory_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<trajectory_planningCudaManager> cuda_manager_;
    
    
    // Callback methods
    void target_pose_callback(const std_msgs::msg::String::SharedPtr msg);
    void current_joints_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void plan_trajectory();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_TRAJECTORY_PLANNING_NODE_HPP 