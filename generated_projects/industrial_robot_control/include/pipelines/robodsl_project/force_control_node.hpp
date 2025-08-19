#ifndef ROBODSL_PROJECT_FORCE_CONTROL_NODE_HPP
#define ROBODSL_PROJECT_FORCE_CONTROL_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/force_control_cuda.hpp"


namespace robodsl_project {

class force_controlNode : public rclcpp::Node {
public:
    force_controlNode();
    ~force_controlNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr desired_force_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr measured_force_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr current_pose_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr force_correction_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<force_controlCudaManager> cuda_manager_;
    
    
    // Callback methods
    void desired_force_callback(const std_msgs::msg::String::SharedPtr msg);
    void measured_force_callback(const std_msgs::msg::String::SharedPtr msg);
    void current_pose_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void compute_force_correction();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_FORCE_CONTROL_NODE_HPP 