#ifndef ROBODSL_PROJECT_COLLISION_DETECTION_NODE_HPP
#define ROBODSL_PROJECT_COLLISION_DETECTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/collision_detection_cuda.hpp"


namespace robodsl_project {

class collision_detectionNode : public rclcpp::Node {
public:
    collision_detectionNode();
    ~collision_detectionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr optimized_trajectory_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr environment_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr collision_status_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<collision_detectionCudaManager> cuda_manager_;
    
    
    // Callback methods
    void optimized_trajectory_callback(const std_msgs::msg::String::SharedPtr msg);
    void environment_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void check_collisions();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_COLLISION_DETECTION_NODE_HPP 