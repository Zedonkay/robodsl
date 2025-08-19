#ifndef ROBODSL_PROJECT_FEATURE_MATCHING_NODE_HPP
#define ROBODSL_PROJECT_FEATURE_MATCHING_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/feature_matching_cuda.hpp"


namespace robodsl_project {

class feature_matchingNode : public rclcpp::Node {
public:
    feature_matchingNode();
    ~feature_matchingNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr current_image_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr reference_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr feature_matches_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<feature_matchingCudaManager> cuda_manager_;
    
    
    // Callback methods
    void current_image_callback(const std_msgs::msg::String::SharedPtr msg);
    void reference_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void match_features();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_FEATURE_MATCHING_NODE_HPP 