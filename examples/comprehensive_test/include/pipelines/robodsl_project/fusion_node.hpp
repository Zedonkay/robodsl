#ifndef ROBODSL_PROJECT_FUSION_NODE_HPP
#define ROBODSL_PROJECT_FUSION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>



namespace robodsl_project {

class fusionNode : public rclcpp::Node {
public:
    fusionNode();
    ~fusionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr classification_result,detection_result,segmentation_result_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr final_result_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    
    // Callback methods
    void classification_result,detection_result,segmentation_result_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void fuse_results();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_FUSION_NODE_HPP 