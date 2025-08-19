#ifndef ROBODSL_PROJECT_OPTICAL_FLOW_NODE_HPP
#define ROBODSL_PROJECT_OPTICAL_FLOW_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/optical_flow_onnx.hpp"

namespace robodsl_project {

class optical_flowNode : public rclcpp::Node {
public:
    optical_flowNode();
    ~optical_flowNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr current_image_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr previous_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr flow_field_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<optical_flowOnnxManager> onnx_manager_;
    
    // Callback methods
    void current_image_callback(const std_msgs::msg::String::SharedPtr msg);
    void previous_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void compute_optical_flow();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_OPTICAL_FLOW_NODE_HPP 