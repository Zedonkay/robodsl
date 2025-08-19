#ifndef ROBODSL_PROJECT_TRAFFIC_SIGN_DETECTION_NODE_HPP
#define ROBODSL_PROJECT_TRAFFIC_SIGN_DETECTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/traffic_sign_detection_onnx.hpp"

namespace robodsl_project {

class traffic_sign_detectionNode : public rclcpp::Node {
public:
    traffic_sign_detectionNode();
    ~traffic_sign_detectionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr camera_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr traffic_signs_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<traffic_sign_detectionOnnxManager> onnx_manager_;
    
    // Callback methods
    void camera_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void detect_traffic_signs();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_TRAFFIC_SIGN_DETECTION_NODE_HPP 