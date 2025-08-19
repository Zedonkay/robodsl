#ifndef ROBODSL_PROJECT_LANE_DETECTION_NODE_HPP
#define ROBODSL_PROJECT_LANE_DETECTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/lane_detection_onnx.hpp"

namespace robodsl_project {

class lane_detectionNode : public rclcpp::Node {
public:
    lane_detectionNode();
    ~lane_detectionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr camera_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lane_markings_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<lane_detectionOnnxManager> onnx_manager_;
    
    // Callback methods
    void camera_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void detect_lanes();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_LANE_DETECTION_NODE_HPP 