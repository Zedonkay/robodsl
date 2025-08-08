#ifndef ROBODSL_PROJECT_DETECTION_NODE_HPP
#define ROBODSL_PROJECT_DETECTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/detection_onnx.hpp"

namespace robodsl_project {

class detectionNode : public rclcpp::Node {
public:
    detectionNode();
    ~detectionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr preprocessed_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detection_result_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<detectionOnnxManager> onnx_manager_;
    
    // Callback methods
    void preprocessed_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void run_detection();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_DETECTION_NODE_HPP 