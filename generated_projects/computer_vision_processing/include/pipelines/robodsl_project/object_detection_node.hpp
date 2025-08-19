#ifndef ROBODSL_PROJECT_OBJECT_DETECTION_NODE_HPP
#define ROBODSL_PROJECT_OBJECT_DETECTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/object_detection_onnx.hpp"

namespace robodsl_project {

class object_detectionNode : public rclcpp::Node {
public:
    object_detectionNode();
    ~object_detectionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr enhanced_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detected_objects_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<object_detectionOnnxManager> onnx_manager_;
    
    // Callback methods
    void enhanced_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void detect_objects();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_OBJECT_DETECTION_NODE_HPP 