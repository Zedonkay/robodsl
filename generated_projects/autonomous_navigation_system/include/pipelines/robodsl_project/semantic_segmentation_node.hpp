#ifndef ROBODSL_PROJECT_SEMANTIC_SEGMENTATION_NODE_HPP
#define ROBODSL_PROJECT_SEMANTIC_SEGMENTATION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/semantic_segmentation_onnx.hpp"

namespace robodsl_project {

class semantic_segmentationNode : public rclcpp::Node {
public:
    semantic_segmentationNode();
    ~semantic_segmentationNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr camera_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr semantic_map_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<semantic_segmentationOnnxManager> onnx_manager_;
    
    // Callback methods
    void camera_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void segment_image();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_SEMANTIC_SEGMENTATION_NODE_HPP 