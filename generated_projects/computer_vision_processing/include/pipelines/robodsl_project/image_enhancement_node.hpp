#ifndef ROBODSL_PROJECT_IMAGE_ENHANCEMENT_NODE_HPP
#define ROBODSL_PROJECT_IMAGE_ENHANCEMENT_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/image_enhancement_cuda.hpp"


namespace robodsl_project {

class image_enhancementNode : public rclcpp::Node {
public:
    image_enhancementNode();
    ~image_enhancementNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr preprocessed_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr enhanced_image_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<image_enhancementCudaManager> cuda_manager_;
    
    
    // Callback methods
    void preprocessed_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void enhance_image();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_IMAGE_ENHANCEMENT_NODE_HPP 