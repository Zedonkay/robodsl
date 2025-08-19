#ifndef ROBODSL_PROJECT_FACE_RECOGNITION_NODE_HPP
#define ROBODSL_PROJECT_FACE_RECOGNITION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>


#include "robodsl_project/face_recognition_onnx.hpp"

namespace robodsl_project {

class face_recognitionNode : public rclcpp::Node {
public:
    face_recognitionNode();
    ~face_recognitionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr enhanced_image_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr face_embeddings_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    
    // ONNX integration
    std::unique_ptr<face_recognitionOnnxManager> onnx_manager_;
    
    // Callback methods
    void enhanced_image_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void recognize_faces();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_onnx();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_FACE_RECOGNITION_NODE_HPP 