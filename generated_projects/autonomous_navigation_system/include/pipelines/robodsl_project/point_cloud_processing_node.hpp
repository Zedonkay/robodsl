#ifndef ROBODSL_PROJECT_POINT_CLOUD_PROCESSING_NODE_HPP
#define ROBODSL_PROJECT_POINT_CLOUD_PROCESSING_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/point_cloud_processing_cuda.hpp"


namespace robodsl_project {

class point_cloud_processingNode : public rclcpp::Node {
public:
    point_cloud_processingNode();
    ~point_cloud_processingNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr raw_point_cloud_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr filtered_point_cloud_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<point_cloud_processingCudaManager> cuda_manager_;
    
    
    // Callback methods
    void raw_point_cloud_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void filter_point_cloud();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_POINT_CLOUD_PROCESSING_NODE_HPP 