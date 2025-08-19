#ifndef ROBODSL_PROJECT_SENSOR_FUSION_NODE_HPP
#define ROBODSL_PROJECT_SENSOR_FUSION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/sensor_fusion_cuda.hpp"


namespace robodsl_project {

class sensor_fusionNode : public rclcpp::Node {
public:
    sensor_fusionNode();
    ~sensor_fusionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr lidar_data_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr camera_data_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr fused_objects_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<sensor_fusionCudaManager> cuda_manager_;
    
    
    // Callback methods
    void lidar_data_callback(const std_msgs::msg::String::SharedPtr msg);
    void camera_data_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void fuse_sensors();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_SENSOR_FUSION_NODE_HPP 