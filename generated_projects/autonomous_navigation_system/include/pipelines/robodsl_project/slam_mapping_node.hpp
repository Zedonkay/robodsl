#ifndef ROBODSL_PROJECT_SLAM_MAPPING_NODE_HPP
#define ROBODSL_PROJECT_SLAM_MAPPING_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/slam_mapping_cuda.hpp"


namespace robodsl_project {

class slam_mappingNode : public rclcpp::Node {
public:
    slam_mappingNode();
    ~slam_mappingNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr filtered_point_cloud_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr occupancy_grid_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<slam_mappingCudaManager> cuda_manager_;
    
    
    // Callback methods
    void filtered_point_cloud_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void update_occupancy_grid();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_SLAM_MAPPING_NODE_HPP 