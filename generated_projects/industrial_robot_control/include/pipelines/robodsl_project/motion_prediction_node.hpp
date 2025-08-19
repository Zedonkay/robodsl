#ifndef ROBODSL_PROJECT_MOTION_PREDICTION_NODE_HPP
#define ROBODSL_PROJECT_MOTION_PREDICTION_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include <memory>
#include <string>
#include <vector>

#include "robodsl_project/motion_prediction_cuda.hpp"


namespace robodsl_project {

class motion_predictionNode : public rclcpp::Node {
public:
    motion_predictionNode();
    ~motion_predictionNode();

private:
    // ROS2 publishers and subscribers
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr joint_history_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr prediction_horizon_sub_;
    
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr predicted_motion_pub_;
    
    // Timer for periodic processing
    rclcpp::TimerBase::SharedPtr timer_;
    
    // CUDA integration
    std::unique_ptr<motion_predictionCudaManager> cuda_manager_;
    
    
    // Callback methods
    void joint_history_callback(const std_msgs::msg::String::SharedPtr msg);
    void prediction_horizon_callback(const std_msgs::msg::String::SharedPtr msg);
    
    void timer_callback();
    
    // Processing methods
    void predict_motion();
    
    // Model processing methods
    
    // Initialize components
    void initialize_ros_components();
    bool initialize_cuda();
};

} // namespace robodsl_project

#endif // ROBODSL_PROJECT_MOTION_PREDICTION_NODE_HPP 