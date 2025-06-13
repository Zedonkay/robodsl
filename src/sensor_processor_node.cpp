#include "sensor_processor/sensor_processor_node.h"

Sensor_processorNode::Sensor_processorNode()
    : Node("sensor_processor")
{
    // Node initialization code here
    RCLCPP_INFO(this->get_logger(), "sensor_processor node has been started");
}
