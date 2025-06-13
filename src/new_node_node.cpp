#include "new_node/new_node_node.h"

New_nodeNode::New_nodeNode()
    : Node("new_node")
{
    // Node initialization code here
    RCLCPP_INFO(this->get_logger(), "new_node node has been started");
}
