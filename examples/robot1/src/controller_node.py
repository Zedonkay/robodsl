#!/usr/bin/env python3

import rclpy
from rclpy.node import Node



class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller')
        self.get_logger().info('controller node started')
        self.get_logger().info('controller node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
