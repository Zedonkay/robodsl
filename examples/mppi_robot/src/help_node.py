#!/usr/bin/env python3

import rclpy
from rclpy.node import Node



class HelpNode(Node):
    def __init__(self):
        super().__init__('help')
        self.get_logger().info('help node started')
        self.get_logger().info('help node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = HelpNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
