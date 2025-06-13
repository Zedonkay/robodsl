#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class Test_nodeNode(Node):
    def __init__(self):
        super().__init__('test_node')
        self.get_logger().info('test_node node has been started')

def main(args=None):
    rclpy.init(args=args)
    node = Test_nodeNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
