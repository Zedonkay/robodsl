#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class Image_preprocessingNode(Node):
    """Image_preprocessing stage node for vision_pipeline pipeline."""
    
    def __init__(self):
        super().__init__('image_preprocessing_node')
        
        # Set namespace
        self.set_namespace('/vision_pipeline/image_preprocessing')
        
        self.get_logger().info(f'Initializing Image_preprocessingNode')
        
        # Initialize state
        self.initialized = False
        self.processing_count = 0
        
        # Declare parameters
        self.declare_parameter('preprocessing_status_topic', '/pipeline/preprocessing_status')
        self.preprocessing_status_topic = self.get_parameter('preprocessing_status_topic').value
        
        # Initialize publishers
        self./pipeline/preprocessed_publisher = self.create_publisher(
            String, '/vision_pipeline/image_preprocessing//pipeline/preprocessed', 10)
        
        # Initialize subscribers
        self./camera/image_raw_subscriber = self.create_subscription(
            String, '/vision_pipeline/image_preprocessing//camera/image_raw',
            self.on_/camera/image_raw_received, 10)
        
        # Initialize timer for periodic processing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        self.initialized = True
        self.get_logger().info('Image_preprocessingNode initialized successfully')
    
    def on_/camera/image_raw_received(self, msg):
        """Handle received /camera/image_raw message."""
        self.get_logger().info(f'Received /camera/image_raw: {msg.data}')
        
        # Process the input
        self.preprocess_image()
        
        # Publish outputs if available
        output_msg = String()
        output_msg.data = f'Processed /camera/image_raw: {msg.data}'
        self./pipeline/preprocessed_publisher.publish(output_msg)
    
    def preprocess_image(self):
        """Execute preprocess_image processing."""
        self.get_logger().debug(f'Executing preprocess_image')
        
        # TODO: Implement preprocess_image logic
        # This is where the actual processing logic would go
        
        self.processing_count += 1
    
    def timer_callback(self):
        """Periodic timer callback."""
        if not self.initialized:
            return
        
        # Periodic processing
        self.get_logger().debug(f'Image_preprocessingNode timer callback, count: {self.processing_count}')
        
        # Publish periodic status
        status_msg = String()
        status_msg.data = f'Image_preprocessingNode status: {self.processing_count}'
        self./pipeline/preprocessed_publisher.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = Image_preprocessingNode()
    
    node.get_logger().info('Starting Image_preprocessingNode')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 