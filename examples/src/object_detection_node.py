#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class Object_detectionNode(Node):
    """Object_detection stage node for vision_pipeline pipeline."""
    
    def __init__(self):
        super().__init__('object_detection_node')
        
        # Set namespace
        self.set_namespace('/vision_pipeline/object_detection')
        
        self.get_logger().info(f'Initializing Object_detectionNode')
        
        # Initialize state
        self.initialized = False
        self.processing_count = 0
        
        # Declare parameters
        self.declare_parameter('detection_status_topic', '/pipeline/detection_status')
        self.detection_status_topic = self.get_parameter('detection_status_topic').value
        
        # Initialize publishers
        self./pipeline/detections_publisher = self.create_publisher(
            String, '/vision_pipeline/object_detection//pipeline/detections', 10)
        
        # Initialize subscribers
        self./pipeline/preprocessed_subscriber = self.create_subscription(
            String, '/vision_pipeline/object_detection//pipeline/preprocessed',
            self.on_/pipeline/preprocessed_received, 10)
        
        # Initialize timer for periodic processing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        self.initialized = True
        self.get_logger().info('Object_detectionNode initialized successfully')
    
    def on_/pipeline/preprocessed_received(self, msg):
        """Handle received /pipeline/preprocessed message."""
        self.get_logger().info(f'Received /pipeline/preprocessed: {msg.data}')
        
        # Process the input
        self.detect_objects()
        
        # Publish outputs if available
        output_msg = String()
        output_msg.data = f'Processed /pipeline/preprocessed: {msg.data}'
        self./pipeline/detections_publisher.publish(output_msg)
    
    def detect_objects(self):
        """Execute detect_objects processing."""
        self.get_logger().debug(f'Executing detect_objects')
        
        # TODO: Implement detect_objects logic
        # This is where the actual processing logic would go
        
        self.processing_count += 1
    
    def timer_callback(self):
        """Periodic timer callback."""
        if not self.initialized:
            return
        
        # Periodic processing
        self.get_logger().debug(f'Object_detectionNode timer callback, count: {self.processing_count}')
        
        # Publish periodic status
        status_msg = String()
        status_msg.data = f'Object_detectionNode status: {self.processing_count}'
        self./pipeline/detections_publisher.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = Object_detectionNode()
    
    node.get_logger().info('Starting Object_detectionNode')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 