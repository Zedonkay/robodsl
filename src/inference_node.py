#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time


class InferenceNode(Node):
    """Inference stage node for onnx_pipeline pipeline."""
    
    def __init__(self):
        super().__init__('inference_node')
        
        # Set namespace
        self.set_namespace('/onnx_pipeline/inference')
        
        self.get_logger().info(f'Initializing InferenceNode')
        
        # Initialize state
        self.initialized = False
        self.processing_count = 0
        
        # Declare parameters
        self.declare_parameter('inference_topic', '/pipeline/inference')
        self.inference_topic = self.get_parameter('inference_topic').value
        
        # Initialize publishers
        self.output_data_publisher = self.create_publisher(
            String, '/onnx_pipeline/inference/output_data', 10)
        
        # Initialize subscribers
        self.input_data_subscriber = self.create_subscription(
            String, '/onnx_pipeline/inference/input_data',
            self.on_input_data_received, 10)
        
        # Initialize timer for periodic processing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        
        self.initialized = True
        self.get_logger().info('InferenceNode initialized successfully')
    
    def on_input_data_received(self, msg):
        """Handle received input_data message."""
        self.get_logger().info(f'Received input_data: {msg.data}')
        
        # Process the input
        self.run_inference()
        
        # Publish outputs if available
        output_msg = String()
        output_msg.data = f'Processed input_data: {msg.data}'
        self.output_data_publisher.publish(output_msg)
    
    def run_inference(self):
        """Execute run_inference processing."""
        self.get_logger().debug(f'Executing run_inference')
        
        # TODO: Implement run_inference logic
        # This is where the actual processing logic would go
        
        self.processing_count += 1
    
    def timer_callback(self):
        """Periodic timer callback."""
        if not self.initialized:
            return
        
        # Periodic processing
        self.get_logger().debug(f'InferenceNode timer callback, count: {self.processing_count}')
        
        # Publish periodic status
        status_msg = String()
        status_msg.data = f'InferenceNode status: {self.processing_count}'
        self.output_data_publisher.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = InferenceNode()
    
    node.get_logger().info('Starting InferenceNode')
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 