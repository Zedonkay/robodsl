"""ONNX Integration Generator for RoboDSL.

This module generates pure C++/CUDA ONNX Runtime integration code for ROS2 nodes,
ensuring all inference is compiled and not interpreted.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from jinja2 import Template, Environment, FileSystemLoader

from ..core.ast import OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode


class OnnxIntegrationGenerator:
    """Generates pure C++/CUDA ONNX Runtime integration code."""
    
    def __init__(self, output_dir: str = "generated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent.parent / "templates" / "onnx"
        self.env = Environment(loader=FileSystemLoader(str(template_dir)))
    
    def generate_onnx_integration(self, model: OnnxModelNode, node_name: str) -> Dict[str, str]:
        """Generate pure C++/CUDA ONNX integration code for a model."""
        generated_files = {}
        
        # Generate header file
        header_content = self._generate_header(model, node_name)
        header_path = self.output_dir / f"{node_name}_onnx.hpp"
        generated_files[str(header_path)] = header_content
        
        # Generate implementation file
        impl_content = self._generate_implementation(model, node_name)
        impl_path = self.output_dir / f"{node_name}_onnx.cpp"
        generated_files[str(impl_path)] = impl_content
        
        return generated_files
    
    def _generate_header(self, model: OnnxModelNode, node_name: str) -> str:
        """Generate C++ header file for ONNX integration."""
        template = self.env.get_template("header.hpp.jinja2")
        
        return template.render(
            node_name=node_name,
            model=model
        )
    
    def _generate_implementation(self, model: OnnxModelNode, node_name: str) -> str:
        """Generate C++ implementation file for ONNX integration."""
        template = self.env.get_template("implementation.cpp.jinja2")
        
        # Prepare template variables
        device_type = model.config.device.device if model.config.device else 'cpu'
        optimizations = model.config.optimizations if model.config.optimizations else []
        optimization_names = [opt.optimization for opt in optimizations]
        
        return template.render(
            node_name=node_name,
            device_type=device_type,
            optimizations=optimizations,
            optimization_names=optimization_names
        )
    
    def generate_cmake_integration(self, model: OnnxModelNode, node_name: str) -> str:
        """Generate CMake configuration for ONNX integration."""
        template_str = """
# ONNX Runtime integration for {{ node_name }} (Pure C++/CUDA)

# Find ONNX Runtime
find_package(ONNXRuntime REQUIRED)

# Find OpenCV (for image preprocessing)
find_package(OpenCV REQUIRED)

# Add ONNX Runtime include directories
target_include_directories({{ node_name }} PRIVATE
    ${ONNXRuntime_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

# Link ONNX Runtime libraries
target_link_libraries({{ node_name }} PRIVATE
    ${ONNXRuntime_LIBRARIES}
    ${OpenCV_LIBS}
)

# Copy ONNX model file to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/models/{{ model.name }}.onnx
    ${CMAKE_CURRENT_BINARY_DIR}/{{ model.name }}.onnx
    COPYONLY
)

# Set model path for runtime
target_compile_definitions({{ node_name }} PRIVATE
    ONNX_MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/{{ model.name }}.onnx"
)

# Ensure C++17 standard for ONNX Runtime
set_target_properties({{ node_name }} PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
"""
        
        template = Template(template_str)
        return template.render(
            node_name=node_name,
            model=model
        )
    
    def generate_node_integration(self, model: OnnxModelNode, node_name: str) -> str:
        """Generate C++ code to integrate ONNX model into ROS2 node."""
        template_str = """
// ONNX Model Integration for {{ node_name }} Node
// This code integrates the {{ model.name }} ONNX model into the {{ node_name }} ROS2 node

#include "{{ node_name }}_onnx.hpp"
#include <memory>
#include <string>

class {{ node_name }}Node : public rclcpp::Node {
private:
    // ONNX inference engine
    std::unique_ptr<{{ node_name }}OnnxInference> onnx_inference_;
    
    // Model configuration
    std::string model_path_;
    
    // ROS2 components (publishers, subscribers, etc.)
    // Add your ROS2 components here
    
public:
    {{ node_name }}Node() : Node("{{ node_name }}") {
        // Initialize model path
        this->declare_parameter("model_path", "{{ model.name }}.onnx");
        model_path_ = this->get_parameter("model_path").as_string();
        
        // Initialize ONNX inference engine
        onnx_inference_ = std::make_unique<{{ node_name }}OnnxInference>(model_path_);
        
        if (!onnx_inference_->initialize()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize ONNX model: %s", model_path_.c_str());
            throw std::runtime_error("ONNX model initialization failed");
        }
        
        RCLCPP_INFO(this->get_logger(), "ONNX model initialized successfully: %s", model_path_.c_str());
        
        // Initialize ROS2 components
        initialize_ros_components();
    }
    
    ~{{ node_name }}Node() = default;

private:
    void initialize_ros_components() {
        // Initialize your ROS2 publishers, subscribers, timers, etc.
        // Example:
        // image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        //     "/camera/image_raw", 10,
        //     std::bind(&{{ node_name }}Node::image_callback, this, std::placeholders::_1));
        // 
        // result_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
        //     "/classification/result", 10);
    }
    
    // Example callback method using ONNX inference
    void process_with_onnx(const std::vector<float>& input_data) {
        std::vector<float> output_data;
        
        if (onnx_inference_->run_inference(input_data, output_data)) {
            // Process the output data
            RCLCPP_INFO(this->get_logger(), "Inference completed successfully");
            
            // Publish results or perform further processing
            // publish_results(output_data);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Inference failed");
        }
    }
    
    // Add your specific callback methods here
    // void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
    //     // Convert image to tensor format
    //     std::vector<float> input_tensor = preprocess_image(msg);
    //     
    //     // Run ONNX inference
    //     process_with_onnx(input_tensor);
    // }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    
    try {
        auto node = std::make_shared<{{ node_name }}Node>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("{{ node_name }}"), "Exception in main: %s", e.what());
        return 1;
    }
    
    rclcpp::shutdown();
    return 0;
}
"""
        
        template = Template(template_str)
        return template.render(
            node_name=node_name,
            model=model
        )

    def generate_python_integration(self, model: OnnxModelNode, node_name: str) -> str:
        """Generate Python integration code for ONNX model."""
        template_str = """
# Python ONNX Integration for {{ node_name }}
# This provides Python bindings for the {{ model.name }} ONNX model

import numpy as np
import onnxruntime as ort
from typing import List, Optional, Tuple
import logging

class {{ node_name }}OnnxPython:
    \"\"\"Python wrapper for {{ node_name }} ONNX model.\"\"\"
    
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model_path = model_path
        self.device = device
        self.session = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize ONNX Runtime session
        self._initialize_session()
    
    def _initialize_session(self):
        \"\"\"Initialize ONNX Runtime session with appropriate providers.\"\"\"
        try:
            providers = []
            if self.device == "cuda":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.logger.info(f"ONNX model loaded successfully: {{ model.name }}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def preprocess_input(self, input_data: np.ndarray) -> np.ndarray:
        \"\"\"Preprocess input data for the model.\"\"\"
        # Add your preprocessing logic here
        # Example: normalize, resize, convert format, etc.
        return input_data
    
    def postprocess_output(self, output_data: np.ndarray) -> np.ndarray:
        \"\"\"Postprocess output data from the model.\"\"\"
        # Add your postprocessing logic here
        # Example: apply softmax, convert to probabilities, etc.
        return output_data
    
    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        \"\"\"Run inference on the ONNX model.\"\"\"
        try:
            # Preprocess input
            processed_input = self.preprocess_input(input_data)
            
            # Get input/output names
            input_name = self.session.get_inputs()[0].name
            output_name = self.session.get_outputs()[0].name
            
            # Run inference
            outputs = self.session.run([output_name], {input_name: processed_input})
            
            # Postprocess output
            result = self.postprocess_output(outputs[0])
            
            return result
            
        except Exception as e:
            self.logger.error(f"Inference failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        \"\"\"Get information about the model inputs and outputs.\"\"\"
        info = {
            "model_name": "{{ model.name }}",
            "device": self.device,
            "inputs": [],
            "outputs": []
        }
        
        for input_info in self.session.get_inputs():
            info["inputs"].append({
                "name": input_info.name,
                "shape": input_info.shape,
                "type": str(input_info.type)
            })
        
        for output_info in self.session.get_outputs():
            info["outputs"].append({
                "name": output_info.name,
                "shape": output_info.shape,
                "type": str(output_info.type)
            })
        
        return info

# Example usage
if __name__ == "__main__":
    # Initialize the model
    model = {{ node_name }}OnnxPython("{{ model.name }}.onnx", device="cpu")
    
    # Get model information
    info = model.get_model_info()
    print(f"Model info: {info}")
    
    # Example inference
    # input_data = np.random.random((1, 3, 224, 224)).astype(np.float32)
    # result = model.run_inference(input_data)
    # print(f"Inference result shape: {result.shape}")
"""
        
        template = Template(template_str)
        return template.render(
            node_name=node_name,
            model=model
        ) 