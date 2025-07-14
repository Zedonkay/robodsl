# RoboDSL

A Domain-Specific Language (DSL) for GPU-accelerated robotics applications with ROS2 and CUDA.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://robodsl.readthedocs.io)

## Overview

RoboDSL simplifies the development of GPU-accelerated robotics applications by providing a high-level domain-specific language that automatically generates optimized C++/CUDA code with ROS2 integration.

### Key Features

- ** GPU Acceleration**: Native CUDA kernel generation and optimization
- ** ROS2 Integration**: Automatic ROS2 node generation with QoS configuration
- ** ML Model Support**: ONNX Runtime integration for machine learning models
- ** Pipeline Orchestration**: Multi-stage processing pipeline generation
- ** Image Processing**: OpenCV integration for computer vision tasks
- ** High Performance**: Optimized for real-time robotics applications

### New Features in v0.1.0

- **Lark Parser**: Robust grammar parsing with error recovery
- **ONNX Integration**: Complete ML model inference pipeline
- **TensorRT Support**: GPU optimization for inference
- **Pipeline DSL**: Multi-stage processing workflows
- **Semantic Analysis**: Advanced error checking and validation
- **Code Generation**: Template-based C++/CUDA/ROS2 code generation

## Quick Start

### Installation

Since RoboDSL is not yet published to PyPI, you need to install it from source:

```bash
# Clone the repository
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl

# Install with everything (Linux/Windows with NVIDIA GPU)
pip install -e ".[all]"

# Install with development tools (macOS)
pip install -e ".[dev,docs]"

# Basic installation
pip install -e .
```

### Platform-Specific Installation

#### Linux/Windows (with NVIDIA GPU)
```bash
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl
pip install -e ".[all]"  # Install everything
```

#### macOS
```bash
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl
pip install -e ".[dev,docs]"  # Install with development tools
```

**⚠️ Note**: RoboDSL is currently in development and not yet available on PyPI. Install from source for the latest features.

### Example Usage

```robodsl
// Define CUDA kernels
cuda_kernels {
    kernel preprocess {
        input: uint8* raw_image, int width, int height
        output: float* normalized_image
        code: {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            int total_pixels = width * height * 3;
            if (idx < total_pixels) {
                normalized_image[idx] = (float)raw_image[idx] / 255.0f;
            }
        }
    }
}

// Define ONNX model
onnx_model detector {
    config {
        input: "images" -> "float32"
        output: "output0" -> "float32"
        device: gpu
        optimization: tensorrt
    }
}

// Define processing pipeline
pipeline ml_detection_pipeline {
    stage preprocessing {
        input: "camera_image"
        output: "normalized_image"
        method: "normalize_image"
        cuda_kernel: "preprocess"
        topic: /preprocessing/image
    }
    
    stage detection {
        input: "normalized_image"
        output: "detection_results"
        method: "detect_objects"
        onnx_model: "detector"
        topic: /detection/results
    }
}

// Define ROS2 node
node image_processor {
    subscriber /camera/image_raw: "sensor_msgs/msg/Image"
    publisher /detection/results: "std_msgs/msg/Float32MultiArray"
    
    parameter model_path: "yolo_detector.onnx"
    parameter confidence_threshold: 0.5
    
    lifecycle {
        auto_start: true
        auto_shutdown: false
    }
    
    qos {
        reliability: reliable
        durability: transient_local
        history: keep_last
        depth: 10
    }
}
```

Generate the code:

```bash
robodsl generate example.robodsl --output ./generated
```

## Documentation

- **[Installation Guide](INSTALL.md)**: Comprehensive installation instructions
- **[User Guide](docs/user-guide/)**: Getting started and tutorials
- **[DSL Specification](docs/developer/dsl-specification.html)**: Language reference
- **[API Documentation](docs/developer/)**: Developer documentation
- **[Examples](examples/)**: Code examples and templates

## Features

### Core Language Features

- **Node Definitions**: ROS2 node configuration with publishers, subscribers, and parameters
- **CUDA Kernels**: GPU kernel definitions with block/grid size configuration
- **ONNX Models**: Machine learning model integration with device and optimization settings
- **Pipelines**: Multi-stage processing workflows with data flow between stages
- **QoS Configuration**: ROS2 Quality of Service settings
- **Lifecycle Management**: ROS2 lifecycle node configuration

### Code Generation

- **C++ Nodes**: ROS2-compatible C++ node generation
- **CUDA Integration**: GPU kernel compilation and memory management
- **ONNX Runtime**: Model loading and inference code
- **CMake Build**: Complete build system generation
- **Launch Files**: ROS2 launch file generation
- **Documentation**: Auto-generated API documentation

### Development Tools

- **Semantic Analysis**: Error checking and validation
- **Template System**: Jinja2-based code generation
- **Testing Framework**: Comprehensive test suite
- **IDE Support**: Language server and syntax highlighting
- **CLI Tools**: Command-line interface for code generation

## System Requirements

- **Python**: 3.8 or higher
- **OS**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **GPU**: NVIDIA GPU with CUDA support (for GPU features)
- **ROS2**: Humble or newer (for ROS2 features)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/developer/contributing.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://robodsl.readthedocs.io](https://robodsl.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Zedonkay/robodsl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zedonkay/robodsl/discussions)

## Acknowledgments

- ROS2 community for the robotics framework
- NVIDIA for CUDA and TensorRT
- Microsoft for ONNX Runtime
- OpenCV community for computer vision tools 
