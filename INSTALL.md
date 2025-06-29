# RoboDSL Installation Guide

RoboDSL is a Domain-Specific Language (DSL) for GPU-accelerated robotics applications with ROS2 and CUDA. This guide covers installation for all supported platforms and feature sets.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Installation](#quick-installation)
- [Detailed Installation](#detailed-installation)
- [Feature-Specific Installation](#feature-specific-installation)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 2GB free space

### Required Software

#### For Basic Usage
- Python 3.8+
- pip (Python package installer)
- Git

#### For ROS2 Integration
- ROS2 Humble or newer
- CMake 3.15+
- C++ compiler (GCC 9+ or Clang 12+)

#### For CUDA Features
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0 or newer
- cuDNN 8.0 or newer

#### For ONNX/TensorRT Features
- ONNX Runtime 1.15+
- TensorRT 8.6+ (optional, for GPU optimization)
- OpenCV 4.8+

## Quick Installation

### One-Command Installation (Everything)

Since RoboDSL is not yet published to PyPI, you need to install it from source:

```bash
# Clone the repository
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl

# Install with everything (Linux/Windows with NVIDIA GPU)
pip install -e ".[all]"

# Install with development tools (macOS)
pip install -e ".[dev,docs]"
```

This installs:
- ✅ Core RoboDSL functionality
- ✅ CUDA support for GPU acceleration (Linux/Windows only)
- ✅ TensorRT for optimized inference (Linux/Windows only)
- ✅ ONNX Runtime for ML models
- ✅ Development tools (pytest, black, mypy, etc.)
- ✅ Documentation tools (Sphinx)

**⚠️ Platform Note**: The "all" option includes TensorRT and CUDA-specific packages that are only available on Linux/Windows with NVIDIA GPUs. 

**On macOS**, use this instead:
```bash
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl
pip install -e ".[dev,docs]"
```

This installs everything available on macOS:
- ✅ Core RoboDSL functionality
- ✅ ONNX Runtime for ML models
- ✅ Development tools (pytest, black, mypy, etc.)
- ✅ Documentation tools (Sphinx)

### Platform-Specific Installation

#### Linux/Windows (with NVIDIA GPU)
```bash
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl

# Install everything (recommended for development)
pip install -e ".[all]"

# Or install specific features
pip install -e ".[cuda]"      # CUDA support
pip install -e ".[tensorrt]"  # TensorRT optimization
pip install -e ".[dev]"       # Development tools
```

#### macOS
```bash
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl

# Install with development tools (recommended)
pip install -e ".[dev,docs]"

# Basic installation
pip install -e .

# With specific features (no CUDA/TensorRT on macOS)
pip install -e ".[dev]"       # Development tools
pip install -e ".[docs]"      # Documentation tools
```

### Using pip (When Published to PyPI)

Once RoboDSL is published to PyPI, you'll be able to install it directly:

```bash
# Basic installation
pip install robodsl

# With CUDA support
pip install robodsl[cuda]

# With TensorRT support
pip install robodsl[tensorrt]

# With all features
pip install robodsl[cuda,tensorrt]

# With EVERYTHING (all features + dev tools)
pip install robodsl[all]
```

### Using conda

```bash
# Create new environment
conda create -n robodsl python=3.10
conda activate robodsl

# Install RoboDSL
conda install -c conda-forge robodsl
```

### From Source

```bash
# Clone repository
git clone https://github.com/Zedonkay/robodsl.git
cd robodsl

# Install in development mode
pip install -e ".[dev]"
```

## Detailed Installation

### Ubuntu/Debian

1. **Install system dependencies**:
   ```bash
   sudo apt update
   sudo apt install python3-pip python3-venv build-essential cmake
   ```

2. **Install CUDA (if using GPU features)**:
   ```bash
   # Follow NVIDIA's official CUDA installation guide
   # https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
   ```

3. **Install ROS2 (if using ROS2 features)**:
   ```bash
   # Follow ROS2 installation guide
   # https://docs.ros.org/en/humble/Installation.html
   ```

4. **Install RoboDSL**:
   ```bash
   pip install robodsl[cuda]
   ```

### macOS

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python and dependencies**:
   ```bash
   brew install python@3.10 cmake
   ```

3. **Install RoboDSL**:
   ```bash
   pip install robodsl
   ```

### Windows

1. **Install Python** from [python.org](https://www.python.org/downloads/)

2. **Install Visual Studio Build Tools**:
   - Download from Microsoft Visual Studio
   - Install C++ build tools

3. **Install CUDA** (if using GPU features):
   - Download from NVIDIA's website
   - Follow installation guide

4. **Install RoboDSL**:
   ```bash
   pip install robodsl
   ```

## Feature-Specific Installation

### CUDA Support

CUDA support enables GPU-accelerated code generation and execution.

```bash
# Install with CUDA support
pip install robodsl[cuda]

# Verify CUDA installation
python -c "import cupy; print('CUDA available:', cupy.cuda.is_available())"
```

**Requirements**:
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+
- Compatible GPU drivers

### ONNX Runtime Support

ONNX Runtime enables machine learning model integration.

```bash
# Install with ONNX support
pip install robodsl

# For GPU acceleration with ONNX
pip install robodsl[cuda]
```

**Features**:
- Model loading and inference
- GPU acceleration via CUDA
- TensorRT optimization support

### TensorRT Support

TensorRT provides additional GPU optimizations for inference.

```bash
# Install with TensorRT support
pip install robodsl[tensorrt]
```

**Requirements**:
- NVIDIA GPU
- TensorRT 8.6+
- Compatible CUDA version

### ROS2 Integration

ROS2 integration enables robotics-specific features.

```bash
# Install ROS2 first (see ROS2 documentation)
# Then install RoboDSL
pip install robodsl
```

**Features**:
- ROS2 node generation
- Topic and service integration
- Launch file generation
- QoS configuration

## Development Setup

### Prerequisites

- Git
- Python 3.8+
- Virtual environment tool

### Setup Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Zedonkay/robodsl.git
   cd robodsl
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

5. **Run tests**:
   ```bash
   pytest
   ```

### Development Features

- **Code formatting**: Black and isort
- **Linting**: flake8 and mypy
- **Testing**: pytest with coverage
- **Documentation**: Sphinx
- **Pre-commit hooks**: Automated code quality checks

## Verification

### Basic Installation

```bash
# Check installation
robodsl --version

# Test basic functionality
robodsl --help
```

### CUDA Support

```bash
# Test CUDA availability
python -c "
import robodsl
print('RoboDSL version:', robodsl.__version__)
try:
    import cupy
    print('CUDA available:', cupy.cuda.is_available())
except ImportError:
    print('CUDA not available')
"
```

### ONNX Support

```bash
# Test ONNX Runtime
python -c "
import onnxruntime as ort
print('ONNX Runtime version:', ort.__version__)
print('Available providers:', ort.get_available_providers())
"
```

## Troubleshooting

### Common Issues

#### CUDA Installation Issues

**Problem**: CUDA not detected
```bash
# Check CUDA installation
nvidia-smi
nvcc --version

# Reinstall with specific CUDA version
pip uninstall robodsl
pip install robodsl[cuda]
```

**Problem**: cuDNN not found
```bash
# Install cuDNN from NVIDIA
# Add to PATH or install via conda
conda install cudnn
```

#### ONNX Runtime Issues

**Problem**: ONNX Runtime not found
```bash
# Reinstall ONNX Runtime
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu
```

#### ROS2 Integration Issues

**Problem**: ROS2 not detected
```bash
# Source ROS2 environment
source /opt/ros/humble/setup.bash

# Check ROS2 installation
ros2 --version
```

### Platform-Specific Issues

#### Ubuntu/Debian

**Problem**: Missing system libraries
```bash
sudo apt install libopencv-dev libonnxruntime-dev
```

#### macOS

**Problem**: OpenCV installation issues
```bash
brew install opencv
pip install opencv-python
```

#### Windows

**Problem**: Build tools not found
```bash
# Install Visual Studio Build Tools
# Ensure PATH includes build tools
```

### Getting Help

- **Documentation**: [https://robodsl.readthedocs.io](https://robodsl.readthedocs.io)
- **Issues**: [GitHub Issues](https://github.com/Zedonkay/robodsl/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Zedonkay/robodsl/discussions)

## Advanced Configuration

### Environment Variables

```bash
# CUDA configuration
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=1

# ONNX Runtime configuration
export ORT_DISABLE_ALL=0
export ORT_ENABLE_ALL=1

# RoboDSL configuration
export ROBODSL_LOG_LEVEL=INFO
export ROBODSL_CACHE_DIR=/tmp/robodsl
```

### Configuration Files

Create `~/.robodsl/config.yaml`:
```yaml
cuda:
  device_id: 0
  memory_fraction: 0.8

onnx:
  execution_provider: CUDAExecutionProvider
  optimization_level: 1

templates:
  output_dir: ./generated
  cache_dir: ~/.robodsl/cache
```

## Uninstallation

```bash
# Remove RoboDSL
pip uninstall robodsl

# Remove all dependencies
pip uninstall robodsl lark onnxruntime opencv-python numpy

# Clean up cache
rm -rf ~/.robodsl
```

## License

RoboDSL is licensed under the MIT License. See [LICENSE](LICENSE) for details. 