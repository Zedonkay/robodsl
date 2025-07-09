"""Test configuration and utilities."""

import os
import shutil
import pytest
import tempfile
from pathlib import Path
from robodsl.generators.main_generator import MainGenerator
from robodsl.parsers.lark_parser import RoboDSLParser


def has_ros2():
    """Check if ROS2 is available in the environment."""
    return shutil.which('ros2') is not None or 'AMENT_PREFIX_PATH' in os.environ


def has_cuda():
    """Check if CUDA is available in the environment."""
    return shutil.which('nvcc') is not None


def has_tensorrt():
    """Check if TensorRT is available in the environment."""
    # Check for TensorRT by looking for common installation paths or environment variables
    tensorrt_paths = [
        '/usr/local/tensorrt',
        '/opt/tensorrt',
        '/usr/lib/x86_64-linux-gnu/tensorrt',
        '/usr/lib/aarch64-linux-gnu/tensorrt'
    ]
    
    # Check if any TensorRT path exists
    for path in tensorrt_paths:
        if os.path.exists(path):
            return True
    
    # Check environment variables
    if 'TENSORRT_ROOT' in os.environ or 'TENSORRT_PATH' in os.environ:
        return True
    
    # Check if TensorRT libraries are available
    try:
        import ctypes
        ctypes.CDLL('libnvinfer.so')
        return True
    except (OSError, ImportError):
        pass
    
    return False


def has_onnx():
    """Check if ONNX Runtime is available in the environment."""
    try:
        import onnxruntime
        return True
    except ImportError:
        return False


def skip_if_no_ros2():
    """Skip test if ROS2 is not available."""
    if not has_ros2():
        pytest.skip("Skipping test: ROS2 not available.")


def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    if not has_cuda():
        pytest.skip("Skipping test: CUDA not available.")


def skip_if_no_tensorrt():
    """Skip test if TensorRT is not available."""
    if not has_tensorrt():
        pytest.skip("Skipping test: TensorRT not available.")


def skip_if_no_onnx():
    """Skip test if ONNX Runtime is not available."""
    if not has_onnx():
        pytest.skip("Skipping test: ONNX Runtime not available.")


@pytest.fixture
def test_output_dir():
    """Create a test output directory for each test."""
    # Create test_output directory if it doesn't exist
    test_output_base = Path("test_output")
    test_output_base.mkdir(exist_ok=True)
    
    # Create a unique subdirectory for this test
    with tempfile.TemporaryDirectory(dir=test_output_base) as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def parser():
    """Provide a RoboDSL parser instance."""
    return RoboDSLParser()


@pytest.fixture
def generator(test_output_dir):
    """Provide a MainGenerator instance with test output directory."""
    return MainGenerator(output_dir=str(test_output_dir))


@pytest.fixture
def generator_with_custom_output():
    """Provide a MainGenerator factory that accepts custom output directory."""
    def _create_generator(output_dir):
        return MainGenerator(output_dir=output_dir)
    return _create_generator


def pytest_configure(config):
    """Configure pytest to ensure test_output directory exists."""
    test_output_dir = Path("test_output")
    test_output_dir.mkdir(exist_ok=True)


def pytest_unconfigure(config):
    """Clean up after pytest finishes."""
    # Optionally clean up test_output directory
    # Uncomment the following lines if you want to clean up after all tests
    # test_output_dir = Path("test_output")
    # if test_output_dir.exists():
    #     shutil.rmtree(test_output_dir) 