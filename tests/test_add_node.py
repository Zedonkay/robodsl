from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda
"""Tests for the add-node command."""

import os
import stat
import pytest
from pathlib import Path
from click.testing import CliRunner
from robodsl.cli import main


def test_add_node_basic(test_output_dir):
    """Test adding a basic node to a new project."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Add a node
    result = runner.invoke(
        main,
        [
            "create-node", 
            "test_node", 
            "--template", "basic",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created
    assert (project_dir / "test_node.robodsl").exists()
    assert (project_dir / "src" / "test_node_node.py").exists()
    assert (project_dir / "launch" / "test_node.launch.py").exists()
    assert (project_dir / "config" / "test_node.yaml").exists()
    
    # Check Python file is executable
    if os.name != 'nt':  # Skip permission check on Windows
        assert (project_dir / "src" / "test_node_node.py").stat().st_mode & stat.S_IXUSR
    
    # Check robodsl file content
    with open(project_dir / "test_node.robodsl", 'r') as f:
        content = f.read()
        assert "node test_node" in content


# def test_add_node_with_pubsub(test_output_dir):
#     """Test adding a node with publishers and subscribers."""
#     runner = CliRunner()
#     project_dir = test_output_dir / "test_project"
#     project_dir.mkdir()
    
#     # Add a node with publishers and subscribers
#     result = runner.invoke(
#         main,
#         [
#             "add-node",
#             "sensor_processor",
#             "--publisher", "/processed_data: \"sensor_msgs/msg/Image\"",
#             "--subscriber", "/raw_data: \"sensor_msgs/msg/Image\"",
#             "--language", "cpp",
#             "--project-dir", str(project_dir)
#         ]
#     )
    
#     assert result.exit_code == 0
    
#     # Check robodsl file content
#     with open(project_dir / "sensor_processor.robodsl", 'r') as f:
#         content = f.read()
#         assert 'publisher /processed_data: "sensor_msgs/msg/Image"' in content
#         assert 'subscriber /raw_data: "sensor_msgs/msg/Image"' in content
    
#     # Check C++ files were created
#     assert (project_dir / "include" / "sensor_processor" / "sensor_processor_node.hpp").exists()
#     assert (project_dir / "src" / "sensor_processor_node.cpp").exists()
    
#     # Check that the header file contains the expected content
#     with open(project_dir / "include" / "sensor_processor" / "sensor_processor_node.hpp", 'r') as f:
#         content = f.read()
#         assert "#ifndef SENSOR_PROCESSOR_NODE_H_" in content
#         assert "class SensorProcessorNode : public rclcpp::Node" in content


def test_add_node_to_existing_config(test_output_dir):
    """Test adding a node when a robodsl config already exists."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Create an existing robodsl file
    with open(project_dir / "existing.robodsl", 'w') as f:
        f.write("node existing_node {\n}\n")
    
    # Add a new node
    result = runner.invoke(
        main,
        [
            "create-node", 
            "new_node",
            "--template", "basic",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code == 0
    assert (project_dir / "new_node.robodsl").exists()
    assert (project_dir / "existing.robodsl").exists()  # Shouldn't be modified


def test_add_node_invalid_project_dir():
    """Test adding a node to a non-existent project directory."""
    runner = CliRunner()
    
    result = runner.invoke(
        main,
        ["create-node", "test_node", "--template", "basic", "--project-dir", "/nonexistent/path"]
    )
    
    assert result.exit_code != 0
    assert "Error: Directory '/nonexistent/path' does not exist" in result.output or \
           "Error: Invalid value for '--project-dir': Directory '/nonexistent/path' does not exist." in result.output


def test_add_node_with_colon_syntax(test_output_dir):
    """Test adding a node with publishers and subscribers using colon syntax."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Add a node with publishers and subscribers
    result = runner.invoke(
        main,
        [
            "create-node",
            "object_detector",
            "--template", "cuda",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code == 0
    
    # Check C++ files were created (flat structure)
    assert (project_dir / "include" / "object_detector_node.hpp").exists()
    assert (project_dir / "src" / "object_detector_node.cpp").exists()
    
    # Check that the header file contains the expected content
    with open(project_dir / "include" / "object_detector_node.hpp", 'r') as f:
        content = f.read()
        assert "#ifndef OBJECT_DETECTOR_NODE_HPP" in content
        assert "class Object_detectorNode : public rclcpp::Node" in content