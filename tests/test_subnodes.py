from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, skip_if_no_cuda
"""Tests for subnode functionality in RoboDSL."""

import os
import stat
import pytest
from pathlib import Path
from click.testing import CliRunner
from robodsl.cli import main


def test_add_subnode_basic(test_output_dir):
    """Test adding a basic subnode to a project."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Add a subnode
    result = runner.invoke(
        main,
        [
            "create-node",
            "sensors.camera",
            "--template", "basic",
            "--project-dir", str(project_dir)
            # No publishers or subscribers for this test
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created in the correct locations
    # RoboDSL files are organized in nested directories for organization
    assert (project_dir / "robodsl" / "nodes" / "sensors" / "camera.robodsl").exists()
    # Source files are in flat structure (subnodes are CLI-only for organization)
    assert (project_dir / "src" / "camera_node.py").exists()
    # Launch files are in flat structure
    assert (project_dir / "launch" / "camera.launch.py").exists()
    
    # Check Python file is executable
    if os.name != 'nt':
        assert (project_dir / "src" / "camera_node.py").stat().st_mode & stat.S_IXUSR
    
    # Check robodsl file content - node name should be simple (no dots)
    with open(project_dir / "robodsl" / "nodes" / "sensors" / "camera.robodsl", 'r') as f:
        content = f.read()
        assert "node camera" in content
        # Ensure no dots in the actual node definition
        assert "node sensors.camera" not in content


def test_add_nested_subnode(test_output_dir):
    """Test adding a deeply nested subnode."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Add a deeply nested subnode
    result = runner.invoke(
        main,
        [
            "create-node",
            "robot.sensors.camera.depth",
            "--template", "cuda",
            "--project-dir", str(project_dir)
            # No publishers or subscribers for this test
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created in the correct locations
    # RoboDSL files are organized in nested directories for organization
    assert (project_dir / "robodsl" / "nodes" / "robot" / "sensors" / "camera" / "depth.robodsl").exists()
    # Source files are in flat structure (subnodes are CLI-only for organization)
    assert (project_dir / "src" / "depth_node.cpp").exists()
    assert (project_dir / "include" / "depth_node.hpp").exists()
    # Launch files are in flat structure
    assert (project_dir / "launch" / "depth.launch.py").exists()


def test_add_subnode_with_pubsub(test_output_dir):
    """Test adding a subnode (without deprecated pubsub options)."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Add a subnode (without deprecated publisher/subscriber options)
    result = runner.invoke(
        main,
        [
            "create-node",
            "perception.object_detector",
            "--template", "basic",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created in the correct locations
    # RoboDSL files are organized in nested directories for organization
    assert (project_dir / "robodsl" / "nodes" / "perception" / "object_detector.robodsl").exists()
    # Source files are in flat structure (subnodes are CLI-only for organization)
    assert (project_dir / "src" / "object_detector_node.py").exists()
    # Launch files are in flat structure
    assert (project_dir / "launch" / "object_detector.launch.py").exists()
    
    # Check robodsl file content - node name should be simple (no dots)
    with open(project_dir / "robodsl" / "nodes" / "perception" / "object_detector.robodsl", 'r') as f:
        content = f.read()
        assert "node object_detector" in content
        # Ensure no dots in the actual node definition
        assert "node perception.object_detector" not in content


def test_invalid_node_name(test_output_dir):
    """Test adding a node with an invalid name."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Try to add a node with invalid characters
    result = runner.invoke(
        main,
        [
            "create-node",
            "invalid-node@name",
            "--template", "basic",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code != 0
    assert "Invalid node name" in result.output


def test_subnode_syntax_rejected_in_robodsl_code(test_output_dir):
    """Test that subnode syntax (with dots) is rejected in robodsl code."""
    from robodsl.parsers import parse_robodsl
    
    # Create a robodsl file with subnode syntax
    robodsl_content = """
node sensors.camera {
    parameter int count = 0
}
"""
    
    # This should raise an error with a clear message
    try:
        parse_robodsl(robodsl_content)
        assert False, "Should have raised an error for subnode syntax"
    except Exception as e:
        error_msg = str(e)
        assert "Subnodes with dots (.) are not allowed in RoboDSL code" in error_msg
        assert "CLI-only feature" in error_msg
        assert "robodsl create-node" in error_msg