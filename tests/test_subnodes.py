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
            "add-node",
            "sensors.camera",
            "--language", "python",
            "--project-dir", str(project_dir)
            # No publishers or subscribers for this test
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created in the correct locations
    assert (project_dir / "robodsl" / "nodes" / "sensors" / "camera.robodsl").exists()
    assert (project_dir / "src" / "sensors" / "camera_node.py").exists()
    assert (project_dir / "launch" / "sensors" / "camera.launch.py").exists()
    
    # Check Python file is executable
    if os.name != 'nt':
        assert (project_dir / "src" / "sensors" / "camera_node.py").stat().st_mode & stat.S_IXUSR
    
    # Check robodsl file content
    with open(project_dir / "robodsl" / "nodes" / "sensors" / "camera.robodsl", 'r') as f:
        content = f.read()
        assert "node camera" in content


def test_add_nested_subnode(test_output_dir):
    """Test adding a deeply nested subnode."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Add a deeply nested subnode
    result = runner.invoke(
        main,
        [
            "add-node",
            "robot.sensors.camera.depth",
            "--language", "cpp",
            "--project-dir", str(project_dir)
            # No publishers or subscribers for this test
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created in the correct locations
    assert (project_dir / "robodsl" / "nodes" / "robot" / "sensors" / "camera" / "depth.robodsl").exists()
    assert (project_dir / "src" / "robot" / "sensors" / "camera" / "depth_node.cpp").exists()
    assert (project_dir / "include" / "robot" / "sensors" / "camera" / "depth_node.hpp").exists()
    assert (project_dir / "launch" / "robot" / "sensors" / "camera" / "depth.launch.py").exists()


def test_add_subnode_with_pubsub(test_output_dir):
    """Test adding a subnode (without deprecated pubsub options)."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Add a subnode (without deprecated publisher/subscriber options)
    result = runner.invoke(
        main,
        [
            "add-node",
            "perception.object_detector",
            "--language", "python",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code == 0
    
    # Check that files were created in the correct locations
    assert (project_dir / "robodsl" / "nodes" / "perception" / "object_detector.robodsl").exists()
    assert (project_dir / "src" / "perception" / "object_detector_node.py").exists()
    assert (project_dir / "launch" / "perception" / "object_detector.launch.py").exists()
    
    # Check robodsl file content
    with open(project_dir / "robodsl" / "nodes" / "perception" / "object_detector.robodsl", 'r') as f:
        content = f.read()
        assert "node object_detector" in content


def test_invalid_node_name(test_output_dir):
    """Test adding a node with an invalid name."""
    runner = CliRunner()
    project_dir = test_output_dir / "test_project"
    project_dir.mkdir()
    
    # Try to add a node with invalid characters
    result = runner.invoke(
        main,
        [
            "add-node",
            "invalid-node@name",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code != 0
    assert "Invalid node name" in result.output
