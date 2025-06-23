"""Tests for the add-node command."""

import os
import stat
import pytest
from pathlib import Path
from click.testing import CliRunner
from robodsl.cli import main


def test_add_node_basic(tmp_path):
    """Test adding a basic node to a new project."""
    runner = CliRunner()
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Add a node
    result = runner.invoke(
        main,
        [
            "add-node", 
            "test_node", 
            "--language", "python",
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


def test_add_node_with_pubsub(tmp_path):
    """Test adding a node with publishers and subscribers."""
    runner = CliRunner()
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Add a node with publishers and subscribers
    result = runner.invoke(
        main,
        [
            "add-node",
            "sensor_processor",
            "--publisher", "/processed_data", "sensor_msgs/msg/Image",
            "--subscriber", "/raw_data", "sensor_msgs/msg/Image",
            "--language", "cpp",
            "--project-dir", str(project_dir)
        ]
    )
    
    assert result.exit_code == 0
    
    # Check robodsl file content
    with open(project_dir / "sensor_processor.robodsl", 'r') as f:
        content = f.read()
        assert "publisher /processed_data sensor_msgs/msg/Image" in content
        assert "subscriber /raw_data sensor_msgs/msg/Image" in content
        # Run the generator on the created robodsl file
        robodsl_file = project_dir / "sensor_processor.robodsl"
        result = runner.invoke(main, ["generate", str(robodsl_file), "--output-dir", str(project_dir)])
        assert result.exit_code == 0, result.output

        # Check C++ files were created
        # The default project name is 'robodsl_project' when not specified
        assert (project_dir / "include" / "robodsl_project" / "sensor_processor.hpp").exists()
        assert (project_dir / "src" / "sensor_processor.cpp").exists()



def test_add_node_to_existing_config(tmp_path):
    """Test adding a node when a robodsl config already exists."""
    runner = CliRunner()
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create an existing robodsl file
    with open(project_dir / "existing.robodsl", 'w') as f:
        f.write("node existing_node {\n}\n")
    
    # Add a new node
    result = runner.invoke(
        main,
        [
            "add-node", 
            "new_node",
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
        ["add-node", "test_node", "--project-dir", "/nonexistent/path"]
    )
    
    assert result.exit_code != 0
    assert "Error: Directory '/nonexistent/path' does not exist" in result.output or \
           "Error: Invalid value for '--project-dir': Directory '/nonexistent/path' does not exist." in result.output
