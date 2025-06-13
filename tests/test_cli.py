"""Tests for the RoboDSL CLI."""

import os
import tempfile
from pathlib import Path
from click.testing import CliRunner
from robodsl.cli import main

def test_cli_help():
    """Test the CLI help command."""
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output

def test_cli_init():
    """Test the init command."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = runner.invoke(
            main, ["init", "test_project", "--output-dir", tmpdir]
        )
        assert result.exit_code == 0
        
        project_dir = Path(tmpdir) / "test_project"
        assert (project_dir / "test_project.robodsl").exists()
        assert (project_dir / "src").is_dir()
        assert (project_dir / "include").is_dir()
        assert (project_dir / "launch").is_dir()
        assert (project_dir / "config").is_dir()
        assert (project_dir / "robodsl").is_dir()

def test_cli_init_existing_dir():
    """Test init command with existing directory."""
    runner = CliRunner()
    with tempfile.TemporaryDirectory() as tmpdir:
        # First run should succeed
        result = runner.invoke(
            main, ["init", "test_project", "--output-dir", tmpdir]
        )
        assert result.exit_code == 0
        
        # Second run should fail
        result = runner.invoke(
            main, ["init", "test_project", "--output-dir", tmpdir]
        )
        assert result.exit_code == 1
        assert "already exists" in result.output
