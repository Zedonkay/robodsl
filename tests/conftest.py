"""Pytest configuration and fixtures for RoboDSL tests."""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from robodsl.generators.main_generator import MainGenerator
from robodsl.parsers.lark_parser import RoboDSLParser


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