"""
RoboDSL - A DSL for GPU-accelerated robotics applications with ROS2 and CUDA.
"""

__version__ = "0.1.0"

# Import main components
from .core import *
from .parsers import *
from .generators import *
from .cli import main

__all__ = [
    'main',
    'core',
    'parsers', 
    'generators',
    'templates',
    'utils'
]
