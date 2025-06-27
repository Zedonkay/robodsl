#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robodsl_package',
            executable='TestNode_node',
            name='TestNode_node',
            namespace='/TestNode',
        ),
    ])
