"""Simulation Configuration Generator for RoboDSL.

This module generates simulation launch files and configuration from RoboDSL AST.
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..core.ast import (
    SimulationConfigNode, SimulationWorldNode, SimulationRobotNode,
    SimulationPluginNode, HardwareInLoopNode
)


class SimulationGenerator:
    """Generates simulation launch files and configuration."""
    
    def __init__(self, output_dir: str = "launch"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_simulation_launch(self, simulation: SimulationConfigNode) -> str:
        """Generate launch file for simulation."""
        if simulation.simulator == "gazebo":
            return self._generate_gazebo_launch(simulation)
        elif simulation.simulator == "isaac_sim":
            return self._generate_isaac_launch(simulation)
        elif simulation.simulator == "gym":
            return self._generate_gym_launch(simulation)
        else:
            raise ValueError(f"Unsupported simulator: {simulation.simulator}")
    
    def generate_hil_launch(self, hil_config: HardwareInLoopNode) -> str:
        """Generate hardware-in-the-loop launch file."""
        return self._generate_hil_launch_content(hil_config)
    
    def write_simulation_launch(self, simulation: SimulationConfigNode, filename: str = "simulation_launch.py") -> str:
        """Generate and write simulation launch file to disk."""
        content = self.generate_simulation_launch(simulation)
        file_path = self.output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return str(file_path)
    
    def write_hil_launch(self, hil_config: HardwareInLoopNode, filename: str = "hil_launch.py") -> str:
        """Generate and write hardware-in-the-loop launch file to disk."""
        content = self.generate_hil_launch(hil_config)
        file_path = self.output_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return str(file_path)
    
    def _generate_gazebo_launch(self, simulation: SimulationConfigNode) -> str:
        """Generate Gazebo launch file."""
        lines = [
            "#!/usr/bin/env python3",
            "",
            "import os",
            "from launch import LaunchDescription",
            "from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription",
            "from launch.substitutions import LaunchConfiguration, PathJoinSubstitution",
            "from launch.launch_description_sources import PythonLaunchDescriptionSource",
            "from launch_ros.actions import Node",
            "from ament_index_python.packages import get_package_share_directory",
            "",
            "def generate_launch_description():",
            "    # Launch arguments",
            "    use_sim_time = LaunchConfiguration('use_sim_time', default='true')",
            "    world_file = LaunchConfiguration('world_file', default='empty.world')",
            "    gui = LaunchConfiguration('gui', default='true')",
            "    headless = LaunchConfiguration('headless', default='false')",
            "",
            "    # Declare launch arguments",
            "    declare_use_sim_time = DeclareLaunchArgument(",
            "        'use_sim_time',",
            "        default_value='true',",
            "        description='Use simulation (Gazebo) clock if true'",
            "    )",
            "",
            "    declare_world_file = DeclareLaunchArgument(",
            "        'world_file',",
            "        default_value='empty.world',",
            "        description='Gazebo world file'",
            "    )",
            "",
            "    declare_gui = DeclareLaunchArgument(",
            "        'gui',",
            "        default_value='true',",
            "        description='Launch Gazebo GUI'",
            "    )",
            "",
            "    declare_headless = DeclareLaunchArgument(",
            "        'headless',",
            "        default_value='false',",
            "        description='Run Gazebo in headless mode'",
            "    )",
            "",
            "    # Include Gazebo launch",
            "    gazebo_launch = IncludeLaunchDescription(",
            "        PythonLaunchDescriptionSource([",
            "            PathJoinSubstitution([",
            "                get_package_share_directory('gazebo_ros'),",
            "                'launch',",
            "                'gazebo.launch.py'",
            "            ])",
            "        ]),",
            "        launch_arguments={",
            "            'world': world_file,",
            "            'gui': gui,",
            "            'headless': headless,",
            "        }.items()",
            "    )",
            ""
        ]
        
        # Add robot spawn nodes
        for robot in simulation.robots:
            lines.extend(self._generate_robot_spawn_node(robot))
        
        # Add plugin nodes
        for plugin in simulation.plugins:
            lines.extend(self._generate_plugin_node(plugin))
        
        lines.extend([
            "    return LaunchDescription([",
            "        declare_use_sim_time,",
            "        declare_world_file,",
            "        declare_gui,",
            "        declare_headless,",
            "        gazebo_launch,",
        ])
        
        # Add robot and plugin nodes to launch description
        for robot in simulation.robots:
            lines.append(f"        # Robot: {robot.model_file},")
        for plugin in simulation.plugins:
            lines.append(f"        # Plugin: {plugin.name},")
        
        lines.extend([
            "    ])",
            ""
        ])
        
        return "\n".join(lines)
    
    def _generate_isaac_launch(self, simulation: SimulationConfigNode) -> str:
        """Generate Isaac Sim launch file."""
        lines = [
            "#!/usr/bin/env python3",
            "",
            "import os",
            "from launch import LaunchDescription",
            "from launch.actions import DeclareLaunchArgument, ExecuteProcess",
            "from launch.substitutions import LaunchConfiguration",
            "from launch_ros.actions import Node",
            "",
            "def generate_launch_description():",
            "    # Launch arguments",
            "    use_sim_time = LaunchConfiguration('use_sim_time', default='true')",
            "    world_file = LaunchConfiguration('world_file', default='default.usd')",
            "",
            "    # Declare launch arguments",
            "    declare_use_sim_time = DeclareLaunchArgument(",
            "        'use_sim_time',",
            "        default_value='true',",
            "        description='Use simulation (Isaac Sim) clock if true'",
            "    )",
            "",
            "    declare_world_file = DeclareLaunchArgument(",
            "        'world_file',",
            "        default_value='default.usd',",
            "        description='Isaac Sim world file'",
            "    )",
            "",
            "    # Isaac Sim process",
            "    isaac_sim = ExecuteProcess(",
            "        cmd=['isaac-sim', '--app', 'omni.isaac.sim.baseapp', '--ext-folder', '/path/to/extensions'],",
            "        output='screen'",
            "    )",
            "",
            "    return LaunchDescription([",
            "        declare_use_sim_time,",
            "        declare_world_file,",
            "        isaac_sim,",
            "    ])",
            ""
        ]
        
        return "\n".join(lines)
    
    def _generate_gym_launch(self, simulation: SimulationConfigNode) -> str:
        """Generate Gym launch file."""
        lines = [
            "#!/usr/bin/env python3",
            "",
            "import os",
            "from launch import LaunchDescription",
            "from launch.actions import DeclareLaunchArgument",
            "from launch.substitutions import LaunchConfiguration",
            "from launch_ros.actions import Node",
            "",
            "def generate_launch_description():",
            "    # Launch arguments",
            "    use_sim_time = LaunchConfiguration('use_sim_time', default='true')",
            "    env_name = LaunchConfiguration('env_name', default='FetchPickAndPlace-v1')",
            "",
            "    # Declare launch arguments",
            "    declare_use_sim_time = DeclareLaunchArgument(",
            "        'use_sim_time',",
            "        default_value='true',",
            "        description='Use simulation (Gym) clock if true'",
            "    )",
            "",
            "    declare_env_name = DeclareLaunchArgument(",
            "        'env_name',",
            "        default_value='FetchPickAndPlace-v1',",
            "        description='Gym environment name'",
            "    )",
            "",
            "    # Gym environment node",
            "    gym_env = Node(",
            "        package='gym_ros',",
            "        executable='gym_env_node',",
            "        name='gym_environment',",
            "        parameters=[",
            "            {'use_sim_time': use_sim_time},",
            "            {'env_name': env_name},",
            "        ],",
            "        output='screen'",
            "    )",
            "",
            "    return LaunchDescription([",
            "        declare_use_sim_time,",
            "        declare_env_name,",
            "        gym_env,",
            "    ])",
            ""
        ]
        
        return "\n".join(lines)
    
    def _generate_robot_spawn_node(self, robot: SimulationRobotNode) -> List[str]:
        """Generate robot spawn node for launch file."""
        lines = [
            f"    # Spawn robot: {robot.model_file}",
            f"    robot_state_publisher = Node(",
            "        package='robot_state_publisher',",
            "        executable='robot_state_publisher',",
            f"        name='{robot.namespace or 'robot'}_state_publisher',",
            "        parameters=[",
            "            {'use_sim_time': use_sim_time},",
            f"            {{'robot_description': open('{robot.model_file}', 'r').read()}},",
            "        ],",
            "        output='screen'",
            "    )",
            "",
            f"    spawn_robot = Node(",
            "        package='gazebo_ros',",
            "        executable='spawn_entity.py',",
            f"        name='spawn_{robot.namespace or 'robot'}',",
            "        arguments=[",
            f"            '-entity', '{robot.namespace or 'robot'}',",
            f"            '-file', '{robot.model_file}',",
        ]
        
        if robot.initial_pose:
            x, y, z, roll, pitch, yaw = robot.initial_pose
            lines.extend([
                f"            '-x', '{x}',",
                f"            '-y', '{y}',",
                f"            '-z', '{z}',",
                f"            '-R', '{roll}',",
                f"            '-P', '{pitch}',",
                f"            '-Y', '{yaw}',",
            ])
        
        lines.extend([
            "        ],",
            "        output='screen'",
            "    )",
            ""
        ])
        
        return lines
    
    def _generate_plugin_node(self, plugin: SimulationPluginNode) -> List[str]:
        """Generate plugin node for launch file."""
        lines = [
            f"    # Plugin: {plugin.name}",
            f"    {plugin.name}_node = Node(",
            "        package='gazebo_plugins',",
            f"        executable='{plugin.name}',",
            f"        name='{plugin.name}',",
            "        parameters=[",
            "            {'use_sim_time': use_sim_time},",
        ]
        
        for param_name, param_value in plugin.parameters.items():
            lines.append(f"            {{'{param_name}': {param_value}}},")
        
        lines.extend([
            "        ],",
            "        output='screen'",
            "    )",
            ""
        ])
        
        return lines
    
    def _generate_hil_launch_content(self, hil_config: HardwareInLoopNode) -> str:
        """Generate hardware-in-the-loop launch file content."""
        lines = [
            "#!/usr/bin/env python3",
            "",
            "import os",
            "from launch import LaunchDescription",
            "from launch.actions import DeclareLaunchArgument, GroupAction",
            "from launch.substitutions import LaunchConfiguration",
            "from launch_ros.actions import Node",
            "",
            "def generate_launch_description():",
            "    # Launch arguments",
            "    use_sim_time = LaunchConfiguration('use_sim_time', default='true')",
            "    bridge_config = LaunchConfiguration('bridge_config', default='hil_bridge.yaml')",
            "",
            "    # Declare launch arguments",
            "    declare_use_sim_time = DeclareLaunchArgument(",
            "        'use_sim_time',",
            "        default_value='true',",
            "        description='Use simulation clock if true'",
            "    )",
            "",
            "    declare_bridge_config = DeclareLaunchArgument(",
            "        'bridge_config',",
            "        default_value='hil_bridge.yaml',",
            "        description='Bridge configuration file'",
            "    )",
            "",
            "    # Simulation nodes group",
            "    simulation_nodes = GroupAction([",
        ]
        
        for node_name in hil_config.simulation_nodes:
            lines.extend([
                f"        Node(",
                f"            package='your_package',",
                f"            executable='{node_name}',",
                f"            name='{node_name}',",
                "            parameters=[",
                "                {'use_sim_time': use_sim_time},",
                "            ],",
                "            output='screen'",
                "        ),",
            ])
        
        lines.extend([
            "    ])",
            "",
            "    # Hardware nodes group",
            "    hardware_nodes = GroupAction([",
        ])
        
        for node_name in hil_config.hardware_nodes:
            lines.extend([
                f"        Node(",
                f"            package='your_package',",
                f"            executable='{node_name}',",
                f"            name='{node_name}',",
                "            parameters=[",
                "                {'use_sim_time': False},",
                "            ],",
                "            output='screen'",
                "        ),",
            ])
        
        lines.extend([
            "    ])",
            "",
            "    # Bridge node",
            "    bridge_node = Node(",
            "        package='ros_ign_bridge',",
            "        executable='parameter_bridge',",
            "        name='hil_bridge',",
            "        arguments=[",
            "            bridge_config,",
            "        ],",
            "        output='screen'",
            "    )",
            "",
            "    return LaunchDescription([",
            "        declare_use_sim_time,",
            "        declare_bridge_config,",
            "        simulation_nodes,",
            "        hardware_nodes,",
            "        bridge_node,",
            "    ])",
            ""
        ])
        
        return "\n".join(lines) 