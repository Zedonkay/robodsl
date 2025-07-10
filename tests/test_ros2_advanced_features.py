"""Advanced ROS2 features comprehensive testing.

This module provides extensive test coverage for advanced ROS2 features including:
- QoS (Quality of Service) configurations
- Lifecycle nodes
- Services and clients
- Actions and action clients
- Parameters and parameter servers
- Advanced communication patterns
- Multi-node systems
- Real-time features
- Security features
"""

import pytest
import tempfile
import shutil
import os
import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2, has_ros2
from robodsl.generators.cpp_node_generator import CppNodeGenerator
from robodsl.generators.python_node_generator import PythonNodeGenerator
from robodsl.core.ast import NodeNode, ServiceNode, ActionNode, ParameterNode


class TestROS2AdvancedFeatures:
    """Advanced ROS2 features test suite."""
    
    @pytest.fixture
    def ros2_config(self):
        """Get ROS2 configuration."""
        return {
            "distribution": self._get_ros2_distribution(),
            "workspace": self._get_ros2_workspace(),
            "available_packages": self._get_available_packages(),
            "qos_profiles": self._get_qos_profiles(),
            "message_types": self._get_message_types()
        }
    
    def _get_ros2_distribution(self):
        """Get ROS2 distribution."""
        try:
            result = subprocess.run(['ros2', '--version'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'distribution' in line.lower():
                        return line.split()[-1]
        except:
            pass
        return "unknown"
    
    def _get_ros2_workspace(self):
        """Get ROS2 workspace."""
        try:
            result = subprocess.run(['printenv', 'COLCON_PREFIX_PATH'], 
                                 capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split(':')[0]
        except:
            pass
        return "/opt/ros/humble"
    
    def _get_available_packages(self):
        """Get available ROS2 packages."""
        try:
            result = subprocess.run(['ros2', 'pkg', 'list'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except:
            pass
        return ["rclcpp", "std_msgs", "sensor_msgs", "geometry_msgs"]
    
    def _get_qos_profiles(self):
        """Get available QoS profiles."""
        return [
            "sensor_data",
            "services_default",
            "parameters_default",
            "parameter_events",
            "system_default",
            "reliable",
            "best_effort",
            "durability_volatile",
            "durability_transient_local",
            "history_keep_last",
            "history_keep_all"
        ]
    
    def _get_message_types(self):
        """Get available message types."""
        try:
            result = subprocess.run(['ros2', 'interface', 'list'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except:
            pass
        return ["std_msgs/String", "std_msgs/Int32", "sensor_msgs/Image"]
    
    def test_qos_configurations_advanced(self, ros2_config):
        """Test advanced QoS configurations."""
        skip_if_no_ros2()

        dsl_code = '''
        cpp_node qos_test_node {
            publisher: "topic1" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
                durable: false
                history: keep_last
                depth: 10
            }
            subscriber: "topic1" -> "std_msgs/String" {
                qos: best_effort
                queue_size: 5
                durable: false
                history: keep_last
                depth: 5
            }
        }
        '''

        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "qos_test_node"
        assert len(node.publishers) == 1
        assert len(node.subscribers) == 1
    
    def test_lifecycle_nodes_advanced(self, ros2_config):
        """Test advanced lifecycle node features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node lifecycle_node {
            lifecycle: true
            lifecycle_states: ["unconfigured", "inactive", "active", "finalized"]
            lifecycle_transitions: ["configure", "activate", "deactivate", "cleanup", "shutdown"]
            
            publisher: "status" -> "lifecycle_msgs/State" {
                qos: reliable
                queue_size: 10
            }
            
            subscriber: "change_state" -> "lifecycle_msgs/ChangeState" {
                qos: reliable
                queue_size: 10
            }
            
            service: "get_state" -> "lifecycle_msgs/GetState"
            service: "change_state" -> "lifecycle_msgs/ChangeState"
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "lifecycle_node"
        assert node.lifecycle is True
    
    def test_services_advanced(self, ros2_config):
        """Test advanced service features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node service_node {
            service: "add_two_ints" -> "example_interfaces/AddTwoInts" {
                qos: services_default
                callback: "add_two_ints_callback"
            }
            
            service: "set_parameters" -> "rcl_interfaces/SetParameters" {
                qos: services_default
                callback: "set_parameters_callback"
            }
            
            service: "get_parameters" -> "rcl_interfaces/GetParameters" {
                qos: services_default
                callback: "get_parameters_callback"
            }
            
            service_client: "external_service" -> "example_interfaces/AddTwoInts" {
                qos: services_default
                timeout: 5.0
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "service_node"
        assert len(node.services) == 3
        assert len(node.service_clients) == 1
    
    def test_actions_advanced(self, ros2_config):
        """Test advanced action features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node action_node {
            action: "fibonacci" -> "example_interfaces/Fibonacci" {
                qos: reliable
                callback: "fibonacci_callback"
                goal_callback: "fibonacci_goal_callback"
                cancel_callback: "fibonacci_cancel_callback"
            }
            
            action: "navigate" -> "nav2_msgs/NavigateToPose" {
                qos: reliable
                callback: "navigate_callback"
                goal_callback: "navigate_goal_callback"
                cancel_callback: "navigate_cancel_callback"
            }
            
            action_client: "external_action" -> "example_interfaces/Fibonacci" {
                qos: reliable
                timeout: 30.0
                goal_timeout: 10.0
                result_timeout: 60.0
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "action_node"
        assert len(node.actions) == 2
        assert len(node.action_clients) == 1
    
    def test_parameters_advanced(self, ros2_config):
        """Test advanced parameter features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node parameter_node {
            parameter: "my_string" -> "string" {
                default: "hello_world"
                description: "A string parameter"
                read_only: false
            }
            
            parameter: "my_int" -> "int" {
                default: 42
                description: "An integer parameter"
                read_only: false
                range: [0, 100]
            }
            
            parameter: "my_double" -> "double" {
                default: 3.14159
                description: "A double parameter"
                read_only: false
                range: [0.0, 10.0]
            }
            
            parameter: "my_bool" -> "bool" {
                default: true
                description: "A boolean parameter"
                read_only: false
            }
            
            parameter: "my_array" -> "double_array" {
                default: [1.0, 2.0, 3.0]
                description: "An array parameter"
                read_only: false
            }
            
            parameter_server: true
            parameter_client: true
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "parameter_node"
        assert len(node.parameters) == 5
        assert node.parameter_server is True
        assert node.parameter_client is True
    
    def test_multi_node_systems(self, ros2_config):
        """Test multi-node system configurations."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node publisher_node {
            publisher: "data_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            timer: "publish_timer" {
                period: 1.0
                callback: "publish_callback"
            }
        }
        
        cpp_node subscriber_node {
            subscriber: "data_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            timer: "process_timer" {
                period: 0.1
                callback: "process_callback"
            }
        }
        
        cpp_node service_node {
            service: "process_data" -> "std_srvs/Trigger" {
                qos: services_default
                callback: "process_data_callback"
            }
            
            subscriber: "data_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 3
        assert ast.nodes[0].name == "publisher_node"
        assert ast.nodes[1].name == "subscriber_node"
        assert ast.nodes[2].name == "service_node"
    
    def test_real_time_features(self, ros2_config):
        """Test real-time features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node realtime_node {
            realtime: true
            realtime_priority: 80
            realtime_policy: "SCHED_FIFO"
            realtime_cpu: 0
            
            publisher: "realtime_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 1
                reliable: true
                durable: false
                history: keep_last
                depth: 1
            }
            
            subscriber: "realtime_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 1
                reliable: true
                durable: false
                history: keep_last
                depth: 1
            }
            
            timer: "realtime_timer" {
                period: 0.001
                callback: "realtime_callback"
                realtime: true
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "realtime_node"
        assert node.realtime is True
    
    def test_security_features(self, ros2_config):
        """Test security features."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node secure_node {
            security: true
            security_enforce: true
            security_identity: "secure_node"
            security_namespace: "/secure"
            
            publisher: "secure_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
                security: true
                encryption: true
                authentication: true
            }
            
            subscriber: "secure_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
                security: true
                encryption: true
                authentication: true
            }
            
            service: "secure_service" -> "std_srvs/Trigger" {
                qos: services_default
                callback: "secure_service_callback"
                security: true
                encryption: true
                authentication: true
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "secure_node"
        assert node.security is True
    
    def test_advanced_communication_patterns(self, ros2_config):
        """Test advanced communication patterns."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node communication_node {
            # Request-reply pattern
            service: "request_reply" -> "std_srvs/Trigger" {
                qos: services_default
                callback: "request_reply_callback"
            }
            
            # Publish-subscribe pattern
            publisher: "data_stream" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            subscriber: "data_stream" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            # Request-response pattern with action
            action: "long_running_task" -> "example_interfaces/Fibonacci" {
                qos: reliable
                callback: "long_running_task_callback"
                goal_callback: "long_running_task_goal_callback"
                cancel_callback: "long_running_task_cancel_callback"
            }
            
            # Parameter pattern
            parameter: "config_param" -> "string" {
                default: "default_value"
                description: "Configuration parameter"
                read_only: false
            }
            
            # Timer-based periodic communication
            timer: "periodic_timer" {
                period: 1.0
                callback: "periodic_callback"
            }
            
            # Event-driven communication
            subscriber: "event_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
                callback: "event_callback"
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "communication_node"
        assert len(node.services) == 1
        assert len(node.publishers) == 1
        assert len(node.subscribers) == 2
        assert len(node.actions) == 1
        assert len(node.parameters) == 1
        assert len(node.timers) == 1
    
    def test_qos_profiles_comprehensive(self, ros2_config):
        """Test comprehensive QoS profiles."""
        skip_if_no_ros2()
        
        qos_profiles = ros2_config["qos_profiles"]
        
        for profile in qos_profiles:
            dsl_code = f'''
            cpp_node qos_comprehensive_{profile.replace("_", "_")} {{
                publisher: "topic_{profile}" -> "std_msgs/String" {{
                    qos: {profile}
                    queue_size: 10
                    reliable: true
                    durable: false
                    history: keep_last
                    depth: 10
                    deadline: 1.0
                    lifespan: 5.0
                    liveliness: automatic
                    liveliness_lease_duration: 2.0
                }}
                
                subscriber: "topic_{profile}" -> "std_msgs/String" {{
                    qos: {profile}
                    queue_size: 10
                    reliable: true
                    durable: false
                    history: keep_last
                    depth: 10
                    deadline: 1.0
                    lifespan: 5.0
                    liveliness: automatic
                    liveliness_lease_duration: 2.0
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.nodes) == 1
            node = ast.nodes[0]
            assert node.name == f"qos_comprehensive_{profile.replace('_', '_')}"
    
    def test_message_types_comprehensive(self, ros2_config):
        """Test comprehensive message types."""
        skip_if_no_ros2()
        
        message_types = ros2_config["message_types"][:10]  # Limit to first 10 for testing
        
        for msg_type in message_types:
            # Extract package and message name
            if '/' in msg_type:
                package, message = msg_type.split('/', 1)
                message_name = message.replace('/', '_')
            else:
                continue
            
            dsl_code = f'''
            cpp_node message_test_{message_name} {{
                publisher: "topic_{message_name}" -> "{msg_type}" {{
                    qos: reliable
                    queue_size: 10
                }}
                
                subscriber: "topic_{message_name}" -> "{msg_type}" {{
                    qos: reliable
                    queue_size: 10
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.nodes) == 1
            node = ast.nodes[0]
            assert node.name == f"message_test_{message_name}"
    
    def test_parameter_types_comprehensive(self, ros2_config):
        """Test comprehensive parameter types."""
        skip_if_no_ros2()
        
        parameter_types = [
            ("string_param", "string", "hello_world"),
            ("int_param", "int", 42),
            ("double_param", "double", 3.14159),
            ("bool_param", "bool", True),
            ("int_array_param", "int_array", [1, 2, 3, 4, 5]),
            ("double_array_param", "double_array", [1.0, 2.0, 3.0]),
            ("string_array_param", "string_array", ["a", "b", "c"]),
            ("bool_array_param", "bool_array", [True, False, True])
        ]
        
        for param_name, param_type, default_value in parameter_types:
            dsl_code = f'''
            cpp_node parameter_test_{param_name} {{
                parameter: "{param_name}" -> "{param_type}" {{
                    default: {default_value}
                    description: "Test parameter of type {param_type}"
                    read_only: false
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.nodes) == 1
            node = ast.nodes[0]
            assert node.name == f"parameter_test_{param_name}"
    
    def test_timer_configurations_comprehensive(self, ros2_config):
        """Test comprehensive timer configurations."""
        skip_if_no_ros2()
        
        timer_configs = [
            ("fast_timer", 0.001),
            ("medium_timer", 0.1),
            ("slow_timer", 1.0),
            ("very_slow_timer", 10.0)
        ]
        
        for timer_name, period in timer_configs:
            dsl_code = f'''
            cpp_node timer_test_{timer_name} {{
                timer: "{timer_name}" {{
                    period: {period}
                    callback: "{timer_name}_callback"
                    oneshot: false
                    autostart: true
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.nodes) == 1
            node = ast.nodes[0]
            assert node.name == f"timer_test_{timer_name}"
    
    def test_service_types_comprehensive(self, ros2_config):
        """Test comprehensive service types."""
        skip_if_no_ros2()
        
        service_types = [
            "std_srvs/Trigger",
            "std_srvs/Empty",
            "example_interfaces/AddTwoInts",
            "rcl_interfaces/SetParameters",
            "rcl_interfaces/GetParameters",
            "rcl_interfaces/ListParameters"
        ]
        
        for service_type in service_types:
            service_name = service_type.replace('/', '_').replace('srv', '')
            dsl_code = f'''
            cpp_node service_test_{service_name} {{
                service: "{service_name}" -> "{service_type}" {{
                    qos: services_default
                    callback: "{service_name}_callback"
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.nodes) == 1
            node = ast.nodes[0]
            assert node.name == f"service_test_{service_name}"
    
    def test_action_types_comprehensive(self, ros2_config):
        """Test comprehensive action types."""
        skip_if_no_ros2()
        
        action_types = [
            "example_interfaces/Fibonacci",
            "nav2_msgs/NavigateToPose",
            "control_msgs/FollowJointTrajectory",
            "tf2_msgs/LookupTransform"
        ]
        
        for action_type in action_types:
            action_name = action_type.replace('/', '_').replace('msg', '').replace('srv', '')
            dsl_code = f'''
            cpp_node action_test_{action_name} {{
                action: "{action_name}" -> "{action_type}" {{
                    qos: reliable
                    callback: "{action_name}_callback"
                    goal_callback: "{action_name}_goal_callback"
                    cancel_callback: "{action_name}_cancel_callback"
                }}
            }}
            '''
            
            ast = parse_robodsl(dsl_code)
            assert len(ast.nodes) == 1
            node = ast.nodes[0]
            assert node.name == f"action_test_{action_name}"
    
    def test_node_lifecycle_comprehensive(self, ros2_config):
        """Test comprehensive node lifecycle management."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node lifecycle_comprehensive {
            lifecycle: true
            lifecycle_states: ["unconfigured", "inactive", "active", "finalized"]
            lifecycle_transitions: ["configure", "activate", "deactivate", "cleanup", "shutdown", "error"]
            
            # Lifecycle state publishers
            publisher: "state" -> "lifecycle_msgs/State" {
                qos: reliable
                queue_size: 10
            }
            
            publisher: "transition_event" -> "lifecycle_msgs/TransitionEvent" {
                qos: reliable
                queue_size: 10
            }
            
            # Lifecycle services
            service: "get_state" -> "lifecycle_msgs/GetState" {
                qos: services_default
                callback: "get_state_callback"
            }
            
            service: "change_state" -> "lifecycle_msgs/ChangeState" {
                qos: services_default
                callback: "change_state_callback"
            }
            
            service: "get_available_states" -> "lifecycle_msgs/GetAvailableStates" {
                qos: services_default
                callback: "get_available_states_callback"
            }
            
            service: "get_available_transitions" -> "lifecycle_msgs/GetAvailableTransitions" {
                qos: services_default
                callback: "get_available_transitions_callback"
            }
            
            # Regular node functionality
            publisher: "data" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            subscriber: "data" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            timer: "lifecycle_timer" {
                period: 1.0
                callback: "lifecycle_timer_callback"
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "lifecycle_comprehensive"
        assert node.lifecycle is True
        assert len(node.publishers) == 3
        assert len(node.subscribers) == 1
        assert len(node.services) == 4
        assert len(node.timers) == 1
    
    def test_component_nodes(self, ros2_config):
        """Test component node configurations."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node component_node {
            component: true
            component_name: "my_component"
            component_namespace: "/my_namespace"
            
            publisher: "component_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            subscriber: "component_topic" -> "std_msgs/String" {
                qos: reliable
                queue_size: 10
            }
            
            service: "component_service" -> "std_srvs/Trigger" {
                qos: services_default
                callback: "component_service_callback"
            }
            
            parameter: "component_param" -> "string" {
                default: "component_value"
                description: "Component parameter"
                read_only: false
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "component_node"
        assert node.component is True
    
    def test_parameter_servers_comprehensive(self, ros2_config):
        """Test comprehensive parameter server configurations."""
        skip_if_no_ros2()
        
        dsl_code = '''
        cpp_node parameter_server_comprehensive {
            parameter_server: true
            parameter_client: true
            
            # Parameter declarations
            parameter: "string_param" -> "string" {
                default: "default_string"
                description: "String parameter"
                read_only: false
            }
            
            parameter: "int_param" -> "int" {
                default: 42
                description: "Integer parameter"
                read_only: false
                range: [0, 100]
            }
            
            parameter: "double_param" -> "double" {
                default: 3.14159
                description: "Double parameter"
                read_only: false
                range: [0.0, 10.0]
            }
            
            parameter: "bool_param" -> "bool" {
                default: true
                description: "Boolean parameter"
                read_only: false
            }
            
            parameter: "array_param" -> "double_array" {
                default: [1.0, 2.0, 3.0, 4.0, 5.0]
                description: "Array parameter"
                read_only: false
            }
            
            # Parameter services
            service: "set_parameters" -> "rcl_interfaces/SetParameters" {
                qos: services_default
                callback: "set_parameters_callback"
            }
            
            service: "get_parameters" -> "rcl_interfaces/GetParameters" {
                qos: services_default
                callback: "get_parameters_callback"
            }
            
            service: "list_parameters" -> "rcl_interfaces/ListParameters" {
                qos: services_default
                callback: "list_parameters_callback"
            }
            
            service: "describe_parameters" -> "rcl_interfaces/DescribeParameters" {
                qos: services_default
                callback: "describe_parameters_callback"
            }
            
            # Parameter events
            publisher: "parameter_events" -> "rcl_interfaces/ParameterEvent" {
                qos: parameter_events
                queue_size: 10
            }
            
            subscriber: "parameter_events" -> "rcl_interfaces/ParameterEvent" {
                qos: parameter_events
                queue_size: 10
            }
        }
        '''
        
        ast = parse_robodsl(dsl_code)
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert node.name == "parameter_server_comprehensive"
        assert node.parameter_server is True
        assert node.parameter_client is True
        assert len(node.parameters) == 5
        assert len(node.services) == 4
        assert len(node.publishers) == 1
        assert len(node.subscribers) == 1 