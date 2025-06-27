"""Python Node Generator for RoboDSL.

This generator creates Python (.py) files for ROS2 nodes.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST, NodeNode


class PythonNodeGenerator(BaseGenerator):
    """Generates Python node files."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate Python node files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Create output directory
        (self.output_dir / 'robodsl').mkdir(parents=True, exist_ok=True)
        
        # Generate files for each node
        for node in ast.nodes:
            # Generate Python node file
            py_path = self._generate_python_node(node)
            generated_files.append(py_path)
        
        return generated_files
    
    def _generate_python_node(self, node: NodeNode) -> Path:
        """Generate a Python file for a ROS2 node."""
        context = self._prepare_node_context(node)
        
        try:
            content = self.render_template('node.py.jinja2', context)
            py_path = self.get_output_path('robodsl', f'{node.name}_node.py')
            return self.write_file(py_path, content)
        except Exception as e:
            print(f"Template error for node {node.name}: {e}")
            # Fallback to simple Python node
            content = f"""#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

class {node.name.capitalize()}Node(Node):
    def __init__(self):
        super().__init__('{node.name}_node')
        self.get_logger().info('{node.name} node started')

def main(args=None):
    rclpy.init(args=args)
    node = {node.name.capitalize()}Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
"""
            py_path = self.get_output_path('robodsl', f'{node.name}_node.py')
            return self.write_file(py_path, content)
    
    def _prepare_node_context(self, node: NodeNode) -> Dict[str, Any]:
        """Prepare context for Python node template rendering."""
        # Prepare imports based on node content
        imports = []
        message_imports = []
        
        # Add message imports for publishers/subscribers
        for pub in node.content.publishers:
            msg_type = pub.msg_type
            if '.' in msg_type:
                package, msg = msg_type.split('.', 1)
                imports.append(f"from {package}.msg import {msg}")
                message_imports.append(msg)
            else:
                imports.append(f"from std_msgs.msg import {msg_type}")
                message_imports.append(msg_type)
        
        for sub in node.content.subscribers:
            msg_type = sub.msg_type
            if '.' in msg_type:
                package, msg = msg_type.split('.', 1)
                imports.append(f"from {package}.msg import {msg}")
                message_imports.append(msg)
            else:
                imports.append(f"from std_msgs.msg import {msg_type}")
                message_imports.append(msg_type)
        
        # Add service imports
        for srv in node.content.services:
            srv_type = srv.srv_type
            if '.' in srv_type:
                package, srv_name = srv_type.split('.', 1)
                imports.append(f"from {package}.srv import {srv_name}")
            else:
                imports.append(f"from std_srvs.srv import {srv_type}")
        
        # Add action imports
        for action in node.content.actions:
            action_type = action.action_type
            if '.' in action_type:
                package, action_name = action_type.split('.', 1)
                imports.append(f"from {action}.action import {action_name}")
            else:
                imports.append(f"from std_msgs.action import {action_type}")
        
        # Determine if this is a lifecycle node
        is_lifecycle = node.content.lifecycle is not None
        
        # Prepare publishers
        publishers = []
        for pub in node.content.publishers:
            msg_type = pub.msg_type.split('.')[-1] if '.' in pub.msg_type else pub.msg_type
            publishers.append({
                'name': pub.topic.split('/')[-1],
                'msg_type': msg_type,
                'topic': pub.topic,
                'qos': self._prepare_qos_context(pub.qos) if pub.qos else None
            })
        
        # Prepare subscribers
        subscribers = []
        for sub in node.content.subscribers:
            msg_type = sub.msg_type.split('.')[-1] if '.' in sub.msg_type else sub.msg_type
            subscribers.append({
                'name': sub.topic.split('/')[-1],
                'msg_type': msg_type,
                'topic': sub.topic,
                'callback_name': f"on_{sub.topic.split('/')[-1]}",
                'qos': self._prepare_qos_context(sub.qos) if sub.qos else None
            })
        
        # Prepare services
        services = []
        for srv in node.content.services:
            srv_type = srv.srv_type.split('.')[-1] if '.' in srv.srv_type else srv.srv_type
            services.append({
                'name': srv.service.split('/')[-1],
                'srv_type': srv_type,
                'service': srv.service,
                'callback_name': f"on_{srv.service.split('/')[-1]}"
            })
        
        # Prepare timers
        timers = []
        for timer in node.content.timers:
            timers.append({
                'name': timer.name,
                'callback_name': f"on_{timer.name}",
                'period': timer.period
            })
        
        # Prepare parameters
        parameters = []
        for param in node.content.parameters:
            parameters.append({
                'name': param.name,
                'type': param.param_type,
                'default_value': param.default_value
            })
        
        return {
            'class_name': f"{node.name.capitalize()}Node",
            'base_class': 'LifecycleNode' if is_lifecycle else 'Node',
            'node_name': f"{node.name}_node",
            'imports': list(set(imports)),  # Remove duplicates
            'message_imports': list(set(message_imports)),
            'is_lifecycle': is_lifecycle,
            'publishers': publishers,
            'subscribers': subscribers,
            'services': services,
            'timers': timers,
            'parameters': parameters
        }
    
    def _prepare_qos_context(self, qos) -> Dict[str, Any]:
        """Prepare QoS context for template rendering."""
        if not qos:
            return {}
        
        qos_settings = {}
        
        # Handle reliability
        if qos.reliability:
            if qos.reliability == "reliable":
                qos_settings['reliability'] = 'ReliabilityPolicy.RELIABLE'
            elif qos.reliability == "best_effort":
                qos_settings['reliability'] = 'ReliabilityPolicy.BEST_EFFORT'
        
        # Handle durability
        if qos.durability:
            if qos.durability == "volatile":
                qos_settings['durability'] = 'DurabilityPolicy.VOLATILE'
            elif qos.durability == "transient_local":
                qos_settings['durability'] = 'DurabilityPolicy.TRANSIENT_LOCAL'
        
        # Handle history
        if qos.history:
            if qos.history == "keep_last":
                qos_settings['history'] = 'HistoryPolicy.KEEP_LAST'
            elif qos.history == "keep_all":
                qos_settings['history'] = 'HistoryPolicy.KEEP_ALL'
        
        # Handle depth
        if qos.depth:
            qos_settings['depth'] = qos.depth
        
        return qos_settings 