"""Message/Service/Action Generator for RoboDSL.

This module generates ROS2 message, service, and action files from RoboDSL AST.
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from ..core.ast import (
    MessageNode, ServiceNode, CustomActionNode, 
    MessageFieldNode, ServiceRequestNode, ServiceResponseNode,
    ActionGoalNode, ActionFeedbackNode, ActionResultNode
)


class MessageGenerator:
    """Generates ROS2 message, service, and action files."""
    
    def __init__(self, output_dir: str = "msg"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def generate_messages(self, messages: List[MessageNode]) -> List[str]:
        """Generate .msg files for custom messages."""
        generated_files = []
        
        for message in messages:
            msg_file = self.output_dir / f"{message.name}.msg"
            content = self._generate_message_content(message)
            
            with open(msg_file, 'w') as f:
                f.write(content)
            
            generated_files.append(str(msg_file))
        
        return generated_files
    
    def generate_services(self, services: List[ServiceNode]) -> List[str]:
        """Generate .srv files for custom services."""
        generated_files = []
        
        for service in services:
            srv_file = self.output_dir / f"{service.name}.srv"
            content = self._generate_service_content(service)
            
            with open(srv_file, 'w') as f:
                f.write(content)
            
            generated_files.append(str(srv_file))
        
        return generated_files
    
    def generate_actions(self, actions: List[CustomActionNode]) -> List[str]:
        """Generate .action files for custom actions."""
        generated_files = []
        
        for action in actions:
            action_file = self.output_dir / f"{action.name}.action"
            content = self._generate_action_content(action)
            
            with open(action_file, 'w') as f:
                f.write(content)
            
            generated_files.append(str(action_file))
        
        return generated_files
    
    def _generate_message_content(self, message: MessageNode) -> str:
        """Generate content for a .msg file."""
        lines = []
        
        # Add constants first
        for constant in message.content.constants:
            if constant.default_value:
                lines.append(f"{constant.type} {constant.name}={constant.default_value.value}")
        
        # Add fields
        for field in message.content.fields:
            field_line = f"{field.type} {field.name}"
            if field.array_spec:
                field_line += field.array_spec
            if field.default_value:
                field_line += f"={field.default_value.value}"
            lines.append(field_line)
        
        return "\n".join(lines)
    
    def _generate_service_content(self, service: ServiceNode) -> str:
        """Generate content for a .srv file."""
        lines = []
        
        # Request part
        for field in service.content.request.fields:
            field_line = f"{field.type} {field.name}"
            if field.array_spec:
                field_line += field.array_spec
            if field.default_value:
                field_line += f"={field.default_value.value}"
            lines.append(field_line)
        
        # Separator
        lines.append("---")
        
        # Response part
        for field in service.content.response.fields:
            field_line = f"{field.type} {field.name}"
            if field.array_spec:
                field_line += field.array_spec
            if field.default_value:
                field_line += f"={field.default_value.value}"
            lines.append(field_line)
        
        return "\n".join(lines)
    
    def _generate_action_content(self, action: CustomActionNode) -> str:
        """Generate content for a .action file."""
        lines = []
        
        # Goal part
        for field in action.content.goal.fields:
            field_line = f"{field.type} {field.name}"
            if field.array_spec:
                field_line += field.array_spec
            if field.default_value:
                field_line += f"={field.default_value.value}"
            lines.append(field_line)
        
        # Separator
        lines.append("---")
        
        # Feedback part
        for field in action.content.feedback.fields:
            field_line = f"{field.type} {field.name}"
            if field.array_spec:
                field_line += field.array_spec
            if field.default_value:
                field_line += f"={field.default_value.value}"
            lines.append(field_line)
        
        # Separator
        lines.append("---")
        
        # Result part
        for field in action.content.result.fields:
            field_line = f"{field.type} {field.name}"
            if field.array_spec:
                field_line += field.array_spec
            if field.default_value:
                field_line += f"={field.default_value.value}"
            lines.append(field_line)
        
        return "\n".join(lines) 