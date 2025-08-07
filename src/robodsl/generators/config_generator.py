"""Config Generator for RoboDSL.

This generator creates YAML configuration files for ROS2 nodes.
"""

from pathlib import Path
from typing import Dict, List, Any

from .base_generator import BaseGenerator
from ..core.ast import RoboDSLAST, NodeNode


class ConfigGenerator(BaseGenerator):
    """Generates YAML configuration files."""
    
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate config files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Create config directory
        (self.output_dir / 'config').mkdir(parents=True, exist_ok=True)
        
        # Generate individual node config files
        for node in ast.nodes:
            config_path = self._generate_node_config(node)
            generated_files.append(config_path)
        
        return generated_files
    
    def _generate_node_config(self, node: NodeNode) -> Path:
        """Generate a YAML config file for a single node."""
        context = self._prepare_node_config_context(node)
        
        try:
            content = self.render_template('node_config.yaml.jinja2', context)
            config_path = self.get_output_path('config', f'{node.name}.yaml')
            return self.write_file(config_path, content)
        except Exception as e:
            print(f"Template error for node {node.name} config: {e}")
            # Fallback to simple config
            content = self._generate_fallback_node_config(node)
            config_path = self.get_output_path('config', f'{node.name}.yaml')
            return self.write_file(config_path, content)
    
    def _prepare_node_config_context(self, node: NodeNode) -> Dict[str, Any]:
        """Prepare context for node config template rendering."""
        # Prepare parameters
        parameters = {}
        for param in node.content.parameters:
            parameters[param.name] = param.value.value
        
        return {
            'node_name': node.name,
            'parameters': parameters,
            'description': f'Configuration for {node.name} node'
        }
    
    def _generate_fallback_node_config(self, node: NodeNode) -> str:
        """Generate a fallback node config file if template fails."""
        content = f"# Configuration for {node.name} node\n"
        content += f"# Generated from RoboDSL specification\n\n"
        
        # Add parameters if any
        if node.content.parameters:
            content += f"{node.name}:\n"
            for param in node.content.parameters:
                value = param.value.value
                if isinstance(value, str):
                    content += f"  {param.name}: '{value}'\n"
                else:
                    content += f"  {param.name}: {value}\n"
        else:
            content += f"{node.name}:\n"
            content += f"  # No parameters defined\n"
        
        return content 