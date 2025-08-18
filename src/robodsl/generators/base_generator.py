"""Base generator class for RoboDSL code generation."""

import jinja2
from pathlib import Path
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

from ..core.ast import RoboDSLAST


class BaseGenerator(ABC):
    """Base class for all code generators."""
    
    def __init__(self, output_dir: str = "test_output", template_dirs: Optional[List[Path]] = None):
        """Initialize the base generator.
        
        Args:
            output_dir: Base directory for generated files (defaults to test_output)
            template_dirs: Additional template directories to search
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up Jinja2 environment
        self.env = self._setup_jinja2_environment(template_dirs or [])
    
    def _setup_jinja2_environment(self, additional_template_dirs: List[Path]) -> jinja2.Environment:
        """Set up Jinja2 environment with template directories."""
        # Default template directories
        default_template_dirs = [
            Path(__file__).parent.parent / 'templates',
            Path(__file__).parent.parent / 'templates/cpp',
            Path(__file__).parent.parent / 'templates/cuda',
            Path(__file__).parent.parent / 'templates/py',
            Path(__file__).parent.parent / 'templates/cmake',
            Path(__file__).parent.parent / 'templates/launch',
        ]
        
        # Add additional template directories
        all_template_dirs = default_template_dirs + additional_template_dirs
        
        # Filter to only existing directories
        existing_dirs = [d for d in all_template_dirs if d.exists()]
        
        if not existing_dirs:
            # Fallback to basic environment if no template directories exist
            return jinja2.Environment(
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )
        
        # Create template loaders
        template_loaders = [jinja2.FileSystemLoader(str(d)) for d in existing_dirs]
        
        # Create environment
        env = jinja2.Environment(
            loader=jinja2.ChoiceLoader(template_loaders),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True
        )
        
        # Add custom filters
        env.filters['indent'] = lambda text, n: '\n'.join(' ' * n + line if line.strip() else line 
                                                for line in text.split('\n'))
        
        # Add dedent filter to remove common leading whitespace
        import textwrap
        env.filters['dedent'] = lambda text: textwrap.dedent(text)
        
        return env
    
    @abstractmethod
    def generate(self, ast: RoboDSLAST) -> List[Path]:
        """Generate files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        pass
    
    def render_template(self, template_name: str, context: Dict[str, Any]) -> str:
        """Render a template with the given context.
        
        Args:
            template_name: Name of the template file
            context: Template context variables
            
        Returns:
            Rendered template content
            
        Raises:
            jinja2.TemplateNotFound: If template doesn't exist
        """
        try:
            template = self.env.get_template(template_name)
            return template.render(**context)
        except jinja2.TemplateNotFound:
            raise jinja2.TemplateNotFound(f"Template '{template_name}' not found")
        except Exception as e:
            raise Exception(f"Error rendering template '{template_name}': {e}")
    
    def write_file(self, file_path: Path, content: str) -> Path:
        """Write content to a file.
        
        Args:
            file_path: Path to write the file to
            content: Content to write
            
        Returns:
            Path object for the written file
        """
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        return file_path
    
    def get_output_path(self, *path_parts: str) -> Path:
        """Get a path relative to the output directory.
        
        Args:
            *path_parts: Path components
            
        Returns:
            Path object
        """
        return self.output_dir.joinpath(*path_parts) 