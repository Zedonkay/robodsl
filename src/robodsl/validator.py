"""RoboDSL Validation System.

This module provides comprehensive validation for RoboDSL files,
including syntax, semantic, and style validation.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .parser.lark_parser import RoboDSLParser, ParseError, SemanticError
from .parser.semantic_analyzer import SemanticAnalyzer


class ValidationLevel(Enum):
    """Validation levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """A validation issue found in RoboDSL content."""
    level: ValidationLevel
    message: str
    line: Optional[int] = None
    column: Optional[int] = None
    file_path: Optional[Path] = None
    rule_id: Optional[str] = None
    context: Optional[str] = None


class RoboDSLValidator:
    """Validator for RoboDSL files and content."""
    
    def __init__(self):
        self.parser = RoboDSLParser()
        self.semantic_analyzer = SemanticAnalyzer()
        
        # Style validation patterns
        self.style_patterns = {
            'trailing_whitespace': r'[ \t]+$',
            'missing_newline_end': r'[^\n]$',
            'inconsistent_indentation': r'^(?!\s*$)(?!\s*//).*\S',
            'multiple_empty_lines': r'\n\s*\n\s*\n',
            'no_space_after_colon': r':[^\s]',
            'no_space_before_colon': r'[^\s]:',
        }
        
        # Naming convention patterns
        self.naming_patterns = {
            'node_name': r'^[a-z][a-z0-9_]*$',
            'parameter_name': r'^[a-z][a-z0-9_]*$',
            'topic_name': r'^/[a-z][a-z0-9_/]*$',
            'service_name': r'^/[a-z][a-z0-9_/]*$',
            'action_name': r'^[a-z][a-z0-9_]*$',
            'kernel_name': r'^[a-z][a-z0-9_]*$',
        }
    
    def validate_file(self, file_path: Path) -> List[ValidationIssue]:
        """Validate a RoboDSL file."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Cannot read file: {e}",
                file_path=file_path
            ))
            return issues
        
        # Add file path to all issues
        for issue in self.validate_string(content):
            issue.file_path = file_path
            issues.append(issue)
        
        return issues
    
    def validate_string(self, content: str) -> List[ValidationIssue]:
        """Validate RoboDSL content string."""
        issues = []
        
        # Parse and semantic validation
        try:
            ast = self.parser.parse(content)
        except ParseError as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Parse error: {str(e)}",
                rule_id="parse_error"
            ))
            return issues
        except SemanticError as e:
            issues.append(ValidationIssue(
                level=ValidationLevel.ERROR,
                message=f"Semantic error: {str(e)}",
                rule_id="semantic_error"
            ))
            return issues
        
        # Style validation
        issues.extend(self._validate_style(content))
        
        # Naming convention validation
        issues.extend(self._validate_naming_conventions(ast))
        
        # Best practices validation
        issues.extend(self._validate_best_practices(ast))
        
        # Performance validation
        issues.extend(self._validate_performance(ast))
        
        return issues
    
    def _validate_style(self, content: str) -> List[ValidationIssue]:
        """Validate code style and formatting."""
        issues = []
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Check for trailing whitespace
            if re.search(self.style_patterns['trailing_whitespace'], line):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message="Trailing whitespace",
                    line=i,
                    column=len(line.rstrip()) + 1,
                    rule_id="trailing_whitespace"
                ))
            
            # Check for inconsistent indentation
            if line.strip() and not line.startswith('//'):
                indent = len(line) - len(line.lstrip())
                if indent % 4 != 0:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message="Indentation should be multiples of 4 spaces",
                        line=i,
                        column=1,
                        rule_id="inconsistent_indentation"
                    ))
        
        # Check for multiple empty lines
        if re.search(self.style_patterns['multiple_empty_lines'], content):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="Multiple consecutive empty lines",
                rule_id="multiple_empty_lines"
            ))
        
        # Check for missing newline at end
        if content and not content.endswith('\n'):
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message="File should end with a newline",
                line=len(lines),
                rule_id="missing_newline_end"
            ))
        
        return issues
    
    def _validate_naming_conventions(self, ast) -> List[ValidationIssue]:
        """Validate naming conventions."""
        issues = []
        
        # Validate node names
        for node in ast.nodes:
            if not re.match(self.naming_patterns['node_name'], node.name):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Node name '{node.name}' should be lowercase with underscores",
                    rule_id="node_naming_convention"
                ))
            
            # Validate parameter names
            for param in node.content.parameters:
                if not re.match(self.naming_patterns['parameter_name'], param.name):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Parameter name '{param.name}' should be lowercase with underscores",
                        rule_id="parameter_naming_convention"
                    ))
            
            # Validate topic names
            for pub in node.content.publishers:
                if not re.match(self.naming_patterns['topic_name'], pub.topic):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Topic name '{pub.topic}' should start with / and use lowercase",
                        rule_id="topic_naming_convention"
                    ))
            
            for sub in node.content.subscribers:
                if not re.match(self.naming_patterns['topic_name'], sub.topic):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Topic name '{sub.topic}' should start with / and use lowercase",
                        rule_id="topic_naming_convention"
                    ))
        
        # Validate kernel names
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                if not re.match(self.naming_patterns['kernel_name'], kernel.name):
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Kernel name '{kernel.name}' should be lowercase with underscores",
                        rule_id="kernel_naming_convention"
                    ))
        
        return issues
    
    def _validate_best_practices(self, ast) -> List[ValidationIssue]:
        """Validate best practices."""
        issues = []
        
        # Check for nodes without any communication
        for node in ast.nodes:
            has_publishers = len(node.content.publishers) > 0
            has_subscribers = len(node.content.subscribers) > 0
            has_services = len(node.content.services) > 0
            has_actions = len(node.content.actions) > 0
            
            if not any([has_publishers, has_subscribers, has_services, has_actions]):
                issues.append(ValidationIssue(
                    level=ValidationLevel.WARNING,
                    message=f"Node '{node.name}' has no communication interfaces",
                    rule_id="isolated_node"
                ))
        
        # Check for topics without subscribers
        all_topics = set()
        subscribed_topics = set()
        
        for node in ast.nodes:
            for pub in node.content.publishers:
                all_topics.add(pub.topic)
            for sub in node.content.subscribers:
                subscribed_topics.add(sub.topic)
        
        for topic in all_topics - subscribed_topics:
            issues.append(ValidationIssue(
                level=ValidationLevel.WARNING,
                message=f"Topic '{topic}' is published but never subscribed to",
                rule_id="unused_topic"
            ))
        
        # Check for CUDA kernels without proper configuration
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                if not kernel.content.block_size:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"CUDA kernel '{kernel.name}' should specify block size",
                        rule_id="missing_block_size"
                    ))
                
                if not kernel.content.parameters:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"CUDA kernel '{kernel.name}' has no parameters",
                        rule_id="empty_kernel"
                    ))
        
        return issues
    
    def _validate_performance(self, ast) -> List[ValidationIssue]:
        """Validate performance-related issues."""
        issues = []
        
        # Check for large QoS depths
        for node in ast.nodes:
            for pub in node.content.publishers:
                if pub.qos and pub.qos.depth and pub.qos.depth > 100:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Large QoS depth ({pub.qos.depth}) for topic '{pub.topic}' may cause memory issues",
                        rule_id="large_qos_depth"
                    ))
            
            for sub in node.content.subscribers:
                if sub.qos and sub.qos.depth and sub.qos.depth > 100:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Large QoS depth ({sub.qos.depth}) for topic '{sub.topic}' may cause memory issues",
                        rule_id="large_qos_depth"
                    ))
        
        # Check for CUDA kernels with large shared memory
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                if kernel.content.shared_memory and kernel.content.shared_memory > 16384:
                    issues.append(ValidationIssue(
                        level=ValidationLevel.WARNING,
                        message=f"Large shared memory ({kernel.content.shared_memory} bytes) for kernel '{kernel.name}'",
                        rule_id="large_shared_memory"
                    ))
        
        return issues


class RoboDSLLinter:
    """Linter for RoboDSL files with formatting and style fixes."""
    
    def __init__(self, indent_size: int = 4):
        self.indent_size = indent_size
        self.validator = RoboDSLValidator()
    
    def format_file(self, file_path: Path) -> str:
        """Format a RoboDSL file."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise ValueError(f"Cannot read file: {e}")
        
        return self.format_string(content)
    
    def format_string(self, content: str) -> str:
        """Format RoboDSL content string."""
        # Remove trailing whitespace
        lines = [line.rstrip() for line in content.split('\n')]
        
        # Fix indentation and ensure space before '{' and after ':'
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines and comments
            if not stripped or stripped.startswith('//'):
                formatted_lines.append('')
                continue
            
            # Ensure space before '{' if missing
            if '{' in stripped:
                stripped = re.sub(r'\s*\{', ' {', stripped)
            # Special handling for parameter lines: no space before, one after colon
            if stripped.startswith('parameter '):
                stripped = re.sub(r'\s*:(?!/)(\s*)', r': ', stripped)
            # For publisher/subscriber/service/action: one space before and after colon
            elif any(stripped.startswith(prefix) for prefix in ['publisher ', 'subscriber ', 'service ', 'action ']):
                stripped = re.sub(r'\s*:(?!/)(\s*)', r' : ', stripped)
            else:
                # Default: no space before, one after colon
                stripped = re.sub(r'\s*:(?!/)(\s*)', r': ', stripped)
            
            # Handle indentation changes
            if stripped.endswith('{'):
                formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
                indent_level += 1
            elif stripped.startswith('}'):
                indent_level = max(0, indent_level - 1)
                formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
            else:
                formatted_lines.append(' ' * (indent_level * self.indent_size) + stripped)
        
        # Add newline at end
        result = '\n'.join(formatted_lines)
        if not result.endswith('\n'):
            result += '\n'
        
        return result
    
    def check_formatting(self, content: str) -> List[ValidationIssue]:
        """Check if content is properly formatted."""
        return self.validator.validate_string(content)


def validate_robodsl_file(file_path: Path) -> List[ValidationIssue]:
    """Convenience function to validate a RoboDSL file."""
    validator = RoboDSLValidator()
    return validator.validate_file(file_path)


def format_robodsl_file(file_path: Path) -> str:
    """Convenience function to format a RoboDSL file."""
    linter = RoboDSLLinter()
    return linter.format_file(file_path) 