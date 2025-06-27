"""Lark-based parser for RoboDSL configuration files.

This module replaces the regex-based parser with a proper context-free grammar parser
using Lark, providing better error handling and more robust parsing.
"""

import os
from pathlib import Path
from typing import Optional
from lark import Lark, ParseError
import re

from .ast_builder import ASTBuilder
from .semantic_analyzer import SemanticAnalyzer, SemanticError
from ..ast import RoboDSLAST


class RoboDSLParser:
    """Lark-based parser for RoboDSL configuration files."""
    
    def __init__(self):
        # Load the grammar file
        grammar_file = Path(__file__).parent.parent / "grammar" / "robodsl.lark"
        with open(grammar_file, 'r') as f:
            grammar_content = f.read()
        
        # Create Lark parser
        self.parser = Lark(grammar_content, parser='lalr', start='start')
        self.ast_builder = ASTBuilder()
        self.semantic_analyzer = SemanticAnalyzer()
    
    def parse(self, content: str) -> RoboDSLAST:
        """Parse RoboDSL content and return AST.
        
        Args:
            content: The RoboDSL configuration content as a string
            
        Returns:
            RoboDSLAST: The parsed AST
            
        Raises:
            ParseError: If the content cannot be parsed
            SemanticError: If semantic errors are found
        """
        try:
            # Preprocess content to remove comments
            # Remove line comments
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            # Remove block comments
            content = re.sub(r'/\*[\s\S]*?\*/', '', content)
            
            # Parse with Lark
            parse_tree = self.parser.parse(content)

            # Build AST
            ast = self.ast_builder.build(parse_tree)
            
            # Perform semantic analysis
            if not self.semantic_analyzer.analyze(ast):
                errors = self.semantic_analyzer.get_errors()
                warnings = self.semantic_analyzer.get_warnings()
                
                # Print warnings
                for warning in warnings:
                    print(f"Warning: {warning}")
                
                # Raise error with all semantic errors
                raise SemanticError(f"Semantic errors found:\n" + "\n".join(f"  - {error}" for error in errors))
            
            return ast
            
        except ParseError as e:
            # Provide better error messages for parse errors
            raise ParseError(f"Parse error: {str(e)}")
        except SemanticError as e:
            # Re-raise semantic errors as-is
            raise e
        except Exception as e:
            raise ParseError(f"Parse error: {str(e)}")
    
    def parse_file(self, file_path: str) -> RoboDSLAST:
        """Parse a RoboDSL file and return AST.
        
        Args:
            file_path: Path to the .robodsl file
            
        Returns:
            RoboDSLAST: The parsed AST
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ParseError: If the file cannot be parsed
            SemanticError: If semantic errors are found
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return self.parse(content)


# Global parser instance
_parser = None


def parse_robodsl(content: str) -> RoboDSLAST:
    """Parse RoboDSL content using the Lark parser.
    
    This function maintains compatibility with the existing API while using
    the new Lark-based parser.
    
    Args:
        content: The RoboDSL configuration content as a string
        
    Returns:
        RoboDSLAST: The parsed AST
    """
    global _parser
    if _parser is None:
        _parser = RoboDSLParser()
    
    return _parser.parse(content)


def parse_robodsl_file(file_path: str) -> RoboDSLAST:
    """Parse a RoboDSL file using the Lark parser.
    
    Args:
        file_path: Path to the .robodsl file
        
    Returns:
        RoboDSLAST: The parsed AST
    """
    global _parser
    if _parser is None:
        _parser = RoboDSLParser()
    
    return _parser.parse_file(file_path) 