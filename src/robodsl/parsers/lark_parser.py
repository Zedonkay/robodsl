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
from ..core.ast import RoboDSLAST


class RoboDSLParser:
    """Lark-based parser for RoboDSL configuration files."""
    
    def __init__(self, debug: bool = False):
        # Load the grammar file
        grammar_file = Path(__file__).parent.parent / "grammar" / "robodsl.lark"
        with open(grammar_file, 'r') as f:
            grammar_content = f.read()
        
        # Create Lark parser
        self.parser = Lark(grammar_content, parser='lalr', start='start')
        self.ast_builder = ASTBuilder(debug=debug)
        self.semantic_analyzer = SemanticAnalyzer(debug=debug)
    
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

    def parse_with_issues(self, content: str):
        """Parse RoboDSL content and return (ast, issues). Issues is a list of dicts with keys: level, message, rule_id."""
        issues = []
        ast = None
        try:
            # Preprocess content to remove comments
            content = re.sub(r'//.*$', '', content, flags=re.MULTILINE)
            content = re.sub(r'/\*[\s\S]*?\*/', '', content)
            # Parse with Lark
            parse_tree = self.parser.parse(content)
            # Build AST
            ast = self.ast_builder.build(parse_tree)
            # Perform semantic analysis
            if not self.semantic_analyzer.analyze(ast):
                errors = self.semantic_analyzer.get_errors()
                warnings = self.semantic_analyzer.get_warnings()
                for warning in warnings:
                    issues.append({
                        'level': 'warning',
                        'message': warning,
                        'rule_id': 'semantic_warning'
                    })
                for error in errors:
                    issues.append({
                        'level': 'error',
                        'message': error,
                        'rule_id': 'semantic_error'
                    })
        except ParseError as e:
            issues.append({
                'level': 'error',
                'message': f'Parse error: {str(e)}',
                'rule_id': 'parse_error'
            })
            return None, issues
        except Exception as e:
            issues.append({
                'level': 'error',
                'message': f'Parse error: {str(e)}',
                'rule_id': 'parse_error'
            })
            return None, issues
        return ast, issues


# Global parser instance
_parser = None


def parse_robodsl(content: str, debug: bool = False) -> RoboDSLAST:
    """Parse RoboDSL content using the Lark parser.
    
    This function maintains compatibility with the existing API while using
    the new Lark-based parser.
    
    Args:
        content: The RoboDSL configuration content as a string
        debug: Enable debug output during parsing
        
    Returns:
        RoboDSLAST: The parsed AST
    """
    global _parser
    if _parser is None:
        _parser = RoboDSLParser(debug=debug)
    elif debug and not _parser.ast_builder.debug:
        # If debug flag changed, create new parser instance
        _parser = RoboDSLParser(debug=debug)
    
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