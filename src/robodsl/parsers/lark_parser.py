"""Lark-based parser for RoboDSL configuration files.

This module replaces the regex-based parser with a proper context-free grammar parser
using Lark, providing better error handling and more robust parsing.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List
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
        self.parser = Lark(grammar_content, parser='lalr', lexer='contextual', start='start')
        self.ast_builder = ASTBuilder(debug=debug)
        self.semantic_analyzer = SemanticAnalyzer(debug=debug)
        self.debug = debug
    
    def _extract_cpp_blocks(self, content: str) -> Tuple[str, List[str]]:
        """Extract C++ code blocks and replace them with placeholders.
        
        Args:
            content: The original content
            
        Returns:
            Tuple of (processed_content, cpp_blocks)
        """
        cpp_blocks = []
        processed_content = content
        
        if self.debug:
            print(f"DEBUG: Original content: {repr(content)}")
        
        # First, handle code: { ... } and cpp: { ... } blocks with proper brace balancing
        processed_content = self._extract_balanced_blocks(processed_content, cpp_blocks, ['code:', 'cpp:'])
        
        # More specific pattern to match C++ code blocks within specific contexts
        # Look for code blocks within kernel definitions, method definitions, etc.
        patterns = [
            # Attribute-decorated function: one or more @attribute lines, then function signature and { ... }
            r'((?:@\w+\s*)+\w+\s*\([^)]*\)(?:\s*->\s*[^\{]*)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            # Template function: template<...> ...(...) ... { ... }
            r'(template\s*<[^>]*>\s*[^\s]+\s+\w+\s*\([^)]*\)(?:\s*->\s*[^\{]*)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            # Function/operator/constructor/destructor/user-defined-literal: signature then { ... }
            r'((?:def|operator\S*|\w+)\s*\([^)]*\)(?:\s*->\s*[^\{]*)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            # Template content: template<...> struct/class NAME { ... } -> template<...> struct/class NAME CPP_BLOCK_PLACEHOLDER
            r'(template\s*<[^>]*>\s*(?:struct|class)\s+\w+(?:\s*:\s*[^{{]]*)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
        ]
        
        for i, pattern in enumerate(patterns):
            if self.debug:
                print(f"DEBUG: Applying pattern {i}: {pattern}")
            
            def replace_block(match):
                # Patterns with only two groups (no suffix)
                if len(match.groups()) == 2:
                    prefix = match.group(1)
                    block_content = match.group(2)
                    placeholder = f"@@CPP_BLOCK_{len(cpp_blocks)}@@"
                    cpp_blocks.append(block_content)
                    if self.debug:
                        print(f"DEBUG: Replaced special block with placeholder: {placeholder}")
                        print(f"DEBUG: Block content: {repr(block_content)}")
                    return f"{prefix}{placeholder}"
                else:
                    prefix = match.group(1)
                    block_content = match.group(2)
                    suffix = match.group(3)
                    placeholder = f"@@CPP_BLOCK_{len(cpp_blocks)}@@"
                    cpp_blocks.append(block_content)
                    if self.debug:
                        print(f"DEBUG: Replaced block with placeholder: {placeholder}")
                        print(f"DEBUG: Block content: {repr(block_content)}")
                    return f"{prefix}{placeholder}{suffix}"
            
            processed_content = re.sub(pattern, replace_block, processed_content)
            if self.debug:
                print(f"DEBUG: After pattern {i}: {repr(processed_content)}")
        
        if self.debug:
            print(f"DEBUG: Final processed content: {repr(processed_content)}")
            print(f"DEBUG: C++ blocks: {cpp_blocks}")
        
        return processed_content, cpp_blocks
    
    def _extract_balanced_blocks(self, content: str, cpp_blocks: List[str], block_types: List[str]) -> str:
        """Extract balanced brace blocks for code: and cpp: blocks.
        
        Args:
            content: The content to process
            cpp_blocks: List to store extracted blocks
            block_types: List of block types to extract (e.g., ['code:', 'cpp:'])
            
        Returns:
            Processed content with placeholders
        """
        processed_content = content
        
        for block_type in block_types:
            # Find all occurrences of block_type followed by {
            pattern = rf'({block_type}\s*{{)'
            matches = list(re.finditer(pattern, processed_content))
            
            # Process matches in reverse order to maintain positions
            for match in reversed(matches):
                start_pos = match.end() - 1  # Position of the opening {
                
                # Find the matching closing brace
                brace_count = 0
                end_pos = start_pos
                
                for i, char in enumerate(processed_content[start_pos:], start_pos):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i
                            break
                
                if brace_count == 0:
                    # Extract the block content
                    block_content = processed_content[start_pos + 1:end_pos]
                    placeholder = f"@@CPP_BLOCK_{len(cpp_blocks)}@@"
                    cpp_blocks.append(block_content)
                    
                    if self.debug:
                        print(f"DEBUG: Extracted {block_type} block with placeholder: {placeholder}")
                        print(f"DEBUG: Block content length: {len(block_content)}")
                    
                    # Replace the block with placeholder
                    processed_content = (
                        processed_content[:start_pos + 1] + 
                        placeholder + 
                        processed_content[end_pos:]
                    )
        
        return processed_content
    
    def _restore_cpp_blocks(self, tree, cpp_blocks: List[str]):
        """Restore C++ code blocks in the parse tree.
        
        Args:
            tree: The parse tree
            cpp_blocks: List of C++ code blocks to restore
        """
        if hasattr(tree, 'children'):
            for i, child in enumerate(tree.children):
                if hasattr(child, 'data'):
                    if child.data == 'function_body':
                        # Find the placeholder in the function body
                        if child.children and hasattr(child.children[0], 'value'):
                            placeholder = child.children[0].value
                            if placeholder.startswith('@@CPP_BLOCK_') and placeholder.endswith('@@'):
                                try:
                                    block_index = int(placeholder[12:-2])  # Extract number from @@CPP_BLOCK_X@@
                                    if block_index < len(cpp_blocks):
                                        # Replace the placeholder with the actual C++ code
                                        child.children[0].value = cpp_blocks[block_index]
                                except (ValueError, IndexError):
                                    pass
                    elif child.data == 'code_block':
                        # Find the placeholder in the code block
                        if child.children and hasattr(child.children[0], 'children'):
                            # Look for placeholders in balanced_content
                            for content_child in child.children[0].children:
                                if hasattr(content_child, 'data') and content_child.data == 'balanced_content':
                                    for token_child in content_child.children:
                                        if hasattr(token_child, 'value'):
                                            placeholder = token_child.value
                                            if placeholder.startswith('@@CPP_BLOCK_') and placeholder.endswith('@@'):
                                                try:
                                                    block_index = int(placeholder[12:-2])  # Extract number from @@CPP_BLOCK_X@@
                                                    if block_index < len(cpp_blocks):
                                                        # Replace the placeholder with the actual C++ code
                                                        token_child.value = cpp_blocks[block_index]
                                                except (ValueError, IndexError):
                                                    pass
                    elif child.data == 'raw_cpp_code':
                        # Handle raw C++ code blocks specifically
                        for cpp_child in child.children:
                            if hasattr(cpp_child, 'data') and cpp_child.data == 'cpp_raw_content':
                                for content_child in cpp_child.children:
                                    if hasattr(content_child, 'value'):
                                        placeholder = content_child.value
                                        if placeholder.startswith('@@CPP_BLOCK_') and placeholder.endswith('@@'):
                                            try:
                                                block_index = int(placeholder[12:-2])  # Extract number from @@CPP_BLOCK_X@@
                                                if block_index < len(cpp_blocks):
                                                    # Replace the placeholder with the actual C++ code
                                                    content_child.value = cpp_blocks[block_index]
                                            except (ValueError, IndexError):
                                                pass
                self._restore_cpp_blocks(child, cpp_blocks)
    
    def parse(self, content: str):
        """Parse RoboDSL content and return the Lark parse tree.
        
        Args:
            content: The RoboDSL configuration content as a string
        
        Returns:
            Lark Tree: The parsed parse tree
        
        Raises:
            ParseError: If the content cannot be parsed
        """
        try:
            # Check if content contains advanced C++ features
            advanced_cpp_keywords = [
                'template<', 'static_assert', 'global ', 'def operator',
                'def __init__', 'def __del__', '#pragma',
                '#include', '#define', '#if', '#ifdef', '#ifndef', '#endif',
                '@device', '@host', 'concept ', 'friend ', 'operator""'
            ]
            has_advanced_cpp = any(keyword in content for keyword in advanced_cpp_keywords)

            # More precise bitfield detection: look for lines like 'NAME : NUMBER' or 'NAME : NAME : NUMBER' (not just any colon)
            bitfield_pattern = re.compile(r'^\s*\w+\s*:\s*\w+\s*:\s*\d+\s*;|^\s*\w+\s*:\s*\d+\s*;', re.MULTILINE)
            has_bitfield = bool(bitfield_pattern.search(content))
            has_advanced_cpp = has_advanced_cpp or has_bitfield
            
            if has_advanced_cpp:
                # For advanced C++ features, parse directly without C++ block extraction
                if self.debug:
                    print('Parsing with advanced C++ features - no block extraction')
                parse_tree = self.parser.parse(content)
            else:
                # Extract C++ code blocks and replace with placeholders for regular content
                processed_content, cpp_blocks = self._extract_cpp_blocks(content)
                if self.debug:
                    print('Processed content:', repr(processed_content))
                # Parse with Lark
                parse_tree = self.parser.parse(processed_content)
                # Restore C++ code blocks in the parse tree
                self._restore_cpp_blocks(parse_tree, cpp_blocks)
            
            return parse_tree
        except ParseError as e:
            # Provide better error messages for parse errors
            error_str = str(e)
            # Only show subnode error for actual subnode parsing issues
            if ("Unexpected token" in error_str and 
                "." in error_str and 
                "NODE_NAME" in error_str and
                "CPP_BLOCK_PLACEHOLDER" not in error_str):
                raise ParseError(
                    "Subnodes with dots (.) are not allowed in RoboDSL code. "
                    "Subnodes are a CLI-only feature for organizing files. "
                    "Use a simple node name without dots, or create subnodes using the CLI command: "
                    "'robodsl create-node <node_name>'"
                )
            raise ParseError(f"Parse error: {error_str}")
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
        
        # Parse the content to get the parse tree
        parse_tree = self.parse(content)
        
        # Build the AST from the parse tree
        ast = self.ast_builder.build(parse_tree, content)
        
        # Run semantic analysis and raise errors if found
        if not self.semantic_analyzer.analyze(ast):
            errors = self.semantic_analyzer.get_errors()
            if errors:
                # Raise all semantic errors together
                from .semantic_analyzer import SemanticError
                error_message = "; ".join(errors)
                raise SemanticError(error_message)
        
        return ast

    def parse_with_issues(self, content: str):
        """Parse RoboDSL content and return (ast, issues). Issues is a list of dicts with keys: level, message, rule_id."""
        issues = []
        ast = None
        try:
            # Parse with proper C++ block handling
            parse_tree = self.parse(content)
            # Build AST with source text for perfect C++ code preservation
            ast = self.ast_builder.build(parse_tree, content)
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
        
    Raises:
        ParseError: If the content cannot be parsed
        SemanticError: If semantic errors are found
    """
    global _parser
    if _parser is None:
        _parser = RoboDSLParser(debug=debug)
    elif debug and not _parser.ast_builder.debug:
        # If debug flag changed, create new parser instance
        _parser = RoboDSLParser(debug=debug)
    
    # Parse the content to get the parse tree
    parse_tree = _parser.parse(content)
    
    # Build the AST from the parse tree
    ast = _parser.ast_builder.build(parse_tree, content)
    
    # Run semantic analysis and raise errors if found
    if not _parser.semantic_analyzer.analyze(ast):
        errors = _parser.semantic_analyzer.get_errors()
        if errors:
            # Raise all semantic errors together
            from .semantic_analyzer import SemanticError
            error_message = "; ".join(errors)
            raise SemanticError(error_message)
    
    return ast


def parse_robodsl_file(file_path: str) -> RoboDSLAST:
    """Parse a RoboDSL file using the Lark parser.
    
    Args:
        file_path: Path to the .robodsl file
        
    Returns:
        RoboDSLAST: The parsed AST
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ParseError: If the file cannot be parsed
        SemanticError: If semantic errors are found
    """
    global _parser
    if _parser is None:
        _parser = RoboDSLParser()
    
    return _parser.parse_file(file_path) 