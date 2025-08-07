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
        if self.debug:
            print(f"DEBUG: After _extract_balanced_blocks: {repr(processed_content)}")
            print(f"DEBUG: C++ blocks after balanced: {cpp_blocks}")
        
        # Handle kernel: | blocks (YAML-style block scalars)
        processed_content = self._extract_kernel_blocks(processed_content, cpp_blocks)
        if self.debug:
            print(f"DEBUG: After _extract_kernel_blocks: {repr(processed_content)}")
            print(f"DEBUG: C++ blocks after kernel: {cpp_blocks}")
        
        # Extract method bodies with C++ code (most common case)
        processed_content = self._extract_method_bodies(processed_content, cpp_blocks)
        if self.debug:
            print(f"DEBUG: After _extract_method_bodies: {repr(processed_content)}")
            print(f"DEBUG: C++ blocks after method bodies: {cpp_blocks}")
        
        # More specific pattern to match C++ code blocks within specific contexts
        # Look for code blocks within kernel definitions, method definitions, etc.
        patterns = [
            # Attribute-decorated function: one or more @attribute lines, then function signature and { ... }
            r'((?:@\w+\s*)+\w+\s*\([^)]*\)(?:\s*->\s*[^\{]*)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            # Template function: template<...> ...(...) ... { ... }
            r'(template\s*<[^>]*>\s*[^\s]+\s+\w+\s*\([^)]*\)(?:\s*->\s*[^\{]*)?\s*)\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
            # Function/operator/constructor/destructor/user-defined-literal: signature then { ... }
            r'((?:def|operator\S*|\w+)\s*\([^)]*\)(?:\s*->\s*[^\{]*)?)\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}',
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
                    placeholder = "CPP_BLOCK_PLACEHOLDER"
                    cpp_blocks.append(block_content)
                    if self.debug:
                        print(f"DEBUG: Replaced special block with placeholder: {placeholder}")
                        print(f"DEBUG: Block content: {repr(block_content)}")
                    # For template/class/struct, wrap placeholder in braces
                    if 'struct' in prefix or 'class' in prefix:
                        return f"{prefix}{{ {placeholder} }}"
                    # Always insert a space before the placeholder for other cases
                    return f"{prefix} {placeholder}"
                else:
                    prefix = match.group(1)
                    block_content = match.group(2)
                    suffix = match.group(3)
                    placeholder = "CPP_BLOCK_PLACEHOLDER"
                    cpp_blocks.append(block_content)
                    if self.debug:
                        print(f"DEBUG: Replaced block with placeholder: {placeholder}")
                        print(f"DEBUG: Block content: {repr(block_content)}")
                    return f"{prefix} {placeholder}{suffix}"
            
            processed_content = re.sub(pattern, replace_block, processed_content)
            if self.debug:
                print(f"DEBUG: After pattern {i}: {repr(processed_content)}")
        
        # Reverse the blocks to match the extraction order (last extracted = first in list)
        cpp_blocks.reverse()
        
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
                    placeholder = "CPP_BLOCK_PLACEHOLDER"
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

    def _extract_kernel_blocks(self, content: str, cpp_blocks: List[str]) -> str:
        """Extract kernel: | blocks (YAML-style block scalars) and replace with placeholders.
        
        Args:
            content: The content to process
            cpp_blocks: List to store extracted blocks
            
        Returns:
            Processed content with placeholders
        """
        if content is None:
            return ""
            
        processed_content = content
        
        # Pattern to match kernel: | followed by indented code (multi-line)
        pattern1 = r'(kernel:\s*\|)(\s*\n(\s+)[^\n]*(\n\3[^\n]*)*)'
        matches1 = list(re.finditer(pattern1, processed_content, re.MULTILINE))
        
        # Pattern to match kernel: | followed by code on the same line
        pattern2 = r'(kernel:\s*\|)(\s*[^{}]*(?:\{[^{}]*\}[^{}]*)*)'
        matches2 = list(re.finditer(pattern2, processed_content))
        
        # Process multi-line matches in reverse order to maintain positions
        for match in reversed(matches1):
            prefix = match.group(1)  # "kernel: |"
            code_block = match.group(2)  # The entire indented code block
            
            # Extract the actual code content (remove the leading whitespace from each line)
            lines = code_block.split('\n')
            code_lines = []
            for line in lines:
                if line.strip():  # Skip empty lines
                    # Remove the common indentation
                    code_lines.append(line)
            
            # Join the code lines
            code_content = '\n'.join(code_lines)
            
            # Store the code block
            cpp_blocks.append(code_content)
            
            # Replace with placeholder
            placeholder = "CPP_BLOCK_PLACEHOLDER"
            processed_content = (
                processed_content[:match.start()] +
                prefix + " " + placeholder +
                processed_content[match.end():]
            )
        
        # Process single-line matches in reverse order to maintain positions
        for match in reversed(matches2):
            prefix = match.group(1)  # "kernel: |"
            code_block = match.group(2)  # The code on the same line
            
            # Find the end of the function body by balancing braces
            start_pos = match.end()
            brace_count = 0
            end_pos = start_pos
            
            for i, char in enumerate(processed_content[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_pos = i + 1
                        break
            
            # Extract the complete function body
            complete_code = processed_content[match.start():end_pos]
            # Remove the "kernel: |" prefix to get just the code
            code_content = complete_code[len(prefix):].strip()
            
            # Store the code block
            cpp_blocks.append(code_content)
            
            # Replace with placeholder
            placeholder = "CPP_BLOCK_PLACEHOLDER"
            processed_content = (
                processed_content[:match.start()] +
                prefix + " " + placeholder +
                processed_content[end_pos:]
            )
        
        return processed_content

    def _extract_method_bodies(self, content: str, cpp_blocks: List[str]) -> str:
        """Extract method bodies that contain C++ code and replace with placeholders.
        Args:
            content: The content to process
            cpp_blocks: List to store extracted blocks
        Returns:
            Processed content with placeholders
        """
        processed_content = content
        pattern = re.compile(r'(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^\{]*)?\s*)\{', re.MULTILINE)
        matches = list(pattern.finditer(processed_content))
        # Process matches in reverse order to not mess up indices
        for match in reversed(matches):
            start_pos = match.end() - 1  # position of the opening brace
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
                body = processed_content[start_pos + 1:end_pos]
                cpp_blocks.append(body)
                # Replace the body with placeholder
                processed_content = (
                    processed_content[:start_pos + 1] +
                    ' CPP_BLOCK_PLACEHOLDER ' +
                    processed_content[end_pos:]
                )
        return processed_content
    
    def _restore_cpp_blocks(self, tree, cpp_blocks: List[str]):
        """Restore C++ code blocks in the parse tree.
        
        Args:
            tree: The parse tree
            cpp_blocks: List of C++ code blocks to restore
        """
        block_index = 0  # Track which block we're currently processing
        
        def restore_blocks_recursive(node):
            nonlocal block_index
            if hasattr(node, 'children'):
                for i, child in enumerate(node.children):
                    if hasattr(child, 'data'):
                        if child.data == 'function_body':
                            # Find the placeholder in the function body
                            if child.children and hasattr(child.children[0], 'value'):
                                placeholder = child.children[0].value
                                if placeholder == 'CPP_BLOCK_PLACEHOLDER':
                                    try:
                                        if block_index < len(cpp_blocks):
                                            # Replace the placeholder with the actual C++ code
                                            child.children[0].value = cpp_blocks[block_index]
                                            block_index += 1
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
                                                if placeholder == 'CPP_BLOCK_PLACEHOLDER':
                                                    if block_index < len(cpp_blocks):
                                                        # Replace the placeholder with the actual C++ code
                                                        token_child.value = cpp_blocks[block_index]
                                                        block_index += 1
                        elif child.data == 'raw_cpp_code':
                            # Handle raw C++ code blocks specifically
                            for cpp_child in child.children:
                                if hasattr(cpp_child, 'data') and cpp_child.data == 'cpp_raw_content':
                                    for content_child in cpp_child.children:
                                        if hasattr(content_child, 'value'):
                                            placeholder = content_child.value
                                            if placeholder == 'CPP_BLOCK_PLACEHOLDER':
                                                try:
                                                    if block_index < len(cpp_blocks):
                                                        # Replace the placeholder with the actual C++ code
                                                        content_child.value = cpp_blocks[block_index]
                                                        block_index += 1
                                                except (ValueError, IndexError):
                                                    pass
                        elif child.data == 'kernel':
                            # Handle kernel blocks specifically
                            for kernel_child in child.children:
                                if hasattr(kernel_child, 'data') and kernel_child.data == 'cpp_raw_content':
                                    for content_child in kernel_child.children:
                                        if hasattr(content_child, 'value'):
                                            placeholder = content_child.value
                                            if placeholder == 'CPP_BLOCK_PLACEHOLDER':
                                                try:
                                                    if block_index < len(cpp_blocks):
                                                        # Replace the placeholder with the actual C++ code
                                                        content_child.value = cpp_blocks[block_index]
                                                        block_index += 1
                                                except (ValueError, IndexError):
                                                    pass
                    restore_blocks_recursive(child)
        
        restore_blocks_recursive(tree)
    
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
            # Check if content contains advanced C++ features or C++ code blocks
            advanced_cpp_keywords = [
                'template<', 'static_assert', 'global ', 'def operator',
                'def __init__', 'def __del__', '#pragma',
                '#include', '#define', '#if', '#ifdef', '#ifndef', '#endif',
                '@device', '@host', 'concept ', 'friend ', 'operator""'
            ]
            has_advanced_cpp = any(keyword in content for keyword in advanced_cpp_keywords)

            # Check for explicit C++ code blocks
            has_cpp_blocks = 'cpp:' in content or 'code:' in content or 'kernel: |' in content

            # Check for C++ code in method bodies (return statements, etc.)
            # Only detect as C++ blocks if they contain complex C++ syntax
            cpp_in_methods = re.search(r'def\s+\w+\s*\([^)]*\)[^{]*\{[^}]*return\s+[^;]+;', content, re.DOTALL)
            has_cpp_blocks = has_cpp_blocks or bool(cpp_in_methods)
            
            # Always treat method bodies with C++ code as advanced C++ features
            if cpp_in_methods:
                has_advanced_cpp = True
                
            # Also check for any method with C++ code (not just return statements)
            cpp_method_pattern = re.search(r'def\s+\w+\s*\([^)]*\)[^{]*\{[^}]*[a-zA-Z_][a-zA-Z0-9_]*\s*[=+\-*/<>!&|^~%]\s*[^;]+;', content, re.DOTALL)
            if cpp_method_pattern:
                has_advanced_cpp = True
                
            # Check for any method with C++ code that should be extracted
            if 'def' in content and '{' in content and '}' in content:
                # Look for method definitions with C++ code
                method_pattern = re.search(r'def\s+\w+\s*\([^)]*\)[^{]*\{[^}]*\}', content, re.DOTALL)
                if method_pattern:
                    method_body = method_pattern.group(0)
                    # If the method body contains C++ code (not just simple RoboDSL syntax)
                    cpp_indicators = ['return', 'std::', 'auto', 'const', 'for', 'if', 'while', 'switch', 'case', 'break', 'continue']
                    if any(keyword in method_body for keyword in cpp_indicators):
                        has_advanced_cpp = True
                        has_cpp_blocks = True

            # More precise bitfield detection: look for lines like 'NAME : NUMBER' or 'NAME : NAME : NUMBER'
            # (not just any colon)
            bitfield_pattern = re.compile(r'^\s*\w+\s*:\s*\w+\s*:\s*\d+\s*;|^\s*\w+\s*:\s*\d+\s*;', re.MULTILINE)
            has_bitfield = bool(bitfield_pattern.search(content))
            has_advanced_cpp = has_advanced_cpp or has_bitfield

            if has_cpp_blocks:
                # Extract C++ code blocks before parsing
                if self.debug:
                    print('Parsing with C++ blocks - extracting C++ blocks first')
                processed_content, cpp_blocks = self._extract_cpp_blocks(content)
                parse_tree = self.parser.parse(processed_content)
                # Restore C++ blocks in the parse tree
                self._restore_cpp_blocks(parse_tree, cpp_blocks)
            else:
                # Parse directly without extracting blocks
                if self.debug:
                    print('Parsing regular content - no C++ block extraction needed')
                parse_tree = self.parser.parse(content)
            
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