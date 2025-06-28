#!/usr/bin/env python3
"""
Python bridge script to integrate RoboDSL parser with VS Code language server.
This script receives RoboDSL content via stdin and returns diagnostics as JSON.
"""

import sys
import json
import traceback
from pathlib import Path

# Add the main project to the path so we can import the parser
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

try:
    from robodsl.parsers.lark_parser import RoboDSLParser
    from robodsl.parsers.semantic_analyzer import SemanticError
    from lark import ParseError
except ImportError as e:
    # If we can't import the parser, return an error diagnostic
    error_diagnostic = {
        "level": "error",
        "message": f"Failed to import RoboDSL parser: {e}",
        "rule_id": "import_error",
        "line": 0,
        "column": 0
    }
    print(json.dumps([error_diagnostic]))
    sys.exit(1)

def parse_robodsl_content(content: str):
    """Parse RoboDSL content and return diagnostics."""
    parser = RoboDSLParser()
    diagnostics = []
    
    try:
        # Parse with issues
        ast, issues = parser.parse_with_issues(content)
        
        # Convert issues to VS Code diagnostics format
        for issue in issues:
            diagnostic = {
                "level": issue["level"],
                "message": issue["message"],
                "rule_id": issue["rule_id"],
                "line": 0,  # Default to line 0, will be improved later
                "column": 0  # Default to column 0, will be improved later
            }
            diagnostics.append(diagnostic)
            
    except Exception as e:
        # Handle any unexpected errors
        error_diagnostic = {
            "level": "error",
            "message": f"Parser error: {str(e)}",
            "rule_id": "parser_error",
            "line": 0,
            "column": 0
        }
        diagnostics.append(error_diagnostic)
    
    return diagnostics

def main():
    """Main function to handle stdin/stdout communication."""
    try:
        # Read content from stdin
        content = sys.stdin.read()
        
        # Parse and get diagnostics
        diagnostics = parse_robodsl_content(content)
        
        # Output diagnostics as JSON
        print(json.dumps(diagnostics))
        
    except Exception as e:
        # Handle any errors in the main function
        error_diagnostic = {
            "level": "error",
            "message": f"Bridge error: {str(e)}",
            "rule_id": "bridge_error",
            "line": 0,
            "column": 0
        }
        print(json.dumps([error_diagnostic]))
        sys.exit(1)

if __name__ == "__main__":
    main() 