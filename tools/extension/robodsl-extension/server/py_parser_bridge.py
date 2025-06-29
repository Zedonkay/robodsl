#!/usr/bin/env python3
"""
Python bridge script to integrate RoboDSL parser with VS Code language server.
This script receives RoboDSL content via stdin and returns diagnostics as JSON.
"""

import sys
import json
import traceback
from pathlib import Path
import re

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
        
        # Convert issues to diagnostics with positions if available
        for issue in issues:
            line = issue.get("line")
            col = issue.get("column")
            if line is None or col is None:
                # Attempt to parse "line X column Y" from the message
                m = re.search(r"line\s+(\d+)\s+column\s+(\d+)", issue.get("message", ""))
                if m:
                    line = int(m.group(1)) - 1
                    col = int(m.group(2)) - 1
            diagnostics.append({
                "level": issue.get("level", "error"),
                "message": issue.get("message", ""),
                "rule_id": issue.get("rule_id", "unknown"),
                "line": line if line is not None else 0,
                "column": col if col is not None else 0
            })
            
    except ParseError as e:
        diagnostics.append({
            "level": "error",
            "message": f"Parse error: {str(e)}",
            "rule_id": "parse_error",
            "line": getattr(e, "line", 0) - 1 if hasattr(e, "line") else 0,
            "column": getattr(e, "column", 0) - 1 if hasattr(e, "column") else 0
        })
    
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