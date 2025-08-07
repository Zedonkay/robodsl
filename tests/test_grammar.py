from robodsl.parsers.lark_parser import parse_robodsl
from conftest import skip_if_no_ros2
#!/usr/bin/env python3
"""Simple test to validate the RoboDSL grammar."""

from pathlib import Path
from lark import Lark

def test_grammar():
    """Test if the grammar file can be loaded and parsed."""
    try:
        # Load the grammar file
        grammar_file = Path("src/robodsl/grammar/robodsl.lark")
        with open(grammar_file, 'r') as f:
            grammar_content = f.read()
        
        print("Grammar content loaded successfully")
        print(f"Grammar file length: {len(grammar_content)} characters")
        
        # Try to create the parser
        parser = Lark(grammar_content, parser='lalr', start='start')
        print("Parser created successfully")
        
        # Test with a simple input
        test_input = """
        node test_node {
            parameter int: test_param = 42
            publisher /test_topic "std_msgs/String"
            subscriber /input_topic "std_msgs/String"
        }
        """
        
        parse_tree = parse_robodsl(test_input)
        print("Test input parsed successfully")
        print(f"Parse tree: {parse_tree}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_grammar() 