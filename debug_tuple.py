#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, 'src')

from robodsl.parsers.lark_parser import RoboDSLParser

# Create parser with debug enabled
parser = RoboDSLParser(debug=True)

# Test content
content = """
cuda_kernels {
    kernel test {
        block_size: (256, 1, 1)
        grid_size: (1, 1, 1)
    }
}
"""

print("Parsing content...")
try:
    parse_tree = parser.parse(content)
    print("Parse tree created successfully")
    
    # Print parse tree structure
    print("Parse tree structure:")
    def print_tree(node, indent=0):
        if hasattr(node, 'data'):
            print("  " * indent + f"{node.data}: {len(node.children)} children")
            for i, child in enumerate(node.children):
                if hasattr(child, 'data'):
                    print("  " * (indent + 1) + f"[{i}] {child.data}")
                    print_tree(child, indent + 2)
                else:
                    print("  " * (indent + 1) + f"[{i}] Token: {child}")
        else:
            print("  " * indent + f"Token: {node}")
    
    print_tree(parse_tree)
    
    # Build AST
    ast = parser.ast_builder.build(parse_tree, content)
    
    print("AST built successfully")
    if ast.cuda_kernels:
        kernel = ast.cuda_kernels.kernels[0]
        print(f"block_size: {kernel.content.block_size}")
        print(f"grid_size: {kernel.content.grid_size}")
    else:
        print("No CUDA kernels found in AST")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc() 