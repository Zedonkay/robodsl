#!/usr/bin/env python3
"""
RoboDSL Command Line Interface

This module provides the entry point for the 'python -m robodsl' command.
"""

import argparse
import sys
from pathlib import Path

from robodsl.parser import parse_robodsl
from robodsl.generator import CodeGenerator

def generate_command(args):
    """Handle the 'generate' subcommand."""
    try:
        # Parse the input file
        with open(args.input_file, 'r') as f:
            content = f.read()
        config = parse_robodsl(content)
        
        # Determine output directory
        output_dir = Path(args.output_dir) if args.output_dir else Path('generated')
        
        # Generate the code
        generator = CodeGenerator(config, output_dir=output_dir)
        generated_files = generator.generate()
        
        # Print summary
        print(f"Generated {len(generated_files)} files:")
        for file_path in generated_files:
            print(f"  - {file_path}")
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    """Main entry point for the robodsl command-line tool."""
    parser = argparse.ArgumentParser(description='RoboDSL code generator')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate code from a RoboDSL file')
    generate_parser.add_argument('input_file', help='Input .robodsl file')
    generate_parser.add_argument('-o', '--output-dir', help='Output directory (default: generated/)')
    generate_parser.set_defaults(func=generate_command)
    
    # Parse arguments
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    # Call the appropriate function
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
