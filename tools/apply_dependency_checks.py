#!/usr/bin/env python3
"""Script to apply dependency checks to all test files."""

import os
import re
from pathlib import Path

def add_dependency_checks_to_file(file_path):
    """Add dependency checks to a test file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Check if already has dependency checks
    if 'from conftest import' in content:
        print(f"Skipping {file_path}: already has dependency checks")
        return
    
    # Determine which dependencies are needed based on file content
    needs_ros2 = any(keyword in content.lower() for keyword in [
        'rclcpp', 'ros2', 'node', 'publisher', 'subscriber', 'service', 'action'
    ])
    
    needs_cuda = any(keyword in content.lower() for keyword in [
        'cuda', 'kernel', 'nvcc', '__global__', 'cudamalloc', 'cudafree'
    ])
    
    needs_tensorrt = any(keyword in content.lower() for keyword in [
        'tensorrt', 'trt', 'nvinfer', 'onnx'
    ])
    
    needs_onnx = any(keyword in content.lower() for keyword in [
        'onnx', 'onnxruntime'
    ])
    
    # Add imports
    import_line = "from conftest import "
    imports = []
    if needs_ros2:
        imports.append("skip_if_no_ros2")
    if needs_cuda:
        imports.append("skip_if_no_cuda")
    if needs_tensorrt:
        imports.append("skip_if_no_tensorrt")
    if needs_onnx:
        imports.append("skip_if_no_onnx")
    
    if not imports:
        print(f"Skipping {file_path}: no dependencies detected")
        return
    
    import_line += ", ".join(imports)
    
    # Find the right place to insert the import
    lines = content.split('\n')
    insert_pos = None
    
    # Look for existing imports
    for i, line in enumerate(lines):
        if line.startswith('from robodsl.') or line.startswith('import robodsl'):
            insert_pos = i + 1
            break
    
    if insert_pos is None:
        # Look for any import
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                insert_pos = i + 1
                break
    
    if insert_pos is None:
        insert_pos = 0
    
    # Insert the import
    lines.insert(insert_pos, import_line)
    
    # Add skip calls to test methods
    new_lines = []
    for line in lines:
        new_lines.append(line)
        
        # Look for test method definitions
        if re.match(r'^\s+def test_', line):
            # Find the method name
            match = re.match(r'^\s+def (test_\w+)', line)
            if match:
                method_name = match.group(1)
                
                # Add appropriate skip calls
                skip_calls = []
                if needs_ros2 and 'cuda' not in method_name.lower():
                    skip_calls.append("skip_if_no_ros2()")
                if needs_cuda and 'cuda' in method_name.lower():
                    skip_calls.append("skip_if_no_cuda()")
                if needs_tensorrt and 'tensorrt' in method_name.lower():
                    skip_calls.append("skip_if_no_tensorrt()")
                if needs_onnx and 'onnx' in method_name.lower():
                    skip_calls.append("skip_if_no_onnx()")
                
                if skip_calls:
                    # Add skip calls after the method definition line
                    indent = len(line) - len(line.lstrip())
                    for skip_call in skip_calls:
                        new_lines.append(" " * (indent + 4) + skip_call)
    
    # Write back to file
    with open(file_path, 'w') as f:
        f.write('\n'.join(new_lines))
    
    print(f"Updated {file_path} with dependencies: {', '.join(imports)}")

def main():
    """Apply dependency checks to all test files."""
    tests_dir = Path(__file__).parent.parent / 'tests'
    
    # Find all test files
    test_files = []
    for pattern in ['test_*.py', '*_test.py']:
        test_files.extend(tests_dir.glob(pattern))
    
    # Skip conftest.py and other non-test files
    test_files = [f for f in test_files if f.name != 'conftest.py' and not f.name.startswith('run_')]
    
    print(f"Found {len(test_files)} test files")
    
    for test_file in test_files:
        try:
            add_dependency_checks_to_file(test_file)
        except Exception as e:
            print(f"Error processing {test_file}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main() 