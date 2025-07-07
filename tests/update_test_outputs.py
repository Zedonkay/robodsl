#!/usr/bin/env python3
"""
Script to update all test files to use test_output directory for generated files.
"""

import re
from pathlib import Path

def update_test_file(file_path: Path):
    """Update a test file to use test_output directory."""
    content = file_path.read_text()
    
    # Pattern to find test methods that generate code
    pattern = r'def test_(\w+)\(self, test_output_dir\):\s*\n\s*"""[^"]*"""\s*\n\s*source = """\s*\n([^"]*?)\s*"""\s*\n\s*ast = parse_robodsl\(source\)\s*\n\s*generated_files = self\.generator\.generate\(ast\)'
    
    def replace_test_method(match):
        test_name = match.group(1)
        source_code = match.group(2)
        
        # Create the updated test method
        updated_method = f'''def test_{test_name}(self, test_output_dir):
        """Test {test_name} generation."""
        source = """
{source_code}
        """
        
        # Create test output directory
        test_output_dir = Path("test_output") / "{test_name}"
        test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save RoboDSL source file
        source_file = test_output_dir / "{test_name}.robodsl"
        with open(source_file, 'w') as f:
            f.write(source)
        
        ast = parse_robodsl(source)
        generated_files = self.generator.generate(ast, output_dir=str(test_output_dir))'''
        
        return updated_method
    
    # Apply the replacement
    updated_content = re.sub(pattern, replace_test_method, content, flags=re.DOTALL)
    
    # Write back the updated content
    file_path.write_text(updated_content)
    print(f"Updated {file_path}")

def main():
    """Update all test files."""
    test_files = [
        "tests/test_pipeline_validation.py",
        "tests/test_onnx_tensorrt_validation.py", 
        "tests/test_advanced_features_runner.py"
    ]
    
    for file_path in test_files:
        if Path(file_path).exists():
            update_test_file(Path(file_path))
        else:
            print(f"File not found: {file_path}")

if __name__ == "__main__":
    main() 