#!/usr/bin/env python3
"""
Cleanup script to move generated files from main directory to test_output.

This script identifies and moves any generated files that accidentally end up
in the main project directory instead of the test_output directory.
"""

import os
import shutil
from pathlib import Path

def cleanup_generated_files():
    """Move generated files from main directory to test_output."""
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    test_output_dir = project_root / "test_output"
    
    # Create test_output directory if it doesn't exist
    test_output_dir.mkdir(exist_ok=True)
    
    # Define file patterns that are typically generated
    generated_patterns = [
        "*.cpp",
        "*.hpp", 
        "*.cmake",
        "*.launch.py",
        "*.yaml",
        "*.xml",
        "CMakeLists.txt",
        "package.xml"
    ]
    
    # Directories that should not be moved (they belong in the main directory)
    protected_dirs = {
        "src", "include", "config", "launch", "docs", "examples", 
        "tests", "tools", ".git", ".venv", ".pytest_cache", "build",
        "dist", "CMakeFiles", "test_output"
    }
    
    # Files that should not be moved (they belong in the main directory)
    protected_files = {
        "README.md", "LICENSE", "setup.py", "pyproject.toml", 
        "requirements-dev.txt", "pytest.ini", "MANIFEST.in",
        ".gitignore", ".DS_Store", "CMakeCache.txt", "package.xml", "CMakeLists.txt"
    }
    
    moved_files = []
    
    print("Scanning for generated files in main directory...")
    
    # Check for generated files in the main directory
    for pattern in generated_patterns:
        for file_path in project_root.glob(pattern):
            # Skip if it's in a protected directory
            if any(protected_dir in file_path.parts for protected_dir in protected_dirs):
                continue
                
            # Skip if it's a protected file
            if file_path.name in protected_files:
                continue
                
            # Skip if it's already in test_output
            if "test_output" in file_path.parts:
                continue
                
            # Move the file to test_output
            target_path = test_output_dir / file_path.name
            
            # Handle filename conflicts
            counter = 1
            original_target = target_path
            while target_path.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_path = test_output_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            try:
                shutil.move(str(file_path), str(target_path))
                moved_files.append((file_path.name, target_path))
                print(f"Moved: {file_path.name} -> {target_path}")
            except Exception as e:
                print(f"Error moving {file_path.name}: {e}")
    
    # Check for generated directories that might have been created
    for item in project_root.iterdir():
        if item.is_dir() and item.name not in protected_dirs:
            # Check if this looks like a generated directory
            if any(pattern in item.name.lower() for pattern in ["node", "model", "pipeline", "test", "output"]):
                target_dir = test_output_dir / item.name
                
                # Handle directory name conflicts
                counter = 1
                original_target = target_dir
                while target_dir.exists():
                    target_dir = test_output_dir / f"{item.name}_{counter}"
                    counter += 1
                
                try:
                    shutil.move(str(item), str(target_dir))
                    moved_files.append((item.name, target_dir))
                    print(f"Moved directory: {item.name} -> {target_dir}")
                except Exception as e:
                    print(f"Error moving directory {item.name}: {e}")
    
    if moved_files:
        print(f"\nMoved {len(moved_files)} files/directories to test_output/")
        print("Files moved:")
        for name, target in moved_files:
            print(f"  - {name} -> {target}")
    else:
        print("No generated files found in main directory.")
    
    return len(moved_files)

if __name__ == "__main__":
    cleanup_generated_files() 