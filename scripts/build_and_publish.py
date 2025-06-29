#!/usr/bin/env python3
"""
RoboDSL Build and Publish Script

This script automates the process of building and publishing RoboDSL to PyPI.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(command, description, check=True):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    print(f"   Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Error: {e}")
        if e.stderr:
            print(f"   Stderr: {e.stderr}")
        return False


def check_prerequisites():
    """Check if all prerequisites are installed."""
    print("🔍 Checking prerequisites...")
    
    required_tools = [
        ("python", "Python"),
        ("pip", "pip"),
        ("twine", "twine"),
        ("build", "build"),
    ]
    
    missing = []
    for tool, name in required_tools:
        if not run_command(f"{tool} --version", f"Checking {name}", check=False):
            missing.append(name)
    
    if missing:
        print(f"❌ Missing prerequisites: {', '.join(missing)}")
        print("Install them with: pip install build twine")
        return False
    
    print("✅ All prerequisites found")
    return True


def clean_build():
    """Clean previous build artifacts."""
    print("🧹 Cleaning build artifacts...")
    
    dirs_to_clean = ["build", "dist", "src/robodsl.egg-info"]
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            run_command(f"rm -rf {dir_path}", f"Removing {dir_path}")
    
    print("✅ Build artifacts cleaned")


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    if not run_command("python -m pytest tests/ -v", "Running pytest"):
        print("❌ Tests failed. Aborting build.")
        return False
    
    print("✅ All tests passed")
    return True


def build_package():
    """Build the package."""
    print("📦 Building package...")
    
    # Build source distribution
    if not run_command("python -m build --sdist", "Building source distribution"):
        return False
    
    # Build wheel
    if not run_command("python -m build --wheel", "Building wheel"):
        return False
    
    print("✅ Package built successfully")
    return True


def check_package():
    """Check the built package."""
    print("🔍 Checking package...")
    
    # Check source distribution
    if not run_command("twine check dist/*.tar.gz", "Checking source distribution"):
        return False
    
    # Check wheel
    if not run_command("twine check dist/*.whl", "Checking wheel"):
        return False
    
    print("✅ Package checks passed")
    return True


def upload_to_testpypi():
    """Upload to TestPyPI."""
    print("🚀 Uploading to TestPyPI...")
    
    if not run_command("twine upload --repository testpypi dist/*", "Uploading to TestPyPI"):
        return False
    
    print("✅ Package uploaded to TestPyPI")
    print("🔗 TestPyPI URL: https://test.pypi.org/project/robodsl/")
    return True


def upload_to_pypi():
    """Upload to PyPI."""
    print("🚀 Uploading to PyPI...")
    
    if not run_command("twine upload dist/*", "Uploading to PyPI"):
        return False
    
    print("✅ Package uploaded to PyPI")
    print("🔗 PyPI URL: https://pypi.org/project/robodsl/")
    return True


def verify_upload():
    """Verify the upload by attempting to install."""
    print("🔍 Verifying upload...")
    
    # Try to install from PyPI
    if not run_command("pip install --upgrade robodsl", "Installing from PyPI", check=False):
        print("⚠️  Could not verify installation from PyPI")
        return False
    
    print("✅ Package verified on PyPI")
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Build and publish RoboDSL to PyPI")
    parser.add_argument("--test", action="store_true", 
                       help="Upload to TestPyPI instead of PyPI")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running tests")
    parser.add_argument("--skip-build", action="store_true",
                       help="Skip building (use existing dist/ files)")
    parser.add_argument("--verify", action="store_true",
                       help="Verify upload by installing from PyPI")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RoboDSL Build and Publish Script")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Clean build artifacts
    clean_build()
    
    # Run tests (unless skipped)
    if not args.skip_tests:
        if not run_tests():
            sys.exit(1)
    
    # Build package (unless skipped)
    if not args.skip_build:
        if not build_package():
            sys.exit(1)
    
    # Check package
    if not check_package():
        sys.exit(1)
    
    # Upload to appropriate repository
    if args.test:
        if not upload_to_testpypi():
            sys.exit(1)
    else:
        if not upload_to_pypi():
            sys.exit(1)
    
    # Verify upload (if requested)
    if args.verify and not args.test:
        if not verify_upload():
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Build and publish completed successfully!")
    print("=" * 60)
    
    if args.test:
        print("📦 Package uploaded to TestPyPI")
        print("🔗 URL: https://test.pypi.org/project/robodsl/")
        print("💡 To install: pip install --index-url https://test.pypi.org/simple/ robodsl")
    else:
        print("📦 Package uploaded to PyPI")
        print("🔗 URL: https://pypi.org/project/robodsl/")
        print("💡 To install: pip install robodsl")


if __name__ == "__main__":
    main() 