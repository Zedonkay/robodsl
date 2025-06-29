#!/usr/bin/env python3
"""
RoboDSL PyPI Setup Script

This script helps set up PyPI credentials and configuration for publishing.
"""

import os
import sys
import getpass
import configparser
from pathlib import Path


def create_pypirc():
    """Create or update .pypirc file with PyPI credentials."""
    print("🔧 Setting up PyPI configuration...")
    
    pypirc_path = Path.home() / ".pypirc"
    
    # Check if .pypirc already exists
    if pypirc_path.exists():
        print(f"📁 Found existing .pypirc at {pypirc_path}")
        response = input("Do you want to overwrite it? (y/N): ").lower()
        if response != 'y':
            print("❌ Setup cancelled")
            return False
    
    # Get PyPI credentials
    print("\n🔑 PyPI Credentials Setup")
    print("You'll need to create an account at https://pypi.org/account/register/")
    print("and get an API token from https://pypi.org/manage/account/token/\n")
    
    username = input("PyPI Username: ").strip()
    if not username:
        print("❌ Username is required")
        return False
    
    password = getpass.getpass("PyPI API Token (password): ")
    if not password:
        print("❌ API token is required")
        return False
    
    # Get TestPyPI credentials (optional)
    print("\n🧪 TestPyPI Credentials Setup (optional)")
    print("You can create an account at https://test.pypi.org/account/register/")
    print("and get an API token from https://test.pypi.org/manage/account/token/\n")
    
    test_username = input("TestPyPI Username (leave empty to skip): ").strip()
    test_password = ""
    if test_username:
        test_password = getpass.getpass("TestPyPI API Token: ")
    
    # Create .pypirc content
    pypirc_content = f"""[distutils]
index-servers =
    pypi
    testpypi

[pypi]
repository = https://upload.pypi.org/legacy/
username = {username}
password = {password}

[testpypi]
repository = https://test.pypi.org/legacy/
username = {test_username}
password = {test_password}
"""
    
    # Write .pypirc file
    try:
        with open(pypirc_path, 'w') as f:
            f.write(pypirc_content)
        
        # Set proper permissions
        os.chmod(pypirc_path, 0o600)
        
        print(f"✅ PyPI configuration saved to {pypirc_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating .pypirc: {e}")
        return False


def install_build_tools():
    """Install required build tools."""
    print("📦 Installing build tools...")
    
    tools = ["build", "twine"]
    
    for tool in tools:
        print(f"   Installing {tool}...")
        try:
            import subprocess
            result = subprocess.run([sys.executable, "-m", "pip", "install", tool], 
                                  capture_output=True, text=True, check=True)
            print(f"   ✅ {tool} installed")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Failed to install {tool}: {e}")
            return False
    
    print("✅ All build tools installed")
    return True


def verify_setup():
    """Verify the PyPI setup."""
    print("🔍 Verifying setup...")
    
    # Check .pypirc
    pypirc_path = Path.home() / ".pypirc"
    if not pypirc_path.exists():
        print("❌ .pypirc file not found")
        return False
    
    # Check build tools
    try:
        import build
        import twine
        print("✅ Build tools available")
    except ImportError:
        print("❌ Build tools not available")
        return False
    
    # Test PyPI connection
    print("🌐 Testing PyPI connection...")
    try:
        import subprocess
        result = subprocess.run(["twine", "check", "--help"], 
                              capture_output=True, text=True, check=True)
        print("✅ Twine working correctly")
    except subprocess.CalledProcessError:
        print("❌ Twine not working correctly")
        return False
    
    print("✅ PyPI setup verified")
    return True


def show_next_steps():
    """Show next steps for publishing."""
    print("\n" + "=" * 60)
    print("🎯 Next Steps for Publishing")
    print("=" * 60)
    
    print("1. 📝 Update version in pyproject.toml and setup.py")
    print("2. 🧪 Test your package locally:")
    print("   python -m build")
    print("   twine check dist/*")
    print("3. 🧪 Test upload to TestPyPI:")
    print("   python scripts/build_and_publish.py --test")
    print("4. 🚀 Publish to PyPI:")
    print("   python scripts/build_and_publish.py")
    print("5. ✅ Verify installation:")
    print("   pip install robodsl")
    print("   robodsl --version")
    
    print("\n📚 Useful Resources:")
    print("   - PyPI: https://pypi.org/")
    print("   - TestPyPI: https://test.pypi.org/")
    print("   - Python Packaging Guide: https://packaging.python.org/")
    print("   - Twine Documentation: https://twine.readthedocs.io/")


def main():
    """Main function."""
    print("=" * 60)
    print("RoboDSL PyPI Setup")
    print("=" * 60)
    
    # Install build tools
    if not install_build_tools():
        sys.exit(1)
    
    # Create .pypirc
    if not create_pypirc():
        sys.exit(1)
    
    # Verify setup
    if not verify_setup():
        sys.exit(1)
    
    # Show next steps
    show_next_steps()
    
    print("\n🎉 PyPI setup completed successfully!")


if __name__ == "__main__":
    main() 