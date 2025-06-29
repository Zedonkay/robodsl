#!/usr/bin/env python3
"""
RoboDSL Installation Verification Script

This script verifies that RoboDSL and all its dependencies are properly installed.
"""

import sys
import importlib
import subprocess
from pathlib import Path


def print_status(message, status="INFO"):
    """Print a status message with color coding."""
    colors = {
        "INFO": "\033[94m",    # Blue
        "SUCCESS": "\033[92m", # Green
        "WARNING": "\033[93m", # Yellow
        "ERROR": "\033[91m",   # Red
    }
    reset = "\033[0m"
    color = colors.get(status, colors["INFO"])
    print(f"{color}[{status}]{reset} {message}")


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} ✓", "SUCCESS")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - requires 3.8+", "ERROR")
        return False


def check_package(package_name, import_name=None, version_check=None):
    """Check if a package is installed and optionally check its version."""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        if version_check:
            version = getattr(module, '__version__', 'unknown')
            if version_check(version):
                print_status(f"{package_name} {version} ✓", "SUCCESS")
                return True
            else:
                print_status(f"{package_name} {version} - version check failed", "WARNING")
                return False
        else:
            print_status(f"{package_name} ✓", "SUCCESS")
            return True
    except ImportError:
        print_status(f"{package_name} - not installed", "ERROR")
        return False


def check_command(command, description):
    """Check if a command is available."""
    try:
        result = subprocess.run([command, '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip().split('\n')[0]
            print_status(f"{description}: {version} ✓", "SUCCESS")
            return True
        else:
            print_status(f"{description}: command failed", "ERROR")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status(f"{description}: not found", "ERROR")
        return False


def check_cuda():
    """Check CUDA availability."""
    try:
        import cupy as cp
        if cp.cuda.is_available():
            device_count = cp.cuda.runtime.getDeviceCount()
            device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
            print_status(f"CUDA: {device_count} device(s), {device_name} ✓", "SUCCESS")
            return True
        else:
            print_status("CUDA: not available", "WARNING")
            return False
    except ImportError:
        print_status("CUDA: CuPy not installed", "WARNING")
        return False


def check_onnx_runtime():
    """Check ONNX Runtime availability."""
    try:
        import onnxruntime as ort
        version = ort.__version__
        providers = ort.get_available_providers()
        
        print_status(f"ONNX Runtime {version} ✓", "SUCCESS")
        print_status(f"Available providers: {', '.join(providers)}", "INFO")
        
        if 'CUDAExecutionProvider' in providers:
            print_status("ONNX Runtime CUDA support ✓", "SUCCESS")
            return True
        else:
            print_status("ONNX Runtime CUDA support not available", "WARNING")
            return False
    except ImportError:
        print_status("ONNX Runtime: not installed", "ERROR")
        return False


def check_robodsl():
    """Check RoboDSL installation."""
    try:
        import robodsl
        version = robodsl.__version__
        print_status(f"RoboDSL {version} ✓", "SUCCESS")
        
        # Test CLI
        try:
            result = subprocess.run(['robodsl', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print_status("RoboDSL CLI ✓", "SUCCESS")
                return True
            else:
                print_status("RoboDSL CLI: command failed", "ERROR")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print_status("RoboDSL CLI: not found", "ERROR")
            return False
            
    except ImportError:
        print_status("RoboDSL: not installed", "ERROR")
        return False


def check_ros2():
    """Check ROS2 availability."""
    try:
        result = subprocess.run(['ros2', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            version = result.stdout.strip()
            print_status(f"ROS2: {version} ✓", "SUCCESS")
            return True
        else:
            print_status("ROS2: command failed", "WARNING")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print_status("ROS2: not found (optional)", "WARNING")
        return False


def main():
    """Main verification function."""
    print("=" * 60)
    print("RoboDSL Installation Verification")
    print("=" * 60)
    
    checks = []
    
    # Check Python version
    checks.append(check_python_version())
    
    # Check core dependencies
    print("\nCore Dependencies:")
    checks.append(check_package("click", "click"))
    checks.append(check_package("jinja2", "jinja2"))
    checks.append(check_package("lark", "lark"))
    checks.append(check_package("numpy", "numpy"))
    
    # Check ML/AI dependencies
    print("\nMachine Learning Dependencies:")
    checks.append(check_package("opencv-python", "cv2"))
    checks.append(check_onnx_runtime())
    
    # Check CUDA support
    print("\nCUDA Support:")
    checks.append(check_cuda())
    
    # Check RoboDSL
    print("\nRoboDSL Installation:")
    checks.append(check_robodsl())
    
    # Check optional dependencies
    print("\nOptional Dependencies:")
    check_ros2()  # Optional, don't add to checks
    
    # Check build tools
    print("\nBuild Tools:")
    checks.append(check_command("cmake", "CMake"))
    checks.append(check_command("git", "Git"))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print_status(f"All {total} checks passed! ✓", "SUCCESS")
        print_status("RoboDSL is ready to use!", "SUCCESS")
        return 0
    else:
        print_status(f"{passed}/{total} checks passed", "WARNING")
        print_status("Some dependencies may be missing. Check the installation guide.", "WARNING")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 