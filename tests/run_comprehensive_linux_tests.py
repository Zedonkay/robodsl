#!/usr/bin/env python3
"""Comprehensive Linux dependency test runner.

This script runs all comprehensive tests for Linux dependencies including:
- CUDA and GPU acceleration
- ROS2 integration
- TensorRT optimization
- ONNX Runtime
- CMake build system
- Advanced features and edge cases

Usage:
    python run_comprehensive_linux_tests.py [options]

Options:
    --verbose, -v: Enable verbose output
    --parallel, -p: Run tests in parallel
    --report, -r: Generate detailed report
    --coverage, -c: Generate coverage report
    --timeout, -t: Set test timeout in seconds
    --filter, -f: Filter tests by pattern
"""

import argparse
import sys
import os
import subprocess
import time
import json
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import pytest


class ComprehensiveTestRunner:
    """Comprehensive test runner for Linux dependencies."""
    
    def __init__(self, args):
        """Initialize test runner."""
        self.args = args
        self.test_results = {}
        self.test_stats = {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0
        }
        self.start_time = time.time()
        
    def run_tests(self) -> int:
        """Run all comprehensive tests."""
        print("🚀 Starting Comprehensive Linux Dependency Tests")
        print("=" * 60)
        
        # Check system dependencies
        self._check_system_dependencies()
        
        # Define test modules
        test_modules = [
            "test_comprehensive_linux_dependencies",
            "test_cuda_advanced_features",
            "test_tensorrt_advanced_features",
            "test_ros2_advanced_features",
            "test_onnx_advanced_features",
            "test_cmake_advanced_features"
        ]
        
        # Run tests
        if self.args.parallel:
            return self._run_tests_parallel(test_modules)
        else:
            return self._run_tests_sequential(test_modules)
    
    def _check_system_dependencies(self):
        """Check system dependencies."""
        print("🔍 Checking system dependencies...")
        
        dependencies = {
            "CUDA": self._check_cuda(),
            "TensorRT": self._check_tensorrt(),
            "ROS2": self._check_ros2(),
            "ONNX Runtime": self._check_onnx(),
            "CMake": self._check_cmake(),
            "Python": self._check_python(),
            "Compiler": self._check_compiler()
        }
        
        print("\n📋 System Dependencies:")
        for dep, available in dependencies.items():
            status = "✅ Available" if available else "❌ Not Available"
            print(f"  {dep}: {status}")
        
        # Store dependency info for tests
        self.dependencies = dependencies
    
    def _check_cuda(self) -> bool:
        """Check CUDA availability."""
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_tensorrt(self) -> bool:
        """Check TensorRT availability."""
        try:
            import ctypes
            ctypes.CDLL('libnvinfer.so')
            return True
        except (OSError, ImportError):
            return False
    
    def _check_ros2(self) -> bool:
        """Check ROS2 availability."""
        try:
            result = subprocess.run(['ros2', '--version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_onnx(self) -> bool:
        """Check ONNX Runtime availability."""
        try:
            import onnxruntime
            return True
        except ImportError:
            return False
    
    def _check_cmake(self) -> bool:
        """Check CMake availability."""
        try:
            result = subprocess.run(['cmake', '--version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _check_python(self) -> bool:
        """Check Python version."""
        return sys.version_info >= (3, 8)
    
    def _check_compiler(self) -> bool:
        """Check C++ compiler availability."""
        try:
            result = subprocess.run(['gcc', '--version'], 
                                 capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False
    
    def _run_tests_sequential(self, test_modules: List[str]) -> int:
        """Run tests sequentially."""
        print(f"\n🧪 Running {len(test_modules)} test modules sequentially...")
        
        total_exit_code = 0
        
        for module in test_modules:
            print(f"\n📦 Testing module: {module}")
            exit_code = self._run_test_module(module)
            total_exit_code |= exit_code
            
            if exit_code == 0:
                print(f"✅ {module}: PASSED")
            else:
                print(f"❌ {module}: FAILED")
        
        return total_exit_code
    
    def _run_tests_parallel(self, test_modules: List[str]) -> int:
        """Run tests in parallel."""
        print(f"\n🧪 Running {len(test_modules)} test modules in parallel...")
        
        with ThreadPoolExecutor(max_workers=min(len(test_modules), 4)) as executor:
            future_to_module = {
                executor.submit(self._run_test_module, module): module 
                for module in test_modules
            }
            
            total_exit_code = 0
            
            for future in as_completed(future_to_module):
                module = future_to_module[future]
                try:
                    exit_code = future.result()
                    total_exit_code |= exit_code
                    
                    if exit_code == 0:
                        print(f"✅ {module}: PASSED")
                    else:
                        print(f"❌ {module}: FAILED")
                        
                except Exception as e:
                    print(f"❌ {module}: ERROR - {e}")
                    total_exit_code |= 1
        
        return total_exit_code
    
    def _run_test_module(self, module: str) -> int:
        """Run a single test module."""
        test_file = f"tests/{module}.py"
        
        if not os.path.exists(test_file):
            print(f"⚠️  Test file not found: {test_file}")
            return 1
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest", test_file,
            "-v" if self.args.verbose else "-q",
            "--tb=short",
            "--durations=10"
        ]
        
        if self.args.timeout:
            cmd.extend(["--timeout", str(self.args.timeout)])
        
        if self.args.filter:
            cmd.extend(["-k", self.args.filter])
        
        # Run the test
        try:
            result = subprocess.run(
                cmd,
                capture_output=not self.args.verbose,
                text=True,
                timeout=self.args.timeout or 300
            )
            
            # Store results
            self.test_results[module] = {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": 0  # Would need to measure this
            }
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            print(f"⏰ {module}: TIMEOUT")
            return 1
        except Exception as e:
            print(f"💥 {module}: ERROR - {e}")
            return 1
    
    def generate_report(self):
        """Generate comprehensive test report."""
        if not self.args.report:
            return
        
        print("\n📊 Generating comprehensive test report...")
        
        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if result["exit_code"] == 0)
        failed_tests = total_tests - passed_tests
        
        # Generate report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "duration": time.time() - self.start_time,
            "system_info": self._get_system_info(),
            "dependencies": self.dependencies,
            "test_results": self.test_results,
            "statistics": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            }
        }
        
        # Save report
        report_file = "test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📄 Report saved to: {report_file}")
        
        # Print summary
        self._print_summary(report)
    
    def _get_system_info(self) -> Dict:
        """Get system information."""
        import platform
        
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "architecture": platform.architecture()[0],
            "processor": platform.processor(),
            "machine": platform.machine()
        }
    
    def _print_summary(self, report: Dict):
        """Print test summary."""
        stats = report["statistics"]
        
        print("\n" + "=" * 60)
        print("📈 TEST SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {stats['total']}")
        print(f"Passed: {stats['passed']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Duration: {report['duration']:.2f} seconds")
        
        if stats['failed'] > 0:
            print("\n❌ Failed Tests:")
            for module, result in self.test_results.items():
                if result["exit_code"] != 0:
                    print(f"  - {module}")
        
        print("\n" + "=" * 60)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Linux dependency test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_comprehensive_linux_tests.py
  python run_comprehensive_linux_tests.py --verbose --parallel
  python run_comprehensive_linux_tests.py --report --timeout 600
  python run_comprehensive_linux_tests.py --filter "cuda"
        """
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "-p", "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    
    parser.add_argument(
        "-r", "--report",
        action="store_true",
        help="Generate detailed report"
    )
    
    parser.add_argument(
        "-c", "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    
    parser.add_argument(
        "-t", "--timeout",
        type=int,
        default=300,
        help="Set test timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "-f", "--filter",
        type=str,
        help="Filter tests by pattern"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = ComprehensiveTestRunner(args)
    
    try:
        # Run tests
        exit_code = runner.run_tests()
        
        # Generate report
        runner.generate_report()
        
        # Exit with appropriate code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        print("\n⚠️  Test run interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 