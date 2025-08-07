#!/usr/bin/env python3
"""C++ Code Validation Test Runner.

This script runs all C++ validation tests and provides a comprehensive report
on the quality of generated C++ code.
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def run_pytest_tests(test_files: List[str], verbose: bool = False) -> Dict[str, Any]:
    """Run pytest tests and return results."""
    results = {}
    
    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Running {test_file}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = ['python', '-m', 'pytest', test_file]
        if verbose:
            cmd.extend(['-v', '-s'])
        else:
            cmd.extend(['-v'])
        
        try:
            # Run the test
            result = subprocess.run(
                cmd,
                capture_output=not verbose,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            results[test_file] = {
                'returncode': result.returncode,
                'stdout': result.stdout if not verbose else '',
                'stderr': result.stderr if not verbose else '',
                'duration': duration,
                'success': result.returncode == 0
            }
            
            if result.returncode == 0:
                print(f"✅ {test_file} PASSED ({duration:.2f}s)")
            else:
                print(f"❌ {test_file} FAILED ({duration:.2f}s)")
                if not verbose:
                    print(f"Error output:\n{result.stderr}")
            
        except subprocess.TimeoutExpired:
            results[test_file] = {
                'returncode': -1,
                'stdout': '',
                'stderr': 'Test timed out after 5 minutes',
                'duration': 300,
                'success': False
            }
            print(f"⏰ {test_file} TIMEOUT")
            
        except Exception as e:
            results[test_file] = {
                'returncode': -1,
                'stdout': '',
                'stderr': str(e),
                'duration': 0,
                'success': False
            }
            print(f"💥 {test_file} ERROR: {e}")
    
    return results


def generate_summary_report(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary report from test results."""
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    failed_tests = total_tests - passed_tests
    
    total_duration = sum(r['duration'] for r in results.values())
    
    # Categorize failures
    failures = []
    for test_file, result in results.items():
        if not result['success']:
            failures.append({
                'test_file': test_file,
                'error': result['stderr'],
                'duration': result['duration']
            })
    
    return {
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'total_duration': total_duration
        },
        'failures': failures,
        'details': results
    }


def print_report(report: Dict[str, Any], output_file: str = None):
    """Print the validation report."""
    summary = report['summary']
    
    print(f"\n{'='*80}")
    print(f"C++ CODE VALIDATION REPORT")
    print(f"{'='*80}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total Tests: {summary['total_tests']}")
    print(f"   Passed: {summary['passed_tests']}")
    print(f"   Failed: {summary['failed_tests']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Total Duration: {summary['total_duration']:.2f}s")
    
    if report['failures']:
        print(f"\n❌ FAILURES:")
        for failure in report['failures']:
            print(f"   • {failure['test_file']} ({failure['duration']:.2f}s)")
            if failure['error']:
                print(f"     Error: {failure['error'][:100]}...")
    
    print(f"\n📋 DETAILED RESULTS:")
    for test_file, result in report['details'].items():
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"   {status} {test_file} ({result['duration']:.2f}s)")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n💾 Report saved to: {output_file}")


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    print("🔍 Checking prerequisites...")
    
    # Check for Python
    try:
        subprocess.run(['python', '--version'], capture_output=True, check=True)
        print("✅ Python found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Python not found")
        return False
    
    # Check for pytest
    try:
        subprocess.run(['python', '-m', 'pytest', '--version'], capture_output=True, check=True)
        print("✅ pytest found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pytest not found")
        return False
    
    # Check for C++ compiler
    try:
        subprocess.run(['g++', '--version'], capture_output=True, check=True)
        print("✅ g++ compiler found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  g++ compiler not found - CUDA tests may fail")
    
    # Check for CUDA compiler
    try:
        subprocess.run(['nvcc', '--version'], capture_output=True, check=True)
        print("✅ nvcc compiler found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  nvcc compiler not found - CUDA tests may fail")
    
    # Check for robodsl
    try:
        import robodsl
        print("✅ robodsl package found")
    except ImportError:
        print("❌ robodsl package not found")
        return False
    
    return True


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run C++ code validation tests')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--output', '-o', help='Output file for JSON report')
    parser.add_argument('--tests', '-t', nargs='+', help='Specific test files to run')
    parser.add_argument('--skip-prerequisites', action='store_true', help='Skip prerequisite checks')
    
    args = parser.parse_args()
    
    # Check prerequisites
    if not args.skip_prerequisites:
        if not check_prerequisites():
            print("\n❌ Prerequisites not met. Please install required dependencies.")
            sys.exit(1)
    
    # Define test files
    if args.tests:
        test_files = args.tests
    else:
        test_files = [
            'tests/test_cpp_code_validation.py',
            'tests/test_cpp_efficiency_validation.py',
            'tests/test_cpp_correctness_validation.py',
            'tests/test_cuda_code_validation.py',
            'tests/test_comprehensive_cpp_validation.py'
        ]
    
    # Filter out non-existent files
    existing_files = []
    for test_file in test_files:
        if Path(test_file).exists():
            existing_files.append(test_file)
        else:
            print(f"⚠️  Test file not found: {test_file}")
    
    if not existing_files:
        print("❌ No test files found to run")
        sys.exit(1)
    
    print(f"\n🚀 Running {len(existing_files)} C++ validation test files...")
    
    # Run tests
    results = run_pytest_tests(existing_files, args.verbose)
    
    # Generate report
    report = generate_summary_report(results)
    
    # Print report
    print_report(report, args.output)
    
    # Exit with appropriate code
    if report['summary']['failed_tests'] > 0:
        print(f"\n❌ Validation failed with {report['summary']['failed_tests']} test failures")
        sys.exit(1)
    else:
        print(f"\n✅ All C++ validation tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main() 