#!/usr/bin/env python3
"""
Advanced Features Validation Test Runner.

This script runs comprehensive validation tests for:
- Pipeline generation
- ONNX model integration  
- TensorRT optimization
- CUDA integration
- Performance optimization

Usage:
    python run_advanced_features_validation.py [--category CATEGORY] [--verbose] [--report]
"""

import os
import sys
import argparse
import subprocess
import tempfile
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.generators import MainGenerator
from test_advanced_features_config import (
    ADVANCED_FEATURES_CONFIG,
    ADVANCED_FEATURES_CATEGORIES,
    generate_pipeline_test_cases,
    generate_onnx_test_cases,
    generate_tensorrt_test_cases,
    generate_comprehensive_test_cases,
    generate_edge_case_test_cases
)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    category: str
    passed: bool
    issues: Dict[str, List[str]]
    file_count: int
    validation_time: float
    error_message: str = ""


class AdvancedFeaturesValidator:
    """Validates advanced features for correctness and efficiency."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.compiler_flags = config["compiler_flags"]
        self.cuda_flags = config["cuda_flags"]
        self.timeout = config["timeout_seconds"]
    
    def validate_syntax(self, cpp_code: str, filename: str = "test.cpp") -> Tuple[bool, List[str]]:
        """Validate C++ syntax using g++ compiler."""
        issues = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.compiler_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode != 0:
                issues.append(f"Compilation failed: {result.stderr}")
                return False, issues
            
            return True, issues
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            issues.append(f"Compiler error: {str(e)}")
            return False, issues
        finally:
            os.unlink(temp_file)
    
    def validate_cuda_syntax(self, cuda_code: str, filename: str = "test.cu") -> Tuple[bool, List[str]]:
        """Validate CUDA syntax using nvcc compiler."""
        issues = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
            f.write(cuda_code)
            temp_file = f.name
        
        try:
            cmd = ['nvcc'] + self.cuda_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
            
            if result.returncode != 0:
                issues.append(f"CUDA compilation failed: {result.stderr}")
                return False, issues
            
            return True, issues
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            issues.append(f"CUDA compiler error: {str(e)}")
            return False, issues
        finally:
            os.unlink(temp_file)
    
    def check_pipeline_features(self, cpp_code: str) -> List[str]:
        """Check for pipeline-specific features."""
        issues = []
        
        if 'namespace' not in cpp_code:
            issues.append("Pipeline code should use proper namespaces")
        
        if 'rclcpp' not in cpp_code:
            issues.append("Pipeline should integrate with ROS2")
        
        if 'initialize_' not in cpp_code:
            issues.append("Pipeline stages should have initialization methods")
        
        if 'input' not in cpp_code or 'output' not in cpp_code:
            issues.append("Pipeline should handle input/output data flow")
        
        return issues
    
    def check_onnx_features(self, cpp_code: str) -> List[str]:
        """Check for ONNX-specific features."""
        issues = []
        
        if 'onnxruntime_cxx_api.h' not in cpp_code:
            issues.append("ONNX Runtime C++ API should be included")
        
        if 'Ort::Session' not in cpp_code:
            issues.append("ONNX Runtime session should be used")
        
        if 'Ort::Value' not in cpp_code:
            issues.append("ONNX Runtime tensor values should be used")
        
        if 'Ort::Exception' not in cpp_code:
            issues.append("ONNX Runtime exceptions should be handled")
        
        return issues
    
    def check_tensorrt_features(self, cpp_code: str) -> List[str]:
        """Check for TensorRT-specific features."""
        issues = []
        
        if 'onnxruntime_providers.h' not in cpp_code:
            issues.append("TensorRT provider should be included")
        
        if 'OrtTensorRTProviderOptions' not in cpp_code:
            issues.append("TensorRT provider options should be configured")
        
        if 'trt_fp16_enable' not in cpp_code and 'trt_int8_enable' not in cpp_code:
            issues.append("TensorRT optimization settings should be configured")
        
        if 'trt_engine_cache_enable' not in cpp_code:
            issues.append("TensorRT engine cache should be enabled")
        
        return issues
    
    def check_cuda_features(self, cpp_code: str) -> List[str]:
        """Check for CUDA-specific features."""
        issues = []
        
        if 'cuda_runtime.h' not in cpp_code:
            issues.append("CUDA runtime should be included")
        
        if 'cudaMalloc' in cpp_code and 'cudaFree' not in cpp_code:
            issues.append("CUDA memory should be properly freed")
        
        if 'cudaMemcpy' not in cpp_code and 'cuda' in cpp_code:
            issues.append("CUDA memory transfers should be handled")
        
        if 'cudaError_t' in cpp_code and 'cudaSuccess' not in cpp_code:
            issues.append("CUDA error codes should be checked")
        
        return issues
    
    def check_performance_features(self, cpp_code: str) -> List[str]:
        """Check for performance optimization features."""
        issues = []
        
        if 'std::vector' in cpp_code and 'reserve' not in cpp_code:
            issues.append("Consider using vector::reserve() for better performance")
        
        if 'const' not in cpp_code and 'std::string' in cpp_code:
            issues.append("Consider using const references to avoid copies")
        
        if 'std::move' not in cpp_code and 'std::vector' in cpp_code:
            issues.append("Consider using move semantics for better performance")
        
        if 'new ' in cpp_code and 'std::unique_ptr' not in cpp_code:
            issues.append("Consider using smart pointers for memory management")
        
        return issues
    
    def check_best_practices(self, cpp_code: str) -> List[str]:
        """Check for C++ best practices."""
        issues = []
        
        if 'using namespace' in cpp_code:
            issues.append("Avoid using namespace in headers")
        
        if '#include <iostream>' in cpp_code and 'std::cout' not in cpp_code:
            issues.append("Remove unused includes")
        
        if 'friend class' in cpp_code:
            issues.append("Consider if friend classes are necessary")
        
        return issues
    
    def run_validation(self, source_code: str, test_name: str, category: str) -> ValidationResult:
        """Run comprehensive validation on advanced features."""
        start_time = time.time()
        
        try:
            # Parse the source code
            ast = parse_robodsl(source_code)
            
            # Create test output directory
            test_output_dir = Path("test_output") / test_name
            test_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save RoboDSL source file
            source_file = test_output_dir / f"{test_name}.robodsl"
            with open(source_file, 'w') as f:
                f.write(source_code)
            
            # Generate code with output directory
            generator = MainGenerator()
            generated_files = generator.generate(ast, output_dir=str(test_output_dir))
            
            # Initialize issue categories
            issues = {
                "syntax": [],
                "pipeline": [],
                "onnx": [],
                "tensorrt": [],
                "cuda": [],
                "performance": [],
                "practices": []
            }
            
            # Validate each generated file
            for file_path in generated_files:
                content = file_path.read_text()
                
                # Check file size
                if len(content) > self.config["max_file_size_kb"] * 1024:
                    issues["syntax"].append(f"File {file_path.name} is too large")
                
                # Validate based on file type
                if file_path.suffix in ['.cpp', '.hpp']:
                    # C++ file validation
                    syntax_ok, syntax_issues = self.validate_syntax(content, str(file_path))
                    issues["syntax"].extend(syntax_issues)
                    
                    if syntax_ok:
                        # Check feature-specific issues
                        if 'pipeline' in str(file_path).lower() or 'stage' in content.lower():
                            issues["pipeline"].extend(self.check_pipeline_features(content))
                        if 'onnx' in str(file_path).lower() or 'Ort::' in content:
                            issues["onnx"].extend(self.check_onnx_features(content))
                        if 'tensorrt' in content.lower() or 'trt_' in content:
                            issues["tensorrt"].extend(self.check_tensorrt_features(content))
                        if 'cuda' in str(file_path).lower() or 'cuda' in content.lower():
                            issues["cuda"].extend(self.check_cuda_features(content))
                        
                        # Always check performance and practices
                        issues["performance"].extend(self.check_performance_features(content))
                        issues["practices"].extend(self.check_best_practices(content))
                    
                elif file_path.suffix in ['.cu', '.cuh']:
                    # CUDA file validation
                    syntax_ok, syntax_issues = self.validate_cuda_syntax(content, str(file_path))
                    issues["syntax"].extend(syntax_issues)
                    
                    if syntax_ok:
                        issues["cuda"].extend(self.check_cuda_features(content))
                        issues["performance"].extend(self.check_performance_features(content))
                        issues["practices"].extend(self.check_best_practices(content))
            
            validation_time = time.time() - start_time
            
            # Determine if test passed based on tolerance levels
            passed = (
                len(issues["syntax"]) <= self.config["syntax_tolerance"] and
                len(issues["pipeline"]) <= self.config["pipeline_tolerance"] and
                len(issues["onnx"]) <= self.config["onnx_tolerance"] and
                len(issues["tensorrt"]) <= self.config["tensorrt_tolerance"] and
                len(issues["cuda"]) <= self.config["cuda_tolerance"] and
                len(issues["performance"]) <= self.config["performance_tolerance"] and
                len(issues["practices"]) <= self.config["practice_tolerance"]
            )
            
            return ValidationResult(
                test_name=test_name,
                category=category,
                passed=passed,
                issues=issues,
                file_count=len(generated_files),
                validation_time=validation_time
            )
            
        except Exception as e:
            validation_time = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                category=category,
                passed=False,
                issues={},
                file_count=0,
                validation_time=validation_time,
                error_message=str(e)
            )


def run_category_tests(category: str, validator: AdvancedFeaturesValidator, verbose: bool = False) -> List[ValidationResult]:
    """Run tests for a specific category."""
    results = []
    
    if category == "pipeline":
        test_cases = list(generate_pipeline_test_cases())
    elif category == "onnx":
        test_cases = list(generate_onnx_test_cases())
    elif category == "tensorrt":
        test_cases = list(generate_tensorrt_test_cases())
    elif category == "comprehensive":
        test_cases = list(generate_comprehensive_test_cases())
    elif category == "edge_cases":
        test_cases = list(generate_edge_case_test_cases())
    else:
        print(f"Unknown category: {category}")
        return results
    
    print(f"\nRunning {category} tests...")
    
    for i, (source_code, expected_issues, test_id) in enumerate(test_cases, 1):
        if verbose:
            print(f"  Test {i}/{len(test_cases)}: {test_id}")
        
        result = validator.run_validation(source_code, test_id, category)
        results.append(result)
        
        if verbose:
            status = "PASS" if result.passed else "FAIL"
            print(f"    {status} - {result.file_count} files, {result.validation_time:.2f}s")
            if not result.passed and result.error_message:
                print(f"    Error: {result.error_message}")
    
    return results


def generate_report(results: List[ValidationResult], output_file: str = None) -> Dict[str, Any]:
    """Generate a comprehensive validation report."""
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r.passed)
    failed_tests = total_tests - passed_tests
    
    # Categorize by category
    category_stats = {}
    for result in results:
        if result.category not in category_stats:
            category_stats[result.category] = {'total': 0, 'passed': 0, 'failed': 0}
        category_stats[result.category]['total'] += 1
        if result.passed:
            category_stats[result.category]['passed'] += 1
        else:
            category_stats[result.category]['failed'] += 1
    
    # Aggregate issues
    total_issues = {
        "syntax": 0,
        "pipeline": 0,
        "onnx": 0,
        "tensorrt": 0,
        "cuda": 0,
        "performance": 0,
        "practices": 0
    }
    
    for result in results:
        for issue_type, issues in result.issues.items():
            total_issues[issue_type] += len(issues)
    
    report = {
        'summary': {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
        },
        'category_stats': category_stats,
        'total_issues': total_issues,
        'detailed_results': [
            {
                'test_name': r.test_name,
                'category': r.category,
                'passed': r.passed,
                'file_count': r.file_count,
                'validation_time': r.validation_time,
                'issues': r.issues,
                'error_message': r.error_message
            }
            for r in results
        ]
    }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_file}")
    
    return report


def print_summary(report: Dict[str, Any]):
    """Print a summary of the validation results."""
    summary = report['summary']
    category_stats = report['category_stats']
    total_issues = report['total_issues']
    
    print("\n" + "="*60)
    print("ADVANCED FEATURES VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nOverall Results:")
    print(f"  Total Tests: {summary['total_tests']}")
    print(f"  Passed: {summary['passed_tests']}")
    print(f"  Failed: {summary['failed_tests']}")
    print(f"  Success Rate: {summary['success_rate']:.1f}%")
    
    print(f"\nCategory Breakdown:")
    for category, stats in category_stats.items():
        success_rate = (stats['passed'] / stats['total'] * 100) if stats['total'] > 0 else 0
        print(f"  {category.capitalize()}: {stats['passed']}/{stats['total']} ({success_rate:.1f}%)")
    
    print(f"\nIssue Summary:")
    for issue_type, count in total_issues.items():
        if count > 0:
            print(f"  {issue_type.capitalize()}: {count}")
    
    print("\n" + "="*60)


def main():
    """Main function for the advanced features validation runner."""
    parser = argparse.ArgumentParser(description="Advanced Features Validation Test Runner")
    parser.add_argument("--category", choices=list(ADVANCED_FEATURES_CATEGORIES.keys()) + ["all"], 
                       default="all", help="Test category to run")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", "-r", help="Save detailed report to file")
    parser.add_argument("--config", help="Custom configuration file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ADVANCED_FEATURES_CONFIG.copy()
    if args.config:
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
    
    # Initialize validator
    validator = AdvancedFeaturesValidator(config)
    
    # Run tests
    all_results = []
    
    if args.category == "all":
        categories = list(ADVANCED_FEATURES_CATEGORIES.keys())
    else:
        categories = [args.category]
    
    for category in categories:
        if category in ADVANCED_FEATURES_CATEGORIES:
            results = run_category_tests(category, validator, args.verbose)
            all_results.extend(results)
        else:
            print(f"Unknown category: {category}")
    
    # Generate and print report
    report = generate_report(all_results, args.report)
    print_summary(report)
    
    # Exit with appropriate code
    if report['summary']['failed_tests'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main() 