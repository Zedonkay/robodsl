"""Comprehensive C++ Code Validation Test Runner.

This test runner executes all C++ validation tests and provides detailed reporting
on the quality of generated C++ code, including:
- Syntax correctness
- Efficiency analysis
- Best practices compliance
- Performance optimization opportunities
- Resource management validation
"""

import os
import sys
import pytest
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


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    issues: List[str]
    file_path: str
    content: str
    validation_time: float


class ComprehensiveCppValidator:
    """Comprehensive C++ code validator that runs all validation tests."""
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.compiler_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O2',
            '-fno-exceptions', '-fno-rtti', '-DNDEBUG'
        ]
        self.cuda_flags = [
            '-std=c++17', '-Wall', '-Wextra', '-O2',
            '-arch=sm_60', '-DNDEBUG'
        ]
    
    def validate_syntax(self, cpp_code: str, filename: str = "test.cpp") -> Tuple[bool, List[str]]:
        """Validate C++ syntax using g++ compiler."""
        issues = []
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(cpp_code)
            temp_file = f.name
        
        try:
            cmd = ['g++'] + self.compiler_flags + ['-c', temp_file, '-o', '/dev/null']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
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
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                issues.append(f"CUDA compilation failed: {result.stderr}")
                return False, issues
            
            return True, issues
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            issues.append(f"CUDA compiler error: {str(e)}")
            return False, issues
        finally:
            os.unlink(temp_file)
    
    def check_memory_management(self, cpp_code: str) -> List[str]:
        """Check for memory management issues."""
        issues = []
        
        # Check for new/delete balance
        new_count = cpp_code.count('new ')
        delete_count = cpp_code.count('delete ')
        
        if new_count > delete_count:
            issues.append(f"Potential memory leak: {new_count} new vs {delete_count} delete")
        
        # Check for malloc/free balance
        malloc_count = cpp_code.count('malloc(')
        free_count = cpp_code.count('free(')
        
        if malloc_count > free_count:
            issues.append(f"Potential memory leak: {malloc_count} malloc vs {free_count} free")
        
        # Check for smart pointer usage
        if 'new ' in cpp_code and 'std::unique_ptr' not in cpp_code and 'std::shared_ptr' not in cpp_code:
            issues.append("Consider using smart pointers instead of raw pointers")
        
        return issues
    
    def check_modern_cpp_practices(self, cpp_code: str) -> List[str]:
        """Check for modern C++ best practices."""
        issues = []
        
        # Check for const correctness
        if 'const' not in cpp_code and 'std::string' in cpp_code:
            issues.append("Consider using const references to avoid copies")
        
        # Check for range-based for loops
        if 'for' in cpp_code and 'auto&' not in cpp_code and 'auto' in cpp_code:
            issues.append("Consider using range-based for loops with auto&")
        
        # Check for nullptr usage
        if 'NULL' in cpp_code:
            issues.append("Use nullptr instead of NULL")
        
        # Check for C++ style casts
        if '(' in cpp_code and ')' in cpp_code and 'static_cast' not in cpp_code:
            issues.append("Consider using C++ style casts")
        
        return issues
    
    def check_performance_issues(self, cpp_code: str) -> List[str]:
        """Check for performance issues."""
        issues = []
        
        # Check for unnecessary copies
        if 'std::string' in cpp_code and '+' in cpp_code:
            issues.append("Consider using std::stringstream for string concatenation")
        
        # Check for inefficient containers
        if 'std::list' in cpp_code and 'std::vector' not in cpp_code:
            issues.append("Consider std::vector for better cache locality")
        
        # Check for virtual functions in tight loops
        if 'virtual' in cpp_code and 'for' in cpp_code:
            issues.append("Virtual function calls in loops may impact performance")
        
        return issues
    
    def check_cuda_best_practices(self, cuda_code: str) -> List[str]:
        """Check for CUDA best practices."""
        issues = []
        
        # Check for error handling
        if 'cudaError_t' in cuda_code and 'cudaGetLastError' not in cuda_code:
            issues.append("CUDA error handling should include cudaGetLastError()")
        
        # Check for memory management
        if 'cudaMalloc' in cuda_code and 'cudaFree' not in cuda_code:
            issues.append("CUDA memory should be properly freed")
        
        # Check for synchronization
        if 'cudaMemcpyAsync' in cuda_code and 'cudaStreamSynchronize' not in cuda_code:
            issues.append("Asynchronous operations should be synchronized")
        
        return issues
    
    def run_comprehensive_validation(self, source_code: str, test_name: str) -> List[ValidationResult]:
        """Run comprehensive validation on generated code."""
        results = []
        
        try:
            # Parse the source code
            ast = parse_robodsl(source_code)
            
            # Generate code
            generator = MainGenerator()
            generated_files = generator.generate(ast)
            
            # Validate each generated file
            for file_path in generated_files:
                start_time = time.time()
                
                content = file_path.read_text()
                issues = []
                
                # Determine file type and run appropriate validations
                if file_path.suffix in ['.cpp', '.hpp']:
                    # C++ file validation
                    syntax_ok, syntax_issues = self.validate_syntax(content, str(file_path))
                    issues.extend(syntax_issues)
                    
                    if syntax_ok:
                        issues.extend(self.check_memory_management(content))
                        issues.extend(self.check_modern_cpp_practices(content))
                        issues.extend(self.check_performance_issues(content))
                    
                    passed = syntax_ok and len(issues) <= 5  # Allow some issues
                    
                elif file_path.suffix in ['.cu', '.cuh']:
                    # CUDA file validation
                    syntax_ok, syntax_issues = self.validate_cuda_syntax(content, str(file_path))
                    issues.extend(syntax_issues)
                    
                    if syntax_ok:
                        issues.extend(self.check_cuda_best_practices(content))
                    
                    passed = syntax_ok and len(issues) <= 3  # Allow some issues
                    
                else:
                    # Skip other file types
                    continue
                
                validation_time = time.time() - start_time
                
                result = ValidationResult(
                    test_name=f"{test_name}_{file_path.name}",
                    passed=passed,
                    issues=issues,
                    file_path=str(file_path),
                    content=content,
                    validation_time=validation_time
                )
                
                results.append(result)
                
        except Exception as e:
            # Handle parsing or generation errors
            result = ValidationResult(
                test_name=test_name,
                passed=False,
                issues=[f"Generation error: {str(e)}"],
                file_path="",
                content="",
                validation_time=0.0
            )
            results.append(result)
        
        return results
    
    def generate_validation_report(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive validation report."""
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = total_tests - passed_tests
        
        # Categorize issues
        syntax_issues = []
        memory_issues = []
        performance_issues = []
        practice_issues = []
        cuda_issues = []
        
        for result in results:
            for issue in result.issues:
                if 'compilation' in issue.lower() or 'syntax' in issue.lower():
                    syntax_issues.append(issue)
                elif 'memory' in issue.lower() or 'leak' in issue.lower():
                    memory_issues.append(issue)
                elif 'performance' in issue.lower() or 'efficiency' in issue.lower():
                    performance_issues.append(issue)
                elif 'cuda' in issue.lower():
                    cuda_issues.append(issue)
                else:
                    practice_issues.append(issue)
        
        report = {
            'summary': {
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': failed_tests,
                'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            'issues': {
                'syntax_issues': len(syntax_issues),
                'memory_issues': len(memory_issues),
                'performance_issues': len(performance_issues),
                'practice_issues': len(practice_issues),
                'cuda_issues': len(cuda_issues)
            },
            'detailed_results': [
                {
                    'test_name': r.test_name,
                    'passed': r.passed,
                    'file_path': r.file_path,
                    'issues': r.issues,
                    'validation_time': r.validation_time
                }
                for r in results
            ]
        }
        
        return report


class TestComprehensiveCppValidation:
    """Comprehensive C++ validation test suite."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.validator = ComprehensiveCppValidator()
    
    def test_basic_node_validation(self, test_output_dir):
        """Test basic node generation validation."""
        source = """
        node basic_node {
            parameter int max_speed = 10
            publisher /test: std_msgs/String
            subscriber /input: std_msgs/String
        }
        """
        
        results = self.validator.run_comprehensive_validation(source, "basic_node")
        report = self.validator.generate_validation_report(results)
        
        # Basic validation should pass
        assert report['summary']['success_rate'] >= 80, \
            f"Basic node validation failed: {report['summary']['success_rate']}% success rate"
        
        # Should not have critical issues
        assert report['issues']['syntax_issues'] == 0, \
            f"Basic node has syntax issues: {report['issues']['syntax_issues']}"
    
    def test_cuda_kernel_validation(self, test_output_dir):
        """Test CUDA kernel generation validation."""
        source = """
        cuda_kernel test_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        """
        
        results = self.validator.run_comprehensive_validation(source, "cuda_kernel")
        report = self.validator.generate_validation_report(results)
        
        # CUDA validation should pass
        assert report['summary']['success_rate'] >= 70, \
            f"CUDA kernel validation failed: {report['summary']['success_rate']}% success rate"
        
        # Should have CUDA files
        cuda_results = [r for r in results if '.cu' in r.file_path or '.cuh' in r.file_path]
        assert len(cuda_results) > 0, "No CUDA files generated"
    
    def test_advanced_cpp_features_validation(self, test_output_dir):
        """Test advanced C++ features validation."""
        source = """
        template<typename T> struct Vector {
            T* data;
            size_t size;
        }
        
        global PI: constexpr float = 3.14159;
        
        def operator<<(stream: std::ostream&, vec: Vector<int>&) -> std::ostream& {
            stream << "Vector";
            return stream;
        }
        
        node advanced_node {
            parameter int max_size = 1000
            
            def advanced_function(input: const std::vector<float>&) -> std::vector<float> {
                return input;
            }
        }
        """
        
        results = self.validator.run_comprehensive_validation(source, "advanced_cpp")
        report = self.validator.generate_validation_report(results)
        
        # Advanced features should pass
        assert report['summary']['success_rate'] >= 75, \
            f"Advanced C++ features validation failed: {report['summary']['success_rate']}% success rate"
    
    def test_comprehensive_validation(self, test_output_dir):
        """Test comprehensive validation with all features."""
        source = """
        include <iostream>
        include <vector>
        include <memory>
        
        template<typename T> struct SafeContainer {
            std::unique_ptr<T[]> data;
            size_t size;
            
            SafeContainer(size_t n) : size(n) {
                data = std::make_unique<T[]>(n);
            }
        }
        
        global MAX_SIZE: constexpr int = 1000;
        
        cuda_kernel efficient_kernel {
            block_size: (256, 1, 1)
            grid_size: (1, 1, 1)
            input: float data[1000]
            output: float result[1000]
        }
        
        node comprehensive_node {
            parameter int max_speed = 10
            parameter float safety_distance = 1.5
            
            publisher /robot/position: geometry_msgs/Point
            subscriber /robot/command: geometry_msgs/Twist
            service /robot/status: std_srvs/Trigger
            
            def efficient_process(input: const std::vector<float>&) -> std::vector<float> {
                std::vector<float> result;
                result.reserve(input.size());
                
                for (const auto& value : input) {
                    if (value > 0.0f) {
                        result.push_back(value);
                    }
                }
                return result;
            }
            
            def __init__() {
                // Constructor with proper initialization
            }
            
            def __del__() {
                // Destructor with proper cleanup
            }
        }
        """
        
        results = self.validator.run_comprehensive_validation(source, "comprehensive")
        report = self.validator.generate_validation_report(results)
        
        # Comprehensive validation should pass
        assert report['summary']['success_rate'] >= 70, \
            f"Comprehensive validation failed: {report['summary']['success_rate']}% success rate"
        
        # Should have both C++ and CUDA files
        cpp_results = [r for r in results if r.file_path.endswith(('.cpp', '.hpp'))]
        cuda_results = [r for r in results if r.file_path.endswith(('.cu', '.cuh'))]
        
        assert len(cpp_results) > 0, "No C++ files generated"
        assert len(cuda_results) > 0, "No CUDA files generated"
        
        # Print detailed report for debugging
        print(f"\nComprehensive Validation Report:")
        print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Passed: {report['summary']['passed_tests']}")
        print(f"Failed: {report['summary']['failed_tests']}")
        print(f"Issues: {report['issues']}")
    
    def test_edge_cases_validation(self, test_output_dir):
        """Test edge cases validation."""
        source = """
        node edge_case_node {
            parameter int zero_param = 0
            parameter float negative_param = -1.0
            parameter std::string empty_string = ""
            
            def edge_case_function() -> int {
                return 0;
            }
        }
        """
        
        results = self.validator.run_comprehensive_validation(source, "edge_cases")
        report = self.validator.generate_validation_report(results)
        
        # Edge cases should still pass
        assert report['summary']['success_rate'] >= 80, \
            f"Edge cases validation failed: {report['summary']['success_rate']}% success rate"
    
    def test_large_scale_validation(self, test_output_dir):
        """Test large-scale code generation validation."""
        source = """
        // Generate multiple nodes to test scalability
        """
        
        # Add multiple nodes
        for i in range(3):
            source += f"""
            node large_scale_node_{i} {{
                parameter int id = {i}
                publisher /node_{i}/data: std_msgs/String
                subscriber /node_{i}/command: std_msgs/String
                
                def process_{i}(input: const std::vector<float>&) -> std::vector<float> {{
                    return input;
                }}
            }}
            """
        
        results = self.validator.run_comprehensive_validation(source, "large_scale")
        report = self.validator.generate_validation_report(results)
        
        # Large scale should pass
        assert report['summary']['success_rate'] >= 75, \
            f"Large scale validation failed: {report['summary']['success_rate']}% success rate"
        
        # Should generate multiple files
        assert report['summary']['total_tests'] >= 3, \
            f"Expected at least 3 tests, got {report['summary']['total_tests']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 