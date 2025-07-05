from robodsl.parsers.lark_parser import parse_robodsl
"""Tests for RoboDSL validation and linting."""

import pytest
from pathlib import Path
from robodsl.core.validator import (
    RoboDSLValidator, RoboDSLLinter, ValidationIssue, ValidationLevel,
    validate_robodsl_file, format_robodsl_file
)


class TestRoboDSLValidator:
    """Test the RoboDSL validator."""
    
    def test_valid_content(self):
        """Test validation of valid RoboDSL content."""
        content = """
        node test_node {
            parameter int test_param = 42
            publisher /test_topic: "std_msgs/msg/String"
            subscriber /input_topic: "std_msgs/msg/String"
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have some style warnings but no semantic errors
        assert all('invalid message type' not in (issue.message or '') for issue in issues)
        assert all(issue.level != ValidationLevel.ERROR or 'semantic error' not in (issue.message or '') for issue in issues)
    
    def test_parse_error(self):
        """Test validation with parse errors."""
        content = """
        node test_node {
            parameter test_param: 42
            invalid_syntax
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have parse error
        assert any(issue.level == ValidationLevel.ERROR for issue in issues)
        assert any("parse error" in issue.message.lower() for issue in issues)
    
    def test_semantic_error(self):
        """Test validation with semantic errors."""
        content = """
        node test_node {
            parameter int test_param = 42
            parameter int test_param = 43  // Duplicate parameter
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have semantic error
        assert any(issue.level == ValidationLevel.ERROR for issue in issues)
        assert any("duplicate parameter name" in issue.message.lower() for issue in issues)
    
    def test_style_validation(self):
        """Test style validation."""
        content = """
        node test_node {
            parameter int test_param = 42
            publisher /test_topic: "std_msgs/String"
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have trailing whitespace warning
        assert any(issue.rule_id == "trailing_whitespace" for issue in issues)
    
    def test_naming_conventions(self):
        """Test naming convention validation."""
        content = """
        node TestNode {
            parameter int TestParam = 42
            publisher /TestTopic: "std_msgs/String"
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have naming convention warnings
        assert any(issue.rule_id == "node_naming_convention" for issue in issues)
        assert any(issue.rule_id == "parameter_naming_convention" for issue in issues)
        assert any(issue.rule_id == "topic_naming_convention" for issue in issues)
    
    def test_best_practices(self):
        """Test best practices validation."""
        content = """
        node isolated_node {
            parameter int test_param = 42
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have isolated node warning
        assert any(issue.rule_id == "isolated_node" for issue in issues)
    
    def test_performance_validation(self):
        """Test performance validation."""
        content = """
        node test_node {
            publisher /test_topic: "std_msgs/String" {
                qos {
                    depth: 200
                }
            }
        }
        """
    
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Debug: print all issues
        print("DEBUG: All issues:")
        for issue in issues:
            print(f"  - {issue.rule_id}: {issue.message}")
    
        # Should have large QoS depth warning
        assert any(issue.rule_id == "large_qos_depth" for issue in issues)
    
    def test_cuda_kernel_validation(self):
        """Test CUDA kernel validation."""
        content = """
        cuda_kernels {
            kernel test_kernel {
                shared_memory: 32768
            }
        }
        """
        
        validator = RoboDSLValidator()
        issues = validator.validate_string(content)
        
        # Should have large shared memory warning
        assert any(issue.rule_id == "large_shared_memory" for issue in issues)


class TestRoboDSLLinter:
    """Test the RoboDSL linter."""
    
    def test_format_string(self):
        """Test string formatting."""
        content = """
        node test_node{
        parameter test_param:42
        publisher /test_topic:"std_msgs/String"
        }
        """
        
        linter = RoboDSLLinter()
        formatted = linter.format_string(content)
        
        # Should be properly formatted
        assert "node test_node {" in formatted
        assert "    parameter test_param: 42" in formatted
        assert 'publisher /test_topic : "std_msgs/String"' in formatted
        assert formatted.endswith('\n')
    
    def test_remove_trailing_whitespace(self):
        """Test removal of trailing whitespace."""
        content = "node test_node {\n    parameter test_param: 42    \n}"
        
        linter = RoboDSLLinter()
        formatted = linter.format_string(content)
        
        # Should not have trailing whitespace
        assert "    parameter test_param: 42" in formatted
        assert "    parameter test_param: 42    " not in formatted
    
    def test_indentation_fixing(self):
        """Test indentation fixing."""
        content = """
        node test_node {
      parameter test_param: 42
            publisher /test_topic : "std_msgs/String"
        }
        """
        
        linter = RoboDSLLinter()
        formatted = linter.format_string(content)
        
        # Should have consistent indentation
        lines = formatted.split('\n')
        for line in lines:
            if line.strip() and not line.strip().startswith('//'):
                if line.strip().startswith('parameter') or line.strip().startswith('publisher'):
                    assert line.startswith('    ')
    
    def test_check_formatting(self):
        """Test formatting check."""
        content = """
        node test_node {
            parameter int test_param = 42
        }
        """
        
        linter = RoboDSLLinter()
        issues = linter.check_formatting(content)
        
        # Should detect formatting issues
        assert any(issue.rule_id == "trailing_whitespace" for issue in issues)


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_validate_robodsl_file(self, tmp_path):
        """Test file validation convenience function."""
        file_path = tmp_path / "test.robodsl"
        file_path.write_text("""
        node test_node {
            parameter int test_param = 42
        }
        """)
        
        issues = validate_robodsl_file(file_path)
        
        # Should validate without errors
        assert all(issue.level != ValidationLevel.ERROR for issue in issues)
    
    def test_format_robodsl_file(self, tmp_path):
        """Test file formatting convenience function."""
        file_path = tmp_path / "test.robodsl"
        file_path.write_text("""
        node test_node{
        parameter test_param:42
        }
        """)
        
        formatted = format_robodsl_file(file_path)
        
        # Should be properly formatted
        assert "node test_node {" in formatted
        assert "    parameter test_param: 42" in formatted