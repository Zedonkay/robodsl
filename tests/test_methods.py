"""Tests for C++ method support in RoboDSL."""

import textwrap
from pathlib import Path
import pytest
from robodsl.parser import parse_robodsl
from robodsl.generator import CodeGenerator


def test_parse_node_with_methods():
    """Test parsing a node with C++ methods."""
    dsl = textwrap.dedent('''
    node test_node {
        methods = [{
            name = "processData"
            return_type = "std::vector<float>"
            parameters = ["const std::vector<float>& input", "float scale"]
            implementation = """
                std::vector<float> result;
                result.reserve(input.size());
                for (const auto& val : input) {
                    result.push_back(val * scale);
                }
                return result;
            """
        }]
    }
    ''').strip()
    config = parse_robodsl(dsl)

    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert node.name == "test_node"
    assert len(node.methods) == 1

    method = node.methods[0]
    assert method.name == "processData"
    assert method.return_type == "std::vector<float>"
    assert method.parameters == ["const std::vector<float>& input", "float scale"]
    assert "result.reserve" in method.implementation
    assert "val * scale" in method.implementation


def test_parse_node_with_multiple_methods():
    """Test parsing a node with multiple C++ methods."""
    dsl = textwrap.dedent('''
    node test_node {
        methods = [{
            name = "validateInput"
            return_type = "bool"
            parameters = ["const std::string& input"]
            implementation = """
                return !input.empty() && input.length() < 100;
            """
        }, {
            name = "computeAverage"
            return_type = "double"
            parameters = ["const std::vector<double>& values"]
            implementation = """
                if (values.empty()) return 0.0;
                double sum = 0.0;
                for (const auto& v : values) {
                    sum += v;
                }
                return sum / values.size();
            """
        }]
    }
    ''').strip()
    config = parse_robodsl(dsl)

    assert len(config.nodes) == 1
    node = config.nodes[0]
    assert len(node.methods) == 2

    # Check first method
    method1 = node.methods[0]
    assert method1.name == "validateInput"
    assert method1.return_type == "bool"
    assert method1.parameters == ["const std::string& input"]
    assert "input.empty()" in method1.implementation

    # Check second method
    method2 = node.methods[1]
    assert method2.name == "computeAverage"
    assert method2.return_type == "double"
    assert method2.parameters == ["const std::vector<double>& values"]
    assert "values.empty()" in method2.implementation


def test_method_with_complex_parameters():
    """Test parsing a method with complex parameter types."""
    dsl = textwrap.dedent('''
    node test_node {
        methods = [{
            name = "processPointCloud"
            return_type = "void"
            parameters = [
                "const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in",
                "pcl::PointCloud<pcl::PointXYZ>& cloud_out",
                "const Eigen::Matrix4f& transform"
            ]
            implementation = """
                // Implementation would go here
                (void)cloud_in;
                (void)cloud_out;
                (void)transform;
            """
        }]
    }
    ''').strip()
    config = parse_robodsl(dsl)

    node = config.nodes[0]
    method = node.methods[0]
    assert method.name == "processPointCloud"
    assert method.return_type == "void"
    assert len(method.parameters) == 3
    assert method.parameters[0] == "const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud_in"
    assert method.parameters[1] == "pcl::PointCloud<pcl::PointXYZ>& cloud_out"
    assert method.parameters[2] == "const Eigen::Matrix4f& transform"


def test_generate_node_with_virtual_method(tmp_path: Path):
    """Test generating a node with a virtual C++ method."""
    dsl = textwrap.dedent('''
    project_name = "TestProject"
    node test_node {
        methods = [{
            name = "setup"
            return_type = "bool"
            parameters = []
            implementation = """
                // Custom setup logic
                return true;
            """
        }]
    }
    ''').strip()
    config = parse_robodsl(dsl)

    # Generate code
    generator = CodeGenerator(config, output_dir=str(tmp_path))
    generator.generate()

    # Check header file
    header_path = tmp_path / 'include' / 'TestProject' / 'test_node.hpp'
    assert header_path.exists()
    header_content = header_path.read_text()

    # Check for virtual method declaration
    expected_declaration = "virtual bool setup();"
    assert expected_declaration in header_content

    # Check source file
    source_path = tmp_path / 'src' / 'test_node.cpp'
    assert source_path.exists()
    source_content = source_path.read_text()

    # Check for method implementation
    assert "bool TestNode::setup()" in source_content
    assert "// Custom setup logic" in source_content
    assert "return true;" in source_content
