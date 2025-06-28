#!/usr/bin/env python3
"""
Test script to integrate the existing RoboDSL parser with the VS Code extension.
This script tests the parser functionality and generates diagnostics for the extension.
"""

import sys
import os
from pathlib import Path

# Add the main project to the path so we can import the parser
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from robodsl.parsers.lark_parser import RoboDSLParser, parse_robodsl
    from robodsl.parsers.semantic_analyzer import SemanticError
    from lark import ParseError
    print("✅ Successfully imported RoboDSL parser")
except ImportError as e:
    print(f"❌ Failed to import RoboDSL parser: {e}")
    sys.exit(1)

def test_parser_with_files():
    """Test the parser with various test files."""
    test_files = [
        "test-debug.robodsl",
        "quick-test.robodsl", 
        "in-depth-test.robodsl",
        "comprehensive_test.robodsl"
    ]
    
    parser = RoboDSLParser()
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"⚠️  Test file {test_file} not found, skipping...")
            continue
            
        print(f"\n🔍 Testing {test_file}:")
        print("=" * 50)
        
        try:
            with open(test_file, 'r') as f:
                content = f.read()
            
            # Test parsing with issues
            ast, issues = parser.parse_with_issues(content)
            
            if ast:
                print(f"✅ Parsed successfully")
                print(f"   - Nodes: {len(ast.nodes)}")
                print(f"   - CUDA kernels: {len(ast.cuda_kernels.kernels) if ast.cuda_kernels else 0}")
                print(f"   - ONNX models: {len(ast.onnx_models)}")
                print(f"   - Pipelines: {len(ast.pipelines)}")
                print(f"   - Includes: {len(ast.includes)}")
            else:
                print(f"❌ Failed to parse")
            
            # Report issues
            if issues:
                print(f"   - Issues found: {len(issues)}")
                for i, issue in enumerate(issues[:5]):  # Show first 5 issues
                    level = issue['level'].upper()
                    message = issue['message']
                    rule_id = issue['rule_id']
                    print(f"     {i+1}. [{level}] {message} (rule: {rule_id})")
                if len(issues) > 5:
                    print(f"     ... and {len(issues) - 5} more issues")
            else:
                print(f"   - No issues found")
                
        except Exception as e:
            print(f"❌ Error testing {test_file}: {e}")

def test_syntax_validation():
    """Test syntax validation for the extension."""
    print(f"\n🔍 Testing syntax validation:")
    print("=" * 50)
    
    # Test cases with expected errors
    test_cases = [
        {
            "name": "Valid syntax",
            "code": """
            node test_node {
                parameter int test_param = 42
                publisher /test_topic: "std_msgs/String"
            }
            """,
            "should_parse": True
        },
        {
            "name": "Missing colon",
            "code": """
            node test_node {
                parameter test_param = 42
            }
            """,
            "should_parse": False
        },
        {
            "name": "Missing quotes",
            "code": """
            node test_node {
                publisher /test_topic: std_msgs/String
            }
            """,
            "should_parse": False
        },
        {
            "name": "CUDA kernel",
            "code": """
            cuda_kernels {
                kernel test_kernel {
                    block_size: (256, 1, 1)
                    code: {
                        int idx = blockIdx.x * blockDim.x + threadIdx.x;
                    }
                }
            }
            """,
            "should_parse": True
        },
        {
            "name": "ONNX model",
            "code": """
            onnx_model test_model {
                input: "input" -> "float32[1,3,224,224]"
                output: "output" -> "float32[1,1000]"
                device: cuda
            }
            """,
            "should_parse": True
        }
    ]
    
    parser = RoboDSLParser()
    
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        try:
            ast, issues = parser.parse_with_issues(test_case['code'])
            if ast and not issues:
                print(f"  ✅ Parsed successfully")
            elif ast and issues:
                print(f"  ⚠️  Parsed with {len(issues)} issues")
                for issue in issues[:3]:
                    print(f"    - [{issue['level']}] {issue['message']}")
            else:
                print(f"  ❌ Failed to parse")
                for issue in issues[:3]:
                    print(f"    - [{issue['level']}] {issue['message']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def generate_extension_diagnostics():
    """Generate diagnostics that the extension can use."""
    print(f"\n🔍 Generating extension diagnostics:")
    print("=" * 50)
    
    # Test the in-depth test file
    test_file = "in-depth-test.robodsl"
    if not os.path.exists(test_file):
        print(f"❌ Test file {test_file} not found")
        return
    
    try:
        with open(test_file, 'r') as f:
            content = f.read()
        
        parser = RoboDSLParser()
        ast, issues = parser.parse_with_issues(content)
        
        print(f"File: {test_file}")
        print(f"Content length: {len(content)} characters")
        print(f"Lines: {len(content.split(chr(10)))}")
        
        if ast:
            print(f"✅ AST generated successfully")
            print(f"   - Nodes: {len(ast.nodes)}")
            if ast.nodes:
                node = ast.nodes[0]
                print(f"   - First node: {node.name}")
                print(f"   - Parameters: {len(node.content.parameters)}")
                print(f"   - Publishers: {len(node.content.publishers)}")
                print(f"   - Subscribers: {len(node.content.subscribers)}")
                print(f"   - Methods: {len(node.content.methods)}")
        else:
            print(f"❌ Failed to generate AST")
        
        print(f"\nIssues found: {len(issues)}")
        for i, issue in enumerate(issues):
            level = issue['level'].upper()
            message = issue['message']
            rule_id = issue['rule_id']
            print(f"  {i+1}. [{level}] {message}")
            print(f"     Rule: {rule_id}")
        
    except Exception as e:
        print(f"❌ Error generating diagnostics: {e}")

def test_completion_suggestions():
    """Test completion suggestions based on the grammar."""
    print(f"\n🔍 Testing completion suggestions:")
    print("=" * 50)
    
    # Keywords from the grammar
    keywords = [
        "node", "cuda_kernels", "kernel", "method", "parameter", "remap", 
        "namespace", "flag", "lifecycle", "timer", "client", "publisher", 
        "subscriber", "service", "action", "include", "input", "output", 
        "code", "in", "out", "inout", "block_size", "grid_size", 
        "shared_memory", "use_thrust", "qos", "onnx_model", "device", 
        "optimization", "pipeline", "stage", "config", "true", "false"
    ]
    
    # ROS types
    ros_types = [
        "std_msgs/String", "std_msgs/Int32", "std_msgs/Float64", "std_msgs/Bool",
        "geometry_msgs/Twist", "geometry_msgs/Pose", "geometry_msgs/Point",
        "sensor_msgs/Image", "sensor_msgs/LaserScan", "sensor_msgs/PointCloud2",
        "nav_msgs/Odometry", "nav_msgs/Path", "nav_msgs/OccupancyGrid"
    ]
    
    # C++ types
    cpp_types = [
        "int", "float", "double", "bool", "char", "string", "std::string",
        "std::vector", "std::array", "std::map", "std::unordered_map",
        "cv::Mat", "cv::Point", "cv::Point2f", "cv::Point3f",
        "Eigen::Vector3d", "Eigen::Matrix3d", "Eigen::Quaterniond"
    ]
    
    print(f"Keywords ({len(keywords)}): {', '.join(keywords[:10])}...")
    print(f"ROS Types ({len(ros_types)}): {', '.join(ros_types[:5])}...")
    print(f"C++ Types ({len(cpp_types)}): {', '.join(cpp_types[:5])}...")
    
    # Test context-aware suggestions
    contexts = [
        ("After 'parameter'", ["int", "float", "double", "bool", "string"]),
        ("After 'publisher'", ros_types),
        ("After 'method'", ["input", "output", "code"]),
        ("After 'cuda_kernels'", ["kernel"]),
        ("After 'onnx_model'", ["input", "output", "device", "optimization"])
    ]
    
    for context, suggestions in contexts:
        print(f"\n{context}:")
        print(f"  Suggestions: {', '.join(suggestions[:5])}...")

if __name__ == "__main__":
    print("🚀 RoboDSL Parser Integration Test")
    print("=" * 60)
    
    test_parser_with_files()
    test_syntax_validation()
    generate_extension_diagnostics()
    test_completion_suggestions()
    
    print(f"\n✅ Integration test completed!") 