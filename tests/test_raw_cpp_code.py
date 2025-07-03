"""Test raw C++ code feature."""

import pytest
from pathlib import Path
from robodsl.parsers.lark_parser import RoboDSLParser
from robodsl.generators.main_generator import MainGenerator


class TestRawCppCode:
    """Test raw C++ code blocks inside and outside nodes."""
    
    def test_raw_cpp_code_global(self):
        """Test raw C++ code blocks outside nodes."""
        robodsl_code = """
include <iostream>
include <vector>

// Global raw C++ code
cpp: {
    namespace global_utils {
        template<typename T>
        class VectorProcessor {
        public:
            static std::vector<T> process(const std::vector<T>& input) {
                std::vector<T> result;
                result.reserve(input.size());
                for (const auto& item : input) {
                    result.push_back(item * 2);
                }
                return result;
            }
        };
    }
}

node test_node {
    parameter int value = 42
    publisher /test_topic: "std_msgs/msg/Int32"
}
"""
        
        # Parse the code
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        # Verify that raw C++ code was parsed
        assert hasattr(ast, 'raw_cpp_code')
        assert len(ast.raw_cpp_code) == 1
        assert ast.raw_cpp_code[0].location == "global"
        assert "namespace global_utils" in ast.raw_cpp_code[0].code
        assert "class VectorProcessor" in ast.raw_cpp_code[0].code
        
        # Generate files
        generator = MainGenerator(output_dir="test_output")
        generated_files = generator.generate(ast)
        
        # Check that global C++ code file was generated
        global_cpp_files = [f for f in generated_files if "global_cpp_code" in str(f)]
        assert len(global_cpp_files) == 1
        
        # Verify the content
        global_cpp_content = global_cpp_files[0].read_text()
        assert "namespace global_utils" in global_cpp_content
        assert "class VectorProcessor" in global_cpp_content
    
    def test_raw_cpp_code_inside_node(self):
        """Test raw C++ code blocks inside nodes."""
        robodsl_code = """
node test_node {
    parameter int value = 42
    publisher /test_topic: "std_msgs/msg/Int32"
    
    // Raw C++ code inside node
    cpp: {
        class NodeHelper {
        public:
            static int processValue(int input) {
                return input * 3 + 1;
            }
        };
    }
}
"""
        
        # Parse the code
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        # Verify that raw C++ code was parsed inside the node
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert hasattr(node.content, 'raw_cpp_code')
        assert len(node.content.raw_cpp_code) == 1
        assert node.content.raw_cpp_code[0].location == "node"
        assert "class NodeHelper" in node.content.raw_cpp_code[0].code
        
        # Generate files
        generator = MainGenerator(output_dir="test_output")
        generated_files = generator.generate(ast)
        
        # Check that node files were generated
        node_files = [f for f in generated_files if "test_node" in str(f)]
        assert len(node_files) >= 2  # Header and source
        
        # Verify the content in the generated files
        header_file = next(f for f in node_files if f.suffix == '.hpp')
        source_file = next(f for f in node_files if f.suffix == '.cpp')
        
        header_content = header_file.read_text()
        source_content = source_file.read_text()
        
        # Raw C++ code should appear in both header and source
        assert "class NodeHelper" in header_content
        assert "class NodeHelper" in source_content
    
    def test_multiple_raw_cpp_blocks(self):
        """Test multiple raw C++ code blocks."""
        robodsl_code = """
// First global block
cpp: {
    namespace utils {
        int global_function() { return 42; }
    }
}

// Second global block
cpp: {
    namespace helpers {
        double helper_function() { return 3.14; }
    }
}

node test_node {
    parameter int value = 42
    
    // First node block
    cpp: {
        class NodeClass1 {
        public:
            void method1() {}
        };
    }
    
    // Second node block
    cpp: {
        class NodeClass2 {
        public:
            void method2() {}
        };
    }
}
"""
        
        # Parse the code
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        # Verify global blocks
        assert len(ast.raw_cpp_code) == 2
        assert any("global_function" in block.code for block in ast.raw_cpp_code)
        assert any("helper_function" in block.code for block in ast.raw_cpp_code)
        
        # Verify node blocks
        assert len(ast.nodes) == 1
        node = ast.nodes[0]
        assert len(node.content.raw_cpp_code) == 2
        assert any("NodeClass1" in block.code for block in node.content.raw_cpp_code)
        assert any("NodeClass2" in block.code for block in node.content.raw_cpp_code)
    
    def test_raw_cpp_code_with_complex_syntax(self):
        """Test raw C++ code with complex C++ syntax."""
        robodsl_code = """
cpp: {
    #include <memory>
    #include <functional>
    
    template<typename T, typename U>
    class ComplexTemplate {
    private:
        std::unique_ptr<T> data_;
        std::function<U(T)> processor_;
        
    public:
        ComplexTemplate(std::unique_ptr<T> data, std::function<U(T)> processor)
            : data_(std::move(data)), processor_(std::move(processor)) {}
        
        U process() const {
            if (data_ && processor_) {
                return processor_(*data_);
            }
            return U{};
        }
        
        template<typename V>
        auto transform(std::function<V(U)> transformer) const {
            return ComplexTemplate<U, V>(
                std::make_unique<U>(process()),
                std::move(transformer)
            );
        }
    };
}
"""
        
        # Parse the code
        parser = RoboDSLParser()
        ast = parser.parse(robodsl_code)
        
        # Verify parsing
        assert len(ast.raw_cpp_code) == 1
        code = ast.raw_cpp_code[0].code
        assert "template<typename T, typename U>" in code
        assert "class ComplexTemplate" in code
        assert "std::unique_ptr<T> data_" in code
        assert "std::function<U(T)> processor_" in code
        
        # Generate files
        generator = MainGenerator(output_dir="test_output")
        generated_files = generator.generate(ast)
        
        # Verify generation
        global_cpp_files = [f for f in generated_files if "global_cpp_code" in str(f)]
        assert len(global_cpp_files) == 1
        
        content = global_cpp_files[0].read_text()
        assert "template<typename T, typename U>" in content
        assert "class ComplexTemplate" in content 