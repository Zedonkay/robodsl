"""Tests for Advanced C++ Features (Phase 8)."""

import pytest
from pathlib import Path
from robodsl.parsers.lark_parser import parse_robodsl
from robodsl.parsers.ast_builder import ASTBuilder
from robodsl.generators.advanced_cpp_generator import AdvancedCppGenerator
from robodsl.core.ast import (
    TemplateStructNode, TemplateClassNode, TemplateFunctionNode, TemplateAliasNode,
    StaticAssertNode, GlobalConstexprNode, GlobalDeviceConstNode, GlobalStaticInlineNode,
    OperatorOverloadNode, ConstructorNode, DestructorNode, BitfieldNode,
    PreprocessorDirectiveNode, FunctionAttributeNode, ConceptNode,
    FriendDeclarationNode, UserDefinedLiteralNode, ValueNode, MethodParamNode,
    TemplateParamNode, BitfieldMemberNode, ConceptRequiresNode
)


class TestAdvancedCppFeatures:
    """Test suite for advanced C++ features."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ast_builder = ASTBuilder(debug=True)
        self.generator = AdvancedCppGenerator()

    def test_template_struct_parsing(self):
        """Test parsing of template struct definitions."""
        source = """
        template<typename T> struct Foo {
            T data;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, TemplateStructNode)
        assert feature.name == "Foo"
        assert len(feature.template_params) == 1
        assert feature.template_params[0].name == "T"
        assert feature.template_params[0].param_type == "typename"

    def test_template_class_parsing(self):
        """Test parsing of template class definitions."""
        source = """
        template<typename T> class Vector {
            T* data;
            size_t size;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, TemplateClassNode)
        assert feature.name == "Vector"
        assert len(feature.template_params) == 1
        assert feature.template_params[0].name == "T"

    def test_template_function_parsing(self):
        """Test parsing of template function definitions."""
        source = """
        template<typename T> T sqr(x: T) {
            return x * x;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, TemplateFunctionNode)
        assert feature.name == "sqr"
        assert len(feature.parameters) == 1
        assert feature.parameters[0].param_name == "x"

    def test_template_alias_parsing(self):
        """Test parsing of template alias definitions."""
        source = """
        template<typename T> using Vec = std::vector<T>;
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, TemplateAliasNode)
        assert feature.name == "Vec"
        assert feature.aliased_type == "std::vector<T>"

    def test_static_assert_parsing(self):
        """Test parsing of static assertions."""
        source = """
        static_assert(sizeof(int) == 4, "int must be 4 bytes");
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, StaticAssertNode)
        assert feature.condition == "sizeof(int) == 4"
        assert feature.message == '"int must be 4 bytes"'

    def test_global_constexpr_parsing(self):
        """Test parsing of global constexpr variables."""
        source = """
        global PI: constexpr float = 3.14159;
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, GlobalConstexprNode)
        assert feature.name == "PI"
        assert feature.type == "float"
        assert feature.value.value == 3.14159

    def test_global_device_const_parsing(self):
        """Test parsing of global device constants."""
        source = """
        global device LUT: __constant__ int[256] = [1, 2, 3, 4];
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, GlobalDeviceConstNode)
        assert feature.name == "LUT"
        assert feature.type == "int"
        assert feature.array_size == "256"
        assert len(feature.values) == 4

    def test_operator_overload_parsing(self):
        """Test parsing of operator overloads."""
        source = """
        def operator<<(stream: std::ostream&, obj: Foo&) -> std::ostream& {
            stream << obj.data;
            return stream;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, OperatorOverloadNode)
        assert feature.operator == "operator<<"
        assert len(feature.parameters) == 2
        assert feature.return_type == "std::ostream&"

    def test_constructor_parsing(self):
        """Test parsing of constructors."""
        source = """
        def __init__(x: float, y: float, z: float) : x(x), y(y), z(z) {
            // Constructor body
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, ConstructorNode)
        assert len(feature.parameters) == 3
        assert len(feature.member_initializers) == 3

    def test_destructor_parsing(self):
        """Test parsing of destructors."""
        source = """
        def __del__() {
            // Destructor body
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, DestructorNode)

    def test_bitfield_parsing(self):
        """Test parsing of bitfields."""
        source = """
        struct Flags {
            uint32_t enabled : 1;
            uint32_t mode : 3;
            uint32_t priority : 4;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, BitfieldNode)
        assert feature.name == "Flags"
        assert len(feature.members) == 3
        assert feature.members[0].bits == 1
        assert feature.members[1].bits == 3
        assert feature.members[2].bits == 4

    def test_preprocessor_directive_parsing(self):
        """Test parsing of preprocessor directives."""
        source = """
        #pragma once
        #include <iostream>
        #if defined(JETSON)
        #define USE_TENSORRT
        #endif
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) >= 3
        directives = [f for f in ast.advanced_cpp_features if isinstance(f, PreprocessorDirectiveNode)]
        assert len(directives) >= 3

    def test_function_attribute_parsing(self):
        """Test parsing of functions with attributes."""
        source = """
        @device @forceinline
        fast_math(x: float) -> float {
            return x * x;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, FunctionAttributeNode)
        assert "device" in feature.attributes
        assert "forceinline" in feature.attributes
        
        # Test that it generates valid C++
        code = self.generator.generate_function_attributes([feature])
        assert "__device__ __forceinline__ float fast_math(float x)" in code
        assert "return x * x;" in code

    def test_concept_parsing(self):
        """Test parsing of concepts."""
        source = """
        concept Arithmetic: requires T: T operator+(T, T) T operator*(T, T)
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, ConceptNode)
        assert feature.name == "Arithmetic"

    def test_concept_token_inspection(self):
        """Test to inspect tokens produced for concept parsing."""
        from robodsl.parsers.lark_parser import RoboDSLParser
        
        # Test with different spacing and formatting
        test_cases = [
            "concept Arithmetic: requires T: T operator+(T, T) T operator*(T, T)",
            "concept Arithmetic : requires T : T operator + (T, T) T operator * (T, T)",
            "concept Arithmetic: requires T: T operator + (T, T) T operator * (T, T)"
        ]
        
        for i, source in enumerate(test_cases):
            print(f"\nTest case {i+1}: {repr(source)}")
            
            # Create parser instance to access lexer
            parser = RoboDSLParser()
            
            # Get the lexer and tokenize the source
            tokens = list(parser.parser.lex(source))
            
            print(f"Tokens for test case {i+1}:")
            for token in tokens:
                print(f"  {token.type}: '{token.value}'")
            
            # Also try to parse and see where it fails
            try:
                ast = parse_robodsl(source)
                print(f"Test case {i+1}: Parsing succeeded!")
            except Exception as e:
                print(f"Test case {i+1}: Parsing failed with error: {e}")
                # Print the full traceback for debugging
                import traceback
                traceback.print_exc()

    def test_friend_declaration_parsing(self):
        """Test parsing of friend declarations."""
        source = """
        friend class Foo;
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, FriendDeclarationNode)
        assert feature.friend_type == "class"
        assert feature.target == "Foo"

    def test_user_defined_literal_parsing(self):
        """Test parsing of user-defined literals."""
        source = """
        def operator""_mps(value: long double) -> float {
            return value * 0.44704;
        }
        """
        
        ast = parse_robodsl(source)
        
        assert len(ast.advanced_cpp_features) == 1
        feature = ast.advanced_cpp_features[0]
        assert isinstance(feature, UserDefinedLiteralNode)
        assert feature.literal_suffix == "_mps"
        assert feature.return_type == "float"

    def test_template_generation(self):
        """Test generation of template code."""
        template_struct = TemplateStructNode(
            name="Foo",
            template_params=[TemplateParamNode(name="T", param_type="typename")],
            content=None  # Simplified for test
        )
        
        code = self.generator.generate_templates([template_struct])
        assert "template<typename T>" in code
        assert "struct Foo" in code

    def test_static_assert_generation(self):
        """Test generation of static assertions."""
        static_assert = StaticAssertNode(
            condition="sizeof(int) == 4",
            message='"int must be 4 bytes"'
        )
        
        code = self.generator.generate_static_asserts([static_assert])
        assert "static_assert(sizeof(int) == 4" in code

    def test_global_constexpr_generation(self):
        """Test generation of global constexpr variables."""
        constexpr = GlobalConstexprNode(
            name="PI",
            type="float",
            value=ValueNode(value=3.14159)
        )
        
        code = self.generator.generate_global_variables([constexpr])
        assert "constexpr float PI = 3.14159" in code

    def test_operator_overload_generation(self):
        """Test generation of operator overloads."""
        operator = OperatorOverloadNode(
            operator="operator+",
            parameters=[MethodParamNode(param_type="int", param_name="a")],
            return_type="int",
            code="return a + 1;"
        )
        
        code = self.generator.generate_operator_overloads([operator])
        assert "int operator+(int a)" in code
        assert "return a + 1;" in code

    def test_bitfield_generation(self):
        """Test generation of bitfields."""
        bitfield = BitfieldNode(
            name="Flags",
            members=[
                BitfieldMemberNode(type="uint32_t", name="enabled", bits=1),
                BitfieldMemberNode(type="uint32_t", name="mode", bits=3)
            ]
        )
        
        code = self.generator.generate_bitfields([bitfield])
        assert "struct Flags" in code
        assert "enabled : 1" in code

    def test_concept_generation(self):
        """Test generation of concepts."""
        concept = ConceptNode(
            name="Arithmetic",
            requires=ConceptRequiresNode(
                type_param="T",
                requirements=["T operator+(T, T)", "T operator*(T, T)"]
            )
        )
        
        code = self.generator.generate_concepts([concept])
        assert "concept Arithmetic" in code

    def test_user_defined_literal_generation(self):
        """Test generation of user-defined literals."""
        literal = UserDefinedLiteralNode(
            literal_suffix="_mps",
            return_type="float",
            code="return value * 0.44704;"
        )
        
        code = self.generator.generate_user_defined_literals([literal])
        assert 'operator""_mps' in code

    def test_comprehensive_example(self):
        """Test a comprehensive example with multiple advanced C++ features."""
        source = """
        // Template struct
        template<typename T> struct Vector {
            T* data;
            size_t size;
        }
        
        // Static assertion
        static_assert(sizeof(int) == 4, "int must be 4 bytes");
        
        // Global constexpr
        global PI: constexpr float = 3.14159;
        
        // Operator overload
        def operator+(a: Vector<int>&, b: Vector<int>&) -> Vector<int> {
            // Implementation
        }
        
        // Bitfield
        struct Flags {
            uint32_t enabled : 1;
            uint32_t mode : 3;
        }
        
        // Function with attributes
        @device @forceinline
        fast_math(x: float) -> float {
            return x * x;
        }
        
        // Concept
        concept Arithmetic: requires T: T operator+(T, T) T operator*(T, T)
        
        // User-defined literal
        def operator""_deg(value: long double) -> float {
            return value * M_PI / 180.0;
        }
        """
        
        ast = parse_robodsl(source)
        
        # Should have multiple advanced C++ features
        assert len(ast.advanced_cpp_features) >= 7
        
        # Check that different types are present
        feature_types = [type(f).__name__ for f in ast.advanced_cpp_features]
        assert "TemplateStructNode" in feature_types
        assert "StaticAssertNode" in feature_types
        assert "GlobalConstexprNode" in feature_types
        
        # Test generation produces valid C++
        generated_code = self.generator._generate_header_file(ast)
        
        # Check for valid C++ syntax in generated code
        assert "template<typename T>" in generated_code
        assert "struct Vector" in generated_code
        assert "static_assert(sizeof(int) == 4" in generated_code
        assert "constexpr float PI = 3.14159" in generated_code
        assert "__device__ __forceinline__ float fast_math(float x)" in generated_code
        assert "float operator\"\"_deg(long double value)" in generated_code

    def test_error_handling(self):
        """Test error handling for malformed advanced C++ features."""
        # Test with malformed template
        source = """
        template<typename T> struct {
            T data;
        }
        """
        
        with pytest.raises(Exception):
            ast = parse_robodsl(source)




if __name__ == "__main__":
    pytest.main([__file__]) 