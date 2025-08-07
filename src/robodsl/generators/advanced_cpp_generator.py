"""Advanced C++ Features Generator.

This module generates C++ code for advanced features like templates, operator overloads,
concepts, and other modern C++ features commonly used in robotics and CUDA development.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import jinja2

from ..core.ast import (
    TemplateStructNode, TemplateClassNode, TemplateFunctionNode, TemplateAliasNode,
    StaticAssertNode, GlobalConstexprNode, GlobalDeviceConstNode, GlobalStaticInlineNode,
    OperatorOverloadNode, ConstructorNode, DestructorNode, BitfieldNode,
    PreprocessorDirectiveNode, FunctionAttributeNode, ConceptNode,
    FriendDeclarationNode, UserDefinedLiteralNode, RoboDSLAST
)
from .base_generator import BaseGenerator


class AdvancedCppGenerator(BaseGenerator):
    """Generator for advanced C++ features."""

    def __init__(self, output_dir: str = ".", template_dirs: Optional[List[Path]] = None):
        """Initialize the advanced C++ generator."""
        super().__init__(output_dir, template_dirs)

    def generate(self, ast: 'RoboDSLAST') -> List[Path]:
        """Generate files from the AST.
        
        Args:
            ast: The parsed RoboDSL AST
            
        Returns:
            List of Path objects for generated files
        """
        generated_files = []
        
        # Generate header file with advanced C++ features
        if ast.advanced_cpp_features:
            header_content = self._generate_header_file(ast)
            header_path = self.get_output_path("advanced_cpp_features.hpp")
            self.write_file(header_path, header_content)
            generated_files.append(header_path)
        
        return generated_files

    def _generate_header_file(self, ast: 'RoboDSLAST') -> str:
        """Generate header file content for advanced C++ features."""
        code_parts = []
        
        # Add includes
        code_parts.append("#pragma once")
        code_parts.append("#include <iostream>")
        code_parts.append("#include <vector>")
        code_parts.append("#include <type_traits>")
        code_parts.append("")
        
        # Generate each feature type
        templates = [f for f in ast.advanced_cpp_features if isinstance(f, (TemplateStructNode, TemplateClassNode, TemplateFunctionNode, TemplateAliasNode))]
        if templates:
            code_parts.append("// Template definitions")
            code_parts.append(self.generate_templates(templates))
            code_parts.append("")
        
        static_asserts = [f for f in ast.advanced_cpp_features if isinstance(f, StaticAssertNode)]
        if static_asserts:
            code_parts.append("// Static assertions")
            code_parts.append(self.generate_static_asserts(static_asserts))
            code_parts.append("")
        
        global_vars = [f for f in ast.advanced_cpp_features if isinstance(f, (GlobalConstexprNode, GlobalDeviceConstNode, GlobalStaticInlineNode))]
        if global_vars:
            code_parts.append("// Global variables and constants")
            code_parts.append(self.generate_global_variables(global_vars))
            code_parts.append("")
        
        operators = [f for f in ast.advanced_cpp_features if isinstance(f, OperatorOverloadNode)]
        if operators:
            code_parts.append("// Operator overloads")
            code_parts.append(self.generate_operator_overloads(operators))
            code_parts.append("")
        
        constructors = [f for f in ast.advanced_cpp_features if isinstance(f, (ConstructorNode, DestructorNode))]
        if constructors:
            code_parts.append("// Constructors and destructors")
            code_parts.append(self.generate_constructors(constructors))
            code_parts.append("")
        
        bitfields = [f for f in ast.advanced_cpp_features if isinstance(f, BitfieldNode)]
        if bitfields:
            code_parts.append("// Bitfields")
            code_parts.append(self.generate_bitfields(bitfields))
            code_parts.append("")
        
        preprocessor = [f for f in ast.advanced_cpp_features if isinstance(f, PreprocessorDirectiveNode)]
        if preprocessor:
            code_parts.append("// Preprocessor directives")
            code_parts.append(self.generate_preprocessor_directives(preprocessor))
            code_parts.append("")
        
        attributes = [f for f in ast.advanced_cpp_features if isinstance(f, FunctionAttributeNode)]
        if attributes:
            code_parts.append("// Functions with attributes")
            code_parts.append(self.generate_function_attributes(attributes))
            code_parts.append("")
        
        concepts = [f for f in ast.advanced_cpp_features if isinstance(f, ConceptNode)]
        if concepts:
            code_parts.append("// Concepts")
            code_parts.append(self.generate_concepts(concepts))
            code_parts.append("")
        
        friends = [f for f in ast.advanced_cpp_features if isinstance(f, FriendDeclarationNode)]
        if friends:
            code_parts.append("// Friend declarations")
            code_parts.append(self.generate_friend_declarations(friends))
            code_parts.append("")
        
        literals = [f for f in ast.advanced_cpp_features if isinstance(f, UserDefinedLiteralNode)]
        if literals:
            code_parts.append("// User-defined literals")
            code_parts.append(self.generate_user_defined_literals(literals))
            code_parts.append("")
        
        return "\n".join(code_parts)

    def generate_templates(self, template_nodes: List[Any]) -> str:
        """Generate C++ code for template definitions."""
        code_parts = []
        
        for node in template_nodes:
            if isinstance(node, TemplateStructNode):
                code_parts.append(self._generate_template_struct(node))
            elif isinstance(node, TemplateClassNode):
                code_parts.append(self._generate_template_class(node))
            elif isinstance(node, TemplateFunctionNode):
                code_parts.append(self._generate_template_function(node))
            elif isinstance(node, TemplateAliasNode):
                code_parts.append(self._generate_template_alias(node))
        
        return "\n\n".join(code_parts)

    def generate_static_asserts(self, assert_nodes: List[StaticAssertNode]) -> str:
        """Generate C++ code for static assertions."""
        code_parts = []
        
        for node in assert_nodes:
            code_parts.append(f"static_assert({node.condition}, {node.message});")
        
        return "\n".join(code_parts)

    def generate_global_variables(self, global_nodes: List[Any]) -> str:
        """Generate C++ code for global variables and constants."""
        code_parts = []
        
        for node in global_nodes:
            if isinstance(node, GlobalConstexprNode):
                code_parts.append(self._generate_global_constexpr(node))
            elif isinstance(node, GlobalDeviceConstNode):
                code_parts.append(self._generate_global_device_const(node))
            elif isinstance(node, GlobalStaticInlineNode):
                code_parts.append(self._generate_global_static_inline(node))
        
        return "\n\n".join(code_parts)

    def generate_operator_overloads(self, operator_nodes: List[OperatorOverloadNode]) -> str:
        """Generate C++ code for operator overloads."""
        code_parts = []
        
        for node in operator_nodes:
            code_parts.append(self._generate_operator_overload(node))
        
        return "\n\n".join(code_parts)

    def generate_constructors(self, constructor_nodes: List[Any]) -> str:
        """Generate C++ code for constructors and destructors."""
        code_parts = []
        
        for node in constructor_nodes:
            if isinstance(node, ConstructorNode):
                code_parts.append(self._generate_constructor(node))
            elif isinstance(node, DestructorNode):
                code_parts.append(self._generate_destructor(node))
        
        return "\n\n".join(code_parts)

    def generate_bitfields(self, bitfield_nodes: List[BitfieldNode]) -> str:
        """Generate C++ code for bitfields."""
        code_parts = []
        
        for node in bitfield_nodes:
            code_parts.append(self._generate_bitfield(node))
        
        return "\n\n".join(code_parts)

    def generate_preprocessor_directives(self, directive_nodes: List[PreprocessorDirectiveNode]) -> str:
        """Generate C++ code for preprocessor directives."""
        code_parts = []
        
        for node in directive_nodes:
            code_parts.append(self._generate_preprocessor_directive(node))
        
        return "\n".join(code_parts)

    def generate_function_attributes(self, attribute_nodes: List[FunctionAttributeNode]) -> str:
        """Generate C++ code for functions with attributes."""
        code_parts = []
        
        for node in attribute_nodes:
            code_parts.append(self._generate_function_attribute(node))
        
        return "\n\n".join(code_parts)

    def generate_concepts(self, concept_nodes: List[ConceptNode]) -> str:
        """Generate C++ code for concepts."""
        code_parts = []
        
        for node in concept_nodes:
            code_parts.append(self._generate_concept(node))
        
        return "\n\n".join(code_parts)

    def generate_friend_declarations(self, friend_nodes: List[FriendDeclarationNode]) -> str:
        """Generate C++ code for friend declarations."""
        code_parts = []
        
        for node in friend_nodes:
            code_parts.append(self._generate_friend_declaration(node))
        
        return "\n".join(code_parts)

    def generate_user_defined_literals(self, literal_nodes: List[UserDefinedLiteralNode]) -> str:
        """Generate C++ code for user-defined literals."""
        code_parts = []
        
        for node in literal_nodes:
            code_parts.append(self._generate_user_defined_literal(node))
        
        return "\n\n".join(code_parts)

    def _generate_template_struct(self, node: TemplateStructNode) -> str:
        """Generate C++ code for a template struct."""
        template_params = ", ".join([f"{param.param_type} {param.name}" for param in node.template_params])
        
        # Generate struct members
        members = []
        if node.content and hasattr(node.content, 'members'):
            for member in node.content.members:
                member_str = f"    {member.type} {member.name}"
                if member.array_spec:
                    member_str += member.array_spec
                member_str += ";"
                members.append(member_str)
        
        # Generate methods
        methods = []
        if node.content and hasattr(node.content, 'methods'):
            for method in node.content.methods:
                methods.append(self._generate_method(method))
        
        return f"""template<{template_params}>
struct {node.name} {{
{chr(10).join(members)}
{chr(10).join(methods)}
}};"""

    def _generate_template_class(self, node: TemplateClassNode) -> str:
        """Generate C++ code for a template class."""
        template_params = ", ".join([f"{param.param_type} {param.name}" for param in node.template_params])
        
        # Generate inheritance
        inheritance_str = ""
        if node.inheritance:
            bases = []
            for access, base in node.inheritance.base_classes:
                if access:
                    bases.append(f"{access} {base}")
                else:
                    bases.append(base)
            inheritance_str = f" : {', '.join(bases)}"
        
        # Generate class content
        content_parts = []
        
        # Direct members
        if node.content and hasattr(node.content, 'members'):
            for member in node.content.members:
                content_parts.append(f"    {member.type} {member.name};")
        
        # Access sections
        if node.content and hasattr(node.content, 'access_sections'):
            for section in node.content.access_sections:
                content_parts.append(f"    {section.access_specifier}:")
                for member in section.members:
                    content_parts.append(f"        {member.type} {member.name};")
                for method in section.methods:
                    content_parts.append(f"        {self._generate_method(method)}")
        
        # Direct methods
        if node.content and hasattr(node.content, 'methods'):
            for method in node.content.methods:
                content_parts.append(f"    {self._generate_method(method)}")
        
        return f"""template<{template_params}>
class {node.name}{inheritance_str} {{
{chr(10).join(content_parts)}
}};"""

    def _generate_template_function(self, node: TemplateFunctionNode) -> str:
        """Generate C++ code for a template function."""
        template_params = ", ".join([f"{param.param_type} {param.name}" for param in node.template_params])
        func_params = ", ".join([f"{param.param_type} {param.param_name}" for param in node.parameters])
        
        # Proper C++ syntax: return_type function_name(parameters)
        return_type = node.return_type if node.return_type else "void"
        
        return f"""template<{template_params}>
{return_type} {node.name}({func_params}) {{
{node.code}
}}"""

    def _generate_template_alias(self, node: TemplateAliasNode) -> str:
        """Generate C++ code for a template alias."""
        template_params = ", ".join([f"{param.param_type} {param.name}" for param in node.template_params])
        
        return f"""template<{template_params}>
using {node.name} = {node.aliased_type};"""

    def _generate_global_constexpr(self, node: GlobalConstexprNode) -> str:
        """Generate C++ code for a global constexpr variable."""
        value_str = self._value_to_string(node.value)
        return f"constexpr {node.type} {node.name} = {value_str};"

    def _generate_global_device_const(self, node: GlobalDeviceConstNode) -> str:
        """Generate C++ code for a global device constant."""
        values_str = ", ".join([self._value_to_string(v) for v in node.values])
        return f"__constant__ {node.type} {node.name}[{node.array_size}] = {{{values_str}}};"

    def _generate_global_static_inline(self, node: GlobalStaticInlineNode) -> str:
        """Generate C++ code for a global static inline function."""
        func_params = ", ".join([f"{param.param_type} {param.param_name}" for param in node.parameters])
        
        # Proper C++ syntax: return_type function_name(parameters)
        return_type = node.return_type if node.return_type else "void"
        
        return f"""static inline {return_type} {node.name}({func_params}) {{
{node.code}
}}"""

    def _generate_operator_overload(self, node: OperatorOverloadNode) -> str:
        """Generate C++ code for an operator overload."""
        func_params = ", ".join([f"{param.param_type} {param.param_name}" for param in node.parameters])
        
        # Proper C++ syntax: return_type operator_symbol(parameters)
        return_type = node.return_type if node.return_type else "void"
        
        return f"""{return_type} {node.operator}({func_params}) {{
{node.code}
}}"""

    def _generate_constructor(self, node: ConstructorNode) -> str:
        """Generate C++ code for a constructor."""
        func_params = ", ".join([f"{param.param_type} {param.param_name}" for param in node.parameters])
        
        # Generate member initializers
        init_list = ""
        if node.member_initializers:
            init_parts = [f"{member}({value})" for member, value in node.member_initializers]
            init_list = f" : {', '.join(init_parts)}"
        
        # For constructors, we need the class name from context
        # This will be handled by the class generator that calls this method
        # For standalone constructors, we'll use a placeholder
        class_name = getattr(node, 'class_name', 'ClassName')
        
        # Proper C++ constructor syntax: ClassName(parameters) : init_list
        return f"""{class_name}({func_params}){init_list} {{
{node.code}
}}"""

    def _generate_destructor(self, node: DestructorNode) -> str:
        """Generate C++ code for a destructor."""
        # For destructors, we need the class name from context
        # This will be handled by the class generator that calls this method
        # For standalone destructors, we'll use a placeholder
        class_name = getattr(node, 'class_name', 'ClassName')
        
        # Proper C++ destructor syntax: ~ClassName()
        return f"""~{class_name}() {{
{node.code}
}}"""

    def _generate_bitfield(self, node: BitfieldNode) -> str:
        """Generate C++ code for a bitfield."""
        members = []
        for member in node.members:
            members.append(f"    {member.type} {member.name} : {member.bits};")
        
        return f"""struct {node.name} {{
{chr(10).join(members)}
}};"""

    def _generate_preprocessor_directive(self, node: PreprocessorDirectiveNode) -> str:
        """Generate C++ code for a preprocessor directive."""
        return f"#{node.directive_type} {node.content}"

    def _generate_function_attribute(self, node: FunctionAttributeNode) -> str:
        """Generate C++ code for a function with attributes."""
        func_params = ", ".join([f"{param.param_type} {param.param_name}" for param in node.parameters])
        
        # Generate attributes
        attr_str = " ".join([f"__{attr}__" for attr in node.attributes])
        
        # Proper C++ syntax: return_type function_name(parameters)
        return_type = node.return_type if node.return_type else "void"
        
        return f"""{attr_str} {return_type} {node.name}({func_params}) {{
{node.code}
}}"""

    def _generate_concept(self, node: ConceptNode) -> str:
        """Generate C++ code for a concept."""
        requirements = []
        for req in node.requires.requirements:
            requirements.append(f"    {req}")
        
        return f"""template<typename {node.requires.type_param}>
concept {node.name} = requires {{
{chr(10).join(requirements)}
}};"""

    def _generate_friend_declaration(self, node: FriendDeclarationNode) -> str:
        """Generate C++ code for a friend declaration."""
        if node.friend_type == "class":
            return f"friend class {node.target};"
        else:
            return f"friend {node.target};"

    def _generate_user_defined_literal(self, node: UserDefinedLiteralNode) -> str:
        """Generate C++ code for a user-defined literal."""
        # Proper C++ syntax: return_type operator""_suffix(parameters)
        return_type = node.return_type if node.return_type else "void"
        return f"""{return_type} operator""{node.literal_suffix}(long double value) {{
{node.code}
}}"""

    def _generate_method(self, method) -> str:
        """Generate C++ code for a method."""
        func_params = ", ".join([f"{param.param_type} {param.param_name}" for param in method.inputs])
        
        # Proper C++ syntax: return_type method_name(parameters)
        return_type = method.return_type if hasattr(method, 'return_type') and method.return_type else "void"
        
        return f"""    {return_type} {method.name}({func_params}) {{
{method.code}
    }}"""

    def _value_to_string(self, value) -> str:
        """Convert a value node to a string representation."""
        if hasattr(value, 'value'):
            if isinstance(value.value, str):
                return f'"{value.value}"'
            return str(value.value)
        return str(value) 