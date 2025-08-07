"""AST Builder for RoboDSL.

This module builds the AST from the Lark parse tree.
"""
from typing import Any, List, Optional, Union
from lark import Tree, Token
from ..core.ast import (
    RoboDSLAST, IncludeNode, StructNode, ClassNode, EnumNode,
    TypedefNode, UsingNode, NodeNode, CudaKernelsNode, OnnxModelNode,
    PipelineNode, MessageNode, ServiceNode, CustomActionNode,
    DynamicParameterNode, DynamicRemapNode, SimulationConfigNode,
    HardwareInLoopNode, StructContentNode, ClassContentNode,
    AccessSectionNode, StructMemberNode, CppMethodNode, MethodParamNode,
    MessageContentNode, MessageFieldNode,
    ServiceContentNode, ServiceRequestNode, ServiceResponseNode,
    ActionContentNode, ActionGoalNode, ActionFeedbackNode, ActionResultNode,
    ValueNode, InheritanceNode, EnumContentNode, EnumValueNode,
    NodeContentNode, ParameterNode, LifecycleNode, LifecycleSettingNode,
    TimerNode, TimerSettingNode, RemapNode, NamespaceNode, FlagNode,
    PublisherNode, SubscriberNode, ServicePrimitiveNode,
    ClientNode, ActionNode, QoSNode, QoSSettingNode, KernelNode,
    KernelContentNode, KernelParamNode, StageNode, StageContentNode,
    StageInputNode, StageOutputNode, StageMethodNode, StageModelNode,
    StageTopicNode, StageCudaKernelNode, StageOnnxModelNode,
    PipelineContentNode, ModelConfigNode, InputDefNode, OutputDefNode,
    DeviceNode, OptimizationNode, TensorRTConfigNode, SimulationWorldNode, SimulationRobotNode,
    SimulationPluginNode, KernelParameterDirection, RawCppCodeNode,
    # Advanced C++ Features AST Nodes
    TemplateParamNode, TemplateStructNode, TemplateClassNode, TemplateFunctionNode, TemplateAliasNode,
    StaticAssertNode, GlobalConstexprNode, GlobalDeviceConstNode, GlobalStaticInlineNode,
    OperatorOverloadNode, ConstructorNode, DestructorNode, BitfieldNode, BitfieldMemberNode,
    PreprocessorDirectiveNode, FunctionAttributeNode, ConceptNode, ConceptRequiresNode,
    FriendDeclarationNode, UserDefinedLiteralNode, PackageNode
)


class ASTBuilder:
    """AST builder for RoboDSL with improved structure and error handling."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.ast = None
        self.errors = []
        self.in_node_context = False  # Track if we're processing inside a node
        self.source_text = ""  # Store the original source text
        self.cpp_block_index = 0  # Track which cpp block we're processing
    
    def build(self, tree: Tree, source_text: str = "") -> RoboDSLAST:
        """Build AST from Lark parse tree."""
        self.ast = RoboDSLAST()
        self.errors = []
        self.source_text = source_text
        self.cpp_block_index = 0  # Reset block index
        
        if self.debug:
            print(f"Building AST from tree: {tree.data}")
        
        try:
            for child in tree.children:
                if self.debug:
                    print(f"Processing child: {child.data if isinstance(child, Tree) else type(child)}")
                if isinstance(child, Tree):
                    if child.data == "package_def":
                        self._process_package_def(child)
                    elif child.data == "cuda_kernel_def":
                        self._process_cuda_kernel_def(child)
                    elif child.data == "kernel_def":
                        # Handle standalone kernel definitions
                        self._process_standalone_kernel_def(child)
                    elif child.data == "data_structure":
                        self._process_data_structure(child)
                    elif child.data == "custom_interface":
                        self._process_custom_interface(child)
                    elif child.data == "node_def":
                        self._process_node_def(child)
                    elif child.data == "cpp_node_def":
                        self._process_cpp_node_def(child)
                    elif child.data == "cuda_kernels_block":
                        self._process_cuda_kernels_block(child)
                    elif child.data == "onnx_model":
                        self._process_onnx_model(child)
                    elif child.data == "pipeline_def":
                        self._process_pipeline_def(child)
                    elif child.data == "simulation_config":
                        self._process_simulation_config(child)
                    elif child.data == "dynamic_config":
                        self._process_dynamic_config(child)
                    elif child.data == "raw_cpp_code":
                        self._process_raw_cpp_code(child)
                    elif child.data == "advanced_cpp_feature":
                        self._process_advanced_cpp_feature(child)
                    else:
                        self._process_node(child)
        except Exception as e:
            if self.debug:
                print(f"Error during AST building: {e}")
            self.errors.append(f"AST building error: {e}")
        
        return self.ast
    
    def _process_node(self, node: Tree):
        """Process a parse tree node with error handling."""
        if self.debug:
            print(f"Processing node: {node.data}")
        
        try:
            method_name = f"_process_{node.data}"
            if hasattr(self, method_name):
                getattr(self, method_name)(node)
            else:
                if self.debug:
                    print(f"No handler for node type: {node.data}")
        except Exception as e:
            if self.debug:
                print(f"Error processing node {node.data}: {e}")
            self.errors.append(f"Error processing {node.data}: {e}")
    
    def _process_data_structure(self, node: Tree):
        """Process data structure definition (struct, class, enum, typedef, using)."""
        try:
            # data_structure can have multiple children: the definition and optional semicolon
            # Find the actual definition node (first Tree child)
            definition_node = None
            for child in node.children:
                if isinstance(child, Tree):
                    definition_node = child
                    break
            
            if definition_node:
                # Delegate to the appropriate handler
                if definition_node.data == "struct_def":
                    self._process_struct_def(definition_node)
                elif definition_node.data == "class_def":
                    self._process_class_def(definition_node)

                elif definition_node.data == "enum_def":
                    self._process_enum_def(definition_node)
                elif definition_node.data == "typedef_def":
                    self._process_typedef_def(definition_node)
                elif definition_node.data == "using_def":
                    self._process_using_def(definition_node)
                else:
                    if self.debug:
                        print(f"Unknown data structure type: {definition_node.data}")
                    self.errors.append(f"Unknown data structure type: {definition_node.data}")
            else:
                if self.debug:
                    print(f"No definition node found in data_structure")
                self.errors.append(f"No definition node found in data_structure")
        except Exception as e:
            if self.debug:
                print(f"Error processing data structure: {e}")
            self.errors.append(f"Data structure error: {e}")
    
    def _extract_token_value(self, token_or_tree: Union[Token, Tree]) -> str:
        """Extract string value from token or tree."""
        if self.debug:
            print(f"_extract_token_value: {type(token_or_tree)} = {token_or_tree}")
        if isinstance(token_or_tree, Token):
            return token_or_tree.value
        elif isinstance(token_or_tree, Tree):
            # Special handling for cpp_type and cpp_type_name
            if token_or_tree.data == "cpp_type":
                # cpp_type may have children like cpp_type_name, pointer, etc.
                return "".join(self._extract_token_value(child) for child in token_or_tree.children)
            elif token_or_tree.data == "cpp_type_name":
                return "".join(self._extract_token_value(child) for child in token_or_tree.children)
            # If the tree has a single Token child, return its value
            elif len(token_or_tree.children) == 1 and isinstance(token_or_tree.children[0], Token):
                return token_or_tree.children[0].value
            else:
                return str(token_or_tree)
        else:
            return str(token_or_tree)
    
    def _process_include_stmt(self, node: Tree):
        """Process include statement with proper path parsing."""
        try:
            # Check if this is a system include (has include_path child) or local include (direct string)
            if len(node.children) == 1 and isinstance(node.children[0], Tree) and node.children[0].data == "include_path":
                # System include with angle brackets: include <path>
                path_node = node.children[0]
                path = self._parse_include_path(path_node)
                is_system = True
            else:
                # Local include with quotes: include "path"
                path_child = node.children[0]
                path = self._parse_include_path(path_child)
                is_system = False
            
            include_node = IncludeNode(path=path, is_system=is_system)
            self.ast.includes.append(include_node)
            
            if self.debug:
                print(f"Added include: {path} (system: {is_system})")
                
        except Exception as e:
            if self.debug:
                print(f"Error processing include statement: {e}")
            self.errors.append(f"Include statement error: {e}")
    
    def _parse_include_path(self, path_node: Union[Tree, Token]) -> str:
        """Parse include path according to grammar structure."""
        if isinstance(path_node, Token):
            # Direct string token
            return path_node.value.strip('"<>')
        
        if isinstance(path_node, Tree):
            if path_node.data == "include_path":
                # Parse according to grammar: STRING | NAME ("/" NAME)* ("." NAME)?
                parts = []
                for child in path_node.children:
                    if isinstance(child, Token):
                        parts.append(child.value)
                    else:
                        parts.append(str(child))
                
                if self.debug:
                    print(f"Include path parts: {parts}")
                
                # Reconstruct path based on grammar structure
                if len(parts) == 1:
                    # Single STRING or NAME
                    return parts[0]
                elif len(parts) == 2:
                    # NAME "." NAME (extension)
                    return f"{parts[0]}.{parts[1]}"
                else:
                    # NAME ("/" NAME)* ("." NAME)?
                    # For paths like "rclcpp/rclcpp.hpp", we have 3 parts: ['rclcpp', 'rclcpp', 'hpp']
                    # The last part should be treated as an extension
                    if len(parts) == 3:
                        # Special case for "rclcpp/rclcpp.hpp" pattern
                        return f"{parts[0]}/{parts[1]}.{parts[2]}"
                    else:
                        # General case: join all parts except the last with slashes, then add dot + last part
                        path_parts = parts[:-1]
                        extension = parts[-1]
                        path = "/".join(path_parts)
                        path += f".{extension}"
                        return path
            else:
                # Fallback for other tree structures
                return str(path_node)
        
        return str(path_node)
    
    def _process_struct_def(self, node: Tree):
        """Process struct definition."""
        try:
            name = str(node.children[1])  # Use index 1 for name
            content = self._process_struct_content(node.children[3])
            
            # Check if this struct contains bitfield members
            has_bitfield_members = False
            for member in content.members:
                if hasattr(member, 'bits') and member.bits is not None:
                    has_bitfield_members = True
                    break
            
            if has_bitfield_members:
                # Create a BitfieldNode for structs with bitfield members
                bitfield_members = []
                for member in content.members:
                    if hasattr(member, 'bits') and member.bits is not None:
                        bitfield_members.append(BitfieldMemberNode(
                            type=member.type,
                            name=member.name,
                            bits=member.bits
                        ))
                
                bitfield_node = BitfieldNode(name=name, members=bitfield_members)
                self.ast.advanced_cpp_features.append(bitfield_node)
            else:
                # Regular struct
                struct_node = StructNode(name=name, content=content)
                self.ast.data_structures.append(struct_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing struct definition: {e}")
            self.errors.append(f"Struct definition error: {e}")
    
    def _process_struct_content(self, node: Tree, content: Optional[StructContentNode] = None) -> StructContentNode:
        """Process struct content."""
        if content is None:
            content = StructContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "struct_member":
                        content.members.append(self._process_struct_member(child))
                    elif child.data == "cpp_method":
                        content.methods.append(self._process_cpp_method(child))
                    elif child.data == "include_stmt":
                        self._process_include_stmt(child)
        except Exception as e:
            if self.debug:
                print(f"Error processing struct content: {e}")
            self.errors.append(f"Struct content error: {e}")
        return content
    
    def _process_struct_member(self, node: Tree) -> StructMemberNode:
        """Process struct member."""
        try:
            if self.debug:
                print(f"Processing struct_member with {len(node.children)} children: {[type(c) for c in node.children]}")
                for i, child in enumerate(node.children):
                    print(f"  child {i}: {type(child)} = {child}")
            
            # Extract type (first child)
            if isinstance(node.children[0], Tree) and node.children[0].data == "cpp_type":
                type_name = self._extract_cpp_type_string(node.children[0])
            else:
                type_name = str(node.children[0])
            
            # Extract name (second child)
            name = str(node.children[1])
            array_spec = None
            bits = None
            
            # Check if this is a bitfield member (has : bits syntax)
            if len(node.children) > 2:
                if hasattr(node.children[2], 'data') and node.children[2].data == "array_spec":
                    array_spec = str(node.children[2])
                elif len(node.children) == 3 and isinstance(node.children[2], Token) and node.children[2].type == "SIGNED_NUMBER":
                    # This is a bitfield member: type name : bits
                    bits = int(str(node.children[2]))
                    if self.debug:
                        print(f"Found bitfield member: {type_name} {name} : {bits}")
            
            member = StructMemberNode(type=type_name, name=name, array_spec=array_spec)
            if bits is not None:
                member.bits = bits
            return member
        except Exception as e:
            if self.debug:
                print(f"Error processing struct member: {e}")
            self.errors.append(f"Struct member error: {e}")
    
    def _process_class_def(self, node: Tree):
        """Process class definition."""
        try:
            # Find the NAME token among the children
            name = None
            for child in node.children:
                if isinstance(child, Token) and child.type == "NAME":
                    name = child.value
                    break
            if name is None:
                raise ValueError("Class name (NAME token) not found in class_def")
            inheritance = None
            content_node = None
            
            # Find inheritance and class_content among all children
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "inheritance":
                        inheritance = self._process_inheritance(child)
                    elif child.data == "class_content":
                        content_node = self._process_class_content(child)
            
            if content_node is None:
                content_node = ClassContentNode()
            
            class_node = ClassNode(name=name, content=content_node, inheritance=inheritance)
            self.ast.data_structures.append(class_node)
            if self.debug:
                print(f"[DEBUG] Appended ClassNode: {class_node}")
        except Exception as e:
            if self.debug:
                print(f"Error processing class definition: {e}")
            self.errors.append(f"Class definition error: {e}")
    
    def _process_class_content(self, node: Tree, content: Optional[ClassContentNode] = None) -> ClassContentNode:
        """Process class content."""
        if content is None:
            content = ClassContentNode()
        try:
            if self.debug:
                print(f"[DEBUG] Processing class_content with {len(node.children)} children")
            for child in node.children:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"[DEBUG] class_content child: {child.data}")
                    if child.data == "class_access_section":
                        if self.debug:
                            print(f"[DEBUG] Found class_access_section, processing...")
                        content.access_sections.append(self._process_class_access_section(child))
                    elif child.data == "class_direct_member":
                        content.members.append(self._process_struct_member(child.children[0]))
                    elif child.data == "class_direct_method":
                        content.methods.append(self._process_cpp_method(child.children[0]))
                    elif child.data == "include_stmt":
                        self._process_include_stmt(child)
        except Exception as e:
            if self.debug:
                print(f"Error processing class content: {e}")
            self.errors.append(f"Class content error: {e}")
        return content
    
    def _process_class_access_section(self, node: Tree) -> AccessSectionNode:
        """Process class access section."""
        try:
            # access_specifier is a tree node containing a token
            access_specifier_token = node.children[0]
            if isinstance(access_specifier_token, Tree) and access_specifier_token.children:
                access_specifier = str(access_specifier_token.children[0].value)
            elif hasattr(access_specifier_token, 'value'):
                access_specifier = str(access_specifier_token.value)
            else:
                access_specifier = str(access_specifier_token)
            if self.debug:
                print(f"[DEBUG] Creating AccessSectionNode with access_specifier: {access_specifier}")
            section = AccessSectionNode(access_specifier=access_specifier)
            
            for child in node.children[1:]:
                if isinstance(child, Tree):
                    if child.data == "class_section_member":
                        section.members.append(self._process_struct_member(child.children[0]))
                    elif child.data == "class_section_method":
                        cpp_method_node = child.children[0] if child.children else None
                        if cpp_method_node:
                            section.methods.append(self._process_cpp_method(cpp_method_node))
            
            return section
        except Exception as e:
            if self.debug:
                print(f"Error processing class access section: {e}")
            self.errors.append(f"Class access section error: {e}")
            return AccessSectionNode(access_specifier="public")
    
    def _process_inheritance(self, node: Tree) -> InheritanceNode:
        """Process inheritance."""
        try:
            base_classes = []
            current_access = "public"  # Default access
            
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "access_specifier":
                        current_access = str(child.children[0]) if child.children else "public"
                    elif child.data == "cpp_type":
                        class_name = self._extract_cpp_type_string(child)
                        base_classes.append((current_access, class_name))
                        current_access = "public"  # Reset for next base class
                else:
                    # Direct token (class name without access specifier)
                    class_name = str(child)
                    base_classes.append((current_access, class_name))
                    current_access = "public"  # Reset for next base class
            
            return InheritanceNode(base_classes=base_classes)
        except Exception as e:
            if self.debug:
                print(f"Error processing inheritance: {e}")
            self.errors.append(f"Inheritance error: {e}")
            return InheritanceNode(base_classes=[])
    
    def _process_cpp_method(self, node: Tree) -> CppMethodNode:
        try:
            name = str(node.children[0])
            method = CppMethodNode(name=name)
            if self.debug:
                print(f"[DEBUG] _process_cpp_method: method name: {name}")
            
            # Process method parameters and code
            for child in node.children[1:]:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"[DEBUG] Processing child: {child.data}")
                    
                    if child.data == "method_content":
                        # Handle method_content structure
                        for subchild in child.children:
                            if isinstance(subchild, Tree):
                                if subchild.data == "input_param":
                                    method.inputs.append(self._process_method_param(subchild))
                                elif subchild.data == "output_param":
                                    method.outputs.append(self._process_method_param(subchild))
                                elif subchild.data == "code_block":
                                    method.code = self._extract_code_from_block(subchild)
                                elif subchild.data == "balanced_braces":
                                    method.code = self._extract_code_from_balanced_braces(subchild)
                    elif child.data == "input_param":
                        method.inputs.append(self._process_method_param(child))
                    elif child.data == "output_param":
                        method.outputs.append(self._process_method_param(child))
                    elif child.data == "code_block":
                        method.code = self._extract_code_from_block(child)
                    elif child.data == "balanced_braces":
                        method.code = self._extract_code_from_balanced_braces(child)
                    elif child.data == "function_param_list":
                        # Handle function parameters directly
                        method.inputs = self._process_function_param_list(child)
                    elif child.data == "return_type":
                        # Handle return type
                        method.outputs = [MethodParamNode(
                            param_type=self._extract_cpp_type_string(child.children[0]),
                            param_name="return_value",
                            size_expr=None
                        )]
            
            if self.debug:
                print(f"[DEBUG] Method {name} - inputs: {len(method.inputs)}, outputs: {len(method.outputs)}, code: {method.code[:50] if method.code else 'None'}")
            
            return method
        except Exception as e:
            if self.debug:
                print(f"Error processing C++ method: {e}")
            self.errors.append(f"C++ method error: {e}")
            return CppMethodNode(name="")
    
    def _extract_code_from_block(self, code_block_node: Tree) -> str:
        """Extract code from a code block node."""
        if self.debug:
            print(f"[DEBUG] _extract_code_from_block called")
            print(f"[DEBUG] code_block_node: {code_block_node}")
            print(f"[DEBUG] code_block_node type: {type(code_block_node)}")
        
        try:
            if hasattr(code_block_node, 'children'):
                # It's a Tree, extract from children
                def collect_code(node):
                    if hasattr(node, 'value'):
                        return str(node.value)
                    elif hasattr(node, 'children'):
                        return ''.join(collect_code(child) for child in node.children)
                    else:
                        return str(node)
                
                return collect_code(code_block_node)
            else:
                # It's a Token, return its value
                return str(code_block_node.value)
        except Exception as e:
            if self.debug:
                print(f"Error extracting code from block: {e}")
            return ""
    
    def _extract_code_from_balanced_braces(self, balanced_braces_node: Tree) -> str:
        """Extract code from a balanced_braces node."""
        def collect_code(node):
            if isinstance(node, Token):
                return node.value
            elif isinstance(node, Tree):
                return "".join(collect_code(child) for child in node.children)
            else:
                return str(node)
        
        return collect_code(balanced_braces_node)
    
    def _process_method_param(self, node: Tree):
        """Process method parameter."""
        try:
            param_type = self._extract_cpp_type_string(node.children[0])
            param_name = str(node.children[1])
            size_expr = None
            if len(node.children) > 2 and hasattr(node.children[2], 'data') and node.children[2].data == "input_param_size":
                size_expr = str(node.children[2])
            return MethodParamNode(param_type=param_type, param_name=param_name, size_expr=size_expr)
        except Exception as e:
            print(f"Error processing method parameter: {e}")
            self.errors.append(f"Method parameter error: {e}")
            return MethodParamNode(param_type="", param_name="", size_expr=None)
    
    def _process_custom_interface(self, node: Tree):
        """Process custom interface (message, service, or action)."""
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    self._process_node(child)
        except Exception as e:
            if self.debug:
                print(f"Error processing custom interface: {e}")
            self.errors.append(f"Custom interface error: {e}")
    
    def _process_message_def(self, node: Tree):
        """Process message definition."""
        try:
            name = str(node.children[0])
            content = self._process_message_content(node.children[2])
            message_node = MessageNode(name=name, content=content)
            self.ast.messages.append(message_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing message definition: {e}")
            self.errors.append(f"Message definition error: {e}")
    
    def _process_message_content(self, node: Tree) -> MessageContentNode:
        """Process message content."""
        content = MessageContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree) and child.data == "message_field":
                    content.fields.append(self._process_message_field(child))
        except Exception as e:
            if self.debug:
                print(f"Error processing message content: {e}")
            self.errors.append(f"Message content error: {e}")
        return content
    
    def _process_message_field(self, node: Tree) -> MessageFieldNode:
        """Process message field."""
        try:
            field_type = str(node.children[0])
            field_name = str(node.children[1])
            array_spec = None
            default_value = None

            # If type ends with [] or similar, extract array_spec
            if field_type.endswith('[]'):
                array_spec = '[]'
                field_type = field_type[:-2]
                field_type = field_type.strip()

            for child in node.children[2:]:
                if isinstance(child, Tree):
                    if child.data == "default_value":
                        default_value = self._process_value(child.children[0])

            return MessageFieldNode(type=field_type, name=field_name, 
                                  array_spec=array_spec, default_value=default_value)
        except Exception as e:
            if self.debug:
                print(f"Error processing message field: {e}")
            self.errors.append(f"Message field error: {e}")
            return MessageFieldNode(type="", name="", array_spec=None, default_value=None)
    
    def _process_value(self, node) -> ValueNode:
        """Process value node."""
        # Recursively unwrap 'value' nodes
        while isinstance(node, Tree) and node.data == "value" and len(node.children) == 1:
            node = node.children[0]
        if isinstance(node, Tree):
            if node.data == "primitive":
                # Handle primitive values (boolean, string, number)
                primitive_value = node.children[0]
                if hasattr(primitive_value, 'type'):
                    if primitive_value.type == 'BOOLEAN':
                        return ValueNode(value=primitive_value.value == 'true')
                    elif primitive_value.type == 'STRING':
                        return ValueNode(value=primitive_value.value.strip('"'))
                    elif primitive_value.type == 'SIGNED_NUMBER':
                        # Try to convert to int first, then float
                        try:
                            if '.' in primitive_value.value:
                                return ValueNode(value=float(primitive_value.value))
                            else:
                                return ValueNode(value=int(primitive_value.value))
                        except ValueError:
                            return ValueNode(value=primitive_value.value)
                else:
                    return ValueNode(value=str(primitive_value))
            elif node.data == "array":
                # Handle array values
                values = []
                for child in node.children:
                    if isinstance(child, Tree) and child.data == "value_list":
                        for list_child in child.children:
                            if isinstance(list_child, Tree) and list_child.data == "value":
                                values.append(self._process_value(list_child).value)
                return ValueNode(value=values)
            elif node.data == "nested_dict":
                # Handle dictionary values
                result = {}
                for child in node.children:
                    if isinstance(child, Tree) and child.data == "dict_list":
                        for dict_child in child.children:
                            if isinstance(dict_child, Tree) and dict_child.data == "dict_item":
                                key = str(dict_child.children[0])
                                value = self._process_value(dict_child.children[1]).value
                                result[key] = value
                return ValueNode(value=result)
            else:
                return ValueNode(value=str(node))
        else:
            # Handle direct tokens
            if hasattr(node, 'type'):
                if node.type == 'BOOLEAN':
                    return ValueNode(value=node.value == 'true')
                elif node.type == 'STRING':
                    return ValueNode(value=node.value.strip('"'))
                elif node.type == 'SIGNED_NUMBER':
                    try:
                        if '.' in node.value:
                            return ValueNode(value=float(node.value))
                        else:
                            return ValueNode(value=int(node.value))
                    except ValueError:
                        return ValueNode(value=node.value)
            return ValueNode(value=str(node))

    def _process_service_content(self, node: Tree) -> ServiceContentNode:
        """Process service content."""
        request = None
        response = None
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "service_request":
                    request = self._process_service_request(child)
                elif child.data == "service_response":
                    response = self._process_service_response(child)
        return ServiceContentNode(request=request, response=response)

    def _process_service_request(self, node: Tree) -> ServiceRequestNode:
        fields = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "message_field":
                fields.append(self._process_message_field(child))
        return ServiceRequestNode(fields=fields)

    def _process_service_response(self, node: Tree) -> ServiceResponseNode:
        fields = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "message_field":
                fields.append(self._process_message_field(child))
        return ServiceResponseNode(fields=fields)

    def _process_service_def(self, node: Tree):
        name = str(node.children[0])
        # The structure is: service NAME LBRACE service_content RBRACE
        content = self._process_service_content(node.children[2])
        service_node = ServiceNode(name=name, content=content)
        self.ast.services.append(service_node)

    def _process_action_content(self, node: Tree) -> ActionContentNode:
        goal = None
        feedback = None
        result = None
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "action_goal":
                    goal = self._process_action_goal(child)
                elif child.data == "action_feedback":
                    feedback = self._process_action_feedback(child)
                elif child.data == "action_result":
                    result = self._process_action_result(child)
        return ActionContentNode(goal=goal, feedback=feedback, result=result)

    def _process_action_goal(self, node: Tree) -> ActionGoalNode:
        fields = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "message_field":
                fields.append(self._process_message_field(child))
        return ActionGoalNode(fields=fields)

    def _process_action_feedback(self, node: Tree) -> ActionFeedbackNode:
        fields = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "message_field":
                fields.append(self._process_message_field(child))
        return ActionFeedbackNode(fields=fields)

    def _process_action_result(self, node: Tree) -> ActionResultNode:
        fields = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "message_field":
                fields.append(self._process_message_field(child))
        return ActionResultNode(fields=fields)

    def _process_action_def(self, node: Tree):
        name = str(node.children[0])
        # The structure is: action NAME LBRACE action_content RBRACE
        content = self._process_action_content(node.children[2])
        action_node = CustomActionNode(name=name, content=content)
        self.ast.actions.append(action_node)

    def _process_node_def(self, node: Tree):
        """Process node definition."""
        try:
            name = str(node.children[1])  # NODE_NAME is at index 1, NODE is at index 0
            
            # Validate node name - subnodes (with dots) are not allowed in robodsl code
            if '.' in name:
                error_msg = (
                    f"Invalid node name '{name}': Subnodes with dots (.) are not allowed in RoboDSL code. "
                    f"Subnodes are a CLI-only feature for organizing files. "
                    f"Use a simple node name without dots, or create subnodes using the CLI command: "
                    f"'robodsl create-node {name}'"
                )
                self.errors.append(error_msg)
                raise ValueError(error_msg)
            
            # Set context to indicate we're processing inside a node
            self.in_node_context = True
            content = self._process_node_content(node.children[3])  # content is at index 3, LBRACE is at index 2
            # Reset context
            self.in_node_context = False
            node_node = NodeNode(name=name, content=content)
            self.ast.nodes.append(node_node)
        except Exception as e:
            # Reset context on error
            self.in_node_context = False
            if self.debug:
                print(f"Error processing node definition: {e}")
            if str(e) not in self.errors:  # Avoid duplicate error messages
                self.errors.append(f"Node definition error: {e}")

    def _process_node_content(self, node: Tree) -> NodeContentNode:
        """Process node content."""
        content = NodeContentNode()
        try:
            if self.debug:
                print(f"Processing node_content with {len(node.children)} children")
                print(f"node_content type: {type(node)}")
                print(f"node_content: {node}")
            for child in node.children:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"  Processing child: {child.data}")
                        print(f"    Child type: {type(child)}")
                        print(f"    Child children: {len(child.children)}")
                    if child.data == "parameter":
                        content.parameters.append(self._process_parameter(child))
                    elif child.data == "lifecycle":
                        lifecycle_node = self._process_lifecycle(child)
                        if content.lifecycle is None:
                            content.lifecycle = lifecycle_node
                        else:
                            # Merge lifecycle settings
                            content.lifecycle.settings.extend(lifecycle_node.settings)
                    elif child.data == "timer":
                        content.timers.append(self._process_timer(child))
                    elif child.data == "remap":
                        content.remaps.append(self._process_remap(child))
                    elif child.data == "namespace":
                        content.namespace = self._process_namespace(child)
                    elif child.data == "ros_primitive":
                        # Handle ROS primitives (publisher, subscriber, service, client, action)
                        primitive_child = child.children[0]  # The actual primitive (publisher, etc.)
                        if primitive_child.data == "publisher":
                            content.publishers.append(self._process_publisher(primitive_child))
                        elif primitive_child.data == "subscriber":
                            content.subscribers.append(self._process_subscriber(primitive_child))
                        elif primitive_child.data == "service":
                            content.services.append(self._process_service_primitive(primitive_child))
                        elif primitive_child.data == "client":
                            content.clients.append(self._process_client(primitive_child))
                        elif primitive_child.data == "action":
                            content.actions.append(self._process_action_primitive(primitive_child))
                    elif child.data == "flag":
                        content.flags.append(self._process_flag(child))
                    elif child.data == "cpp_method":
                        content.cpp_methods.append(self._process_cpp_method(child))
                    elif child.data == "kernel_def":
                        # Individual kernel definitions within nodes
                        kernel = self._process_kernel_def(child)
                        content.cuda_kernels.append(kernel)
                    elif child.data == "onnx_model_ref":
                        # Process ONNX model reference inside node
                        onnx_model = self._process_onnx_model_ref(child)
                        content.onnx_models.append(onnx_model)
                    elif child.data == "cuda_kernels_block":
                        # Block of kernel definitions
                        kernels_block = self._process_cuda_kernels_block(child)
                        if kernels_block and hasattr(kernels_block, 'kernels'):
                            content.cuda_kernels.extend(kernels_block.kernels)
                    elif child.data == "use_kernel":
                        content.used_kernels.append(self._process_use_kernel(child))
                    elif child.data == "raw_cpp_code":
                        content.raw_cpp_code.append(self._process_raw_cpp_code(child))
                    elif child.data == "parameter_server":
                        content.parameter_server = str(child.children[0]) == "true"
                    elif child.data == "parameter_client":
                        content.parameter_client = str(child.children[0]) == "true"
                    elif child.data == "service_client":
                        content.clients.append(self._process_service_client(child))
                    elif child.data == "action_client":
                        content.clients.append(self._process_action_client(child))
                    elif child.data == "realtime_config":
                        # Handle realtime configuration
                        # For now, just set a flag to indicate realtime is enabled
                        content.realtime_enabled = True
                    elif child.data == "security_config":
                        # Handle security configuration
                        # For now, just set a flag to indicate security is enabled
                        content.security_enabled = True
                    elif child.data == "component_config":
                        # Handle component configuration
                        # For now, just set a flag to indicate component is enabled
                        content.component_enabled = True
                    else:
                        if self.debug:
                            print(f"    No handler for {child.data}")
        except Exception as e:
            if self.debug:
                print(f"Error processing node content: {e}")
            self.errors.append(f"Node content error: {e}")
        return content

    def _process_parameter(self, node: Tree) -> ParameterNode:
        """Process parameter."""
        try:
            if len(node.children) < 4:
                # New parameter syntax: parameter: "name" -> "type" { ... }
                if len(node.children) == 3:
                    name = str(node.children[0]).strip('"')
                    param_type = str(node.children[1]).strip('"')
                    config = node.children[2]
                    
                    # Extract default value from config if present
                    default_value = None
                    if isinstance(config, Tree) and config.data == "parameter_config":
                        for child in config.children:
                            if isinstance(child, Tree) and child.data == "parameter_setting":
                                if len(child.children) >= 2:
                                    setting_name = str(child.children[0])
                                    if setting_name == "default":
                                        default_value = self._process_value(child.children[1])
                    
                    return ParameterNode(type=param_type, name=name, value=default_value)
                else:
                    if self.debug:
                        print(f"Parameter node has insufficient children: {len(node.children)}")
                        print(f"Parameter node children: {[str(c) for c in node.children]}")
                    return ParameterNode(type="", name="", value=None)
            else:
                # Old parameter syntax: parameter: cpp_type NAME = value
                # Use _extract_cpp_type_string to convert cpp_type tree to string
                param_type = self._extract_cpp_type_string(node.children[1])
                name = str(node.children[2])  # NAME is at index 2
                value = self._process_value(node.children[3])  # value is at index 3
                return ParameterNode(type=param_type, name=name, value=value)
        except Exception as e:
            if self.debug:
                print(f"Error processing parameter: {e}")
                print(f"Parameter node children: {[str(c) for c in node.children]}")
            return ParameterNode(type="", name="", value=None)

    def _process_lifecycle(self, node: Tree) -> LifecycleNode:
        """Process lifecycle configuration."""
        settings = []
        
        # Handle different lifecycle syntax forms
        if len(node.children) == 1 and hasattr(node.children[0], 'type') and node.children[0].type == 'BOOLEAN':
            # Simple form: lifecycle: true
            value = str(node.children[0]) == "true"
            if value:
                settings.append(LifecycleSettingNode(name="enabled", value=True))
        elif len(node.children) == 2 and str(node.children[0]) == "lifecycle_states":
            # lifecycle_states: ["unconfigured", "inactive", "active", "finalized"]
            settings.append(LifecycleSettingNode(name="states", value=True))
        elif len(node.children) == 2 and str(node.children[0]) == "lifecycle_transitions":
            # lifecycle_transitions: ["configure", "activate", "deactivate", "cleanup", "shutdown"]
            settings.append(LifecycleSettingNode(name="transitions", value=True))
        else:
            # Complex form with braces
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "lifecycle_setting":
                        settings.append(self._process_lifecycle_setting(child))
                    elif child.data == "lifecycle_config":
                        # Handle lifecycle_config block
                        for config_child in child.children:
                            if isinstance(config_child, Tree) and config_child.data == "lifecycle_setting":
                                settings.append(self._process_lifecycle_setting(config_child))
        
        return LifecycleNode(settings=settings)

    def _process_lifecycle_setting(self, node: Tree) -> LifecycleSettingNode:
        """Process lifecycle setting."""
        name = str(node.children[0])
        value = str(node.children[1]) == "true"
        return LifecycleSettingNode(name=name, value=value)

    def _process_timer(self, node: Tree) -> TimerNode:
        """Process timer."""
        name = str(node.children[0])
        period_node = node.children[1]
        
        # Extract the actual value from the expression tree
        if isinstance(period_node, Tree) and period_node.data == "expr":
            period = self._extract_value_from_expr(period_node)
        else:
            period = str(period_node)
        
        settings = []
        if len(node.children) > 2:
            config = self._process_timer_config(node.children[2])
            if config:
                for setting_name, value in config.items():
                    settings.append(TimerSettingNode(name=setting_name, value=value))
        return TimerNode(name=name, period=period, settings=settings)

    def _process_timer_config(self, node: Tree):
        """Process timer configuration."""
        settings = {}
        for child in node.children:
            if isinstance(child, Tree) and child.data == "timer_setting":
                # Handle timer_setting nodes - they contain oneshot/autostart nodes
                for setting_child in child.children:
                    if isinstance(setting_child, Tree) and setting_child.data in ["oneshot", "autostart"]:
                        if len(setting_child.children) >= 2:
                            name = setting_child.data
                            value_node = setting_child.children[1]
                            if hasattr(value_node, 'type') and value_node.type == 'BOOLEAN':
                                value = value_node.value == 'true'
                            else:
                                value = str(value_node) == 'true'
                            settings[name] = value
            elif isinstance(child, Tree) and child.data in ["oneshot", "autostart"]:
                # Handle direct oneshot/autostart nodes
                if len(child.children) >= 2:
                    name = child.data
                    value_node = child.children[1]
                    if hasattr(value_node, 'type') and value_node.type == 'BOOLEAN':
                        value = value_node.value == 'true'
                    else:
                        value = str(value_node) == 'true'
                    settings[name] = value
        return settings

    def _process_remap(self, node: Tree) -> RemapNode:
        """Process remap."""
        # Handles both 'remap from: /old to: /new' and 'remap /old : /new'
        if len(node.children) == 4:
            # remap from: /old/topic to: /new/topic
            from_topic = self._topic_path_to_str(node.children[1])
            to_topic = self._topic_path_to_str(node.children[3])
        elif len(node.children) == 3:
            # remap /old/topic : /new/topic
            from_topic = self._topic_path_to_str(node.children[0])
            to_topic = self._topic_path_to_str(node.children[2])
        else:
            # fallback (legacy or error)
            from_topic = self._topic_path_to_str(node.children[0])
            to_topic = self._topic_path_to_str(node.children[1])
        return RemapNode(from_topic=from_topic, to_topic=to_topic)

    def _process_namespace(self, node: Tree) -> NamespaceNode:
        """Process namespace."""
        namespace = self._topic_path_to_str(node.children[0])
        return NamespaceNode(namespace=namespace)

    def _process_publisher(self, node: Tree) -> PublisherNode:
        """Process publisher."""
        topic_node = node.children[0]
        topic = self._topic_path_to_str(topic_node) if isinstance(topic_node, Tree) else str(topic_node)
        msg_type_node = node.children[1]
        if isinstance(msg_type_node, Tree):
            if msg_type_node.data == "topic_type":
                # Handle both quoted strings and NAME ("/" NAME)+ syntax
                if len(msg_type_node.children) == 1:
                    msg_type = str(msg_type_node.children[0]).strip('"')
                else:
                    # Concatenate all children for NAME ("/" NAME)+ syntax
                    msg_type = "/".join(str(child) for child in msg_type_node.children)
            else:
                msg_type = str(msg_type_node)
        else:
            msg_type = str(msg_type_node).strip('"')
        qos = None
        if len(node.children) > 2:
            config = self._process_publisher_config(node.children[2])
            qos = config.get("qos") if config else None
        return PublisherNode(topic=topic, msg_type=msg_type, qos=qos)

    def _process_publisher_config(self, node: Tree):
        """Process publisher configuration."""
        settings = {}
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "qos_config":
                    settings["qos"] = self._process_qos_config(child)
                elif child.data == "publisher_setting":
                    # publisher_setting can be either qos_config or NAME ":" value
                    if len(child.children) == 1 and child.children[0].data == "qos_config":
                        settings["qos"] = self._process_qos_config(child.children[0])
                    elif len(child.children) >= 2:
                        name = str(child.children[0])
                        value = str(child.children[1])
                        settings[name] = value
        return settings

    def _process_subscriber(self, node: Tree) -> SubscriberNode:
        """Process subscriber."""
        topic_node = node.children[0]
        topic = self._topic_path_to_str(topic_node) if isinstance(topic_node, Tree) else str(topic_node)
        msg_type_node = node.children[1]
        if isinstance(msg_type_node, Tree):
            if msg_type_node.data == "topic_type":
                # Handle both quoted strings and NAME ("/" NAME)+ syntax
                if len(msg_type_node.children) == 1:
                    msg_type = str(msg_type_node.children[0]).strip('"')
                else:
                    # Concatenate all children for NAME ("/" NAME)+ syntax
                    msg_type = "/".join(str(child) for child in msg_type_node.children)
            else:
                msg_type = str(msg_type_node)
        else:
            msg_type = str(msg_type_node).strip('"')
        qos = None
        if len(node.children) > 2:
            config = self._process_subscriber_config(node.children[2])
            qos = config.get("qos") if config else None
        return SubscriberNode(topic=topic, msg_type=msg_type, qos=qos)

    def _process_subscriber_config(self, node: Tree):
        """Process subscriber configuration."""
        settings = {}
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "qos_config":
                    settings["qos"] = self._process_qos_config(child)
                elif child.data == "subscriber_setting":
                    # subscriber_setting can be either qos_config or NAME ":" value
                    if len(child.children) == 1 and child.children[0].data == "qos_config":
                        settings["qos"] = self._process_qos_config(child.children[0])
                    elif len(child.children) >= 2:
                        name = str(child.children[0])
                        value = str(child.children[1])
                        settings[name] = value
        return settings

    def _process_service_primitive(self, node: Tree) -> ServicePrimitiveNode:
        """Process service primitive."""
        service_node = node.children[0]
        service = self._topic_path_to_str(service_node) if isinstance(service_node, Tree) else str(service_node)
        srv_type_node = node.children[1]
        if isinstance(srv_type_node, Tree):
            if srv_type_node.data == "topic_type":
                # Handle both quoted strings and NAME ("/" NAME)+ syntax
                if len(srv_type_node.children) == 1:
                    srv_type = str(srv_type_node.children[0]).strip('"')
                else:
                    # Concatenate all children for NAME ("/" NAME)+ syntax
                    srv_type = "/".join(str(child) for child in srv_type_node.children)
            else:
                srv_type = str(srv_type_node)
        else:
            srv_type = str(srv_type_node).strip('"')
        qos = None
        if len(node.children) > 2:
            config = self._process_service_config(node.children[2])
            qos = config.get("qos") if config else None
        return ServicePrimitiveNode(service=service, srv_type=srv_type, qos=qos)

    def _process_service_config(self, node: Tree):
        """Process service configuration."""
        settings = {}
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "qos_config":
                    settings["qos"] = self._process_qos_config(child)
                elif child.data == "service_setting":
                    # service_setting can be either qos_config or NAME ":" value
                    if len(child.children) == 1 and child.children[0].data == "qos_config":
                        settings["qos"] = self._process_qos_config(child.children[0])
                    elif len(child.children) >= 2:
                        name = str(child.children[0])
                        value = str(child.children[1])
                        settings[name] = value
        return settings

    def _process_client(self, node: Tree) -> ClientNode:
        """Process client."""
        service_node = node.children[0]
        if isinstance(service_node, Tree):
            # Handle service_path structure
            service_parts = []
            for child in service_node.children:
                if hasattr(child, 'value'):
                    service_parts.append(child.value)
                else:
                    service_parts.append(str(child))
            service = "/" + "/".join(service_parts)
        else:
            service = str(service_node)
        srv_type = str(node.children[1])
        qos = None
        if len(node.children) > 2:
            config = self._process_client_config(node.children[2])
            qos = config.get("qos") if config else None
        return ClientNode(service=service, srv_type=srv_type, qos=qos)

    def _process_service_client(self, node: Tree) -> ClientNode:
        """Process service client."""
        # service_client has the same structure as client
        return self._process_client(node)

    def _process_action_client(self, node: Tree) -> ClientNode:
        """Process action client."""
        # action_client has the same structure as client
        return self._process_client(node)

    def _process_client_config(self, node: Tree):
        """Process client configuration."""
        settings = {}
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "qos_config":
                    settings["qos"] = self._process_qos_config(child)
                elif child.data == "client_setting":
                    # client_setting can be either qos_config or NAME ":" value
                    if len(child.children) == 1 and child.children[0].data == "qos_config":
                        settings["qos"] = self._process_qos_config(child.children[0])
                    elif len(child.children) >= 2:
                        name = str(child.children[0])
                        value = str(child.children[1])
                        settings[name] = value
        return settings

    def _process_action_primitive(self, node: Tree) -> ActionNode:
        """Process action primitive."""
        topic_path_node = node.children[0]
        name = self._topic_path_to_str(topic_path_node) if isinstance(topic_path_node, Tree) else str(topic_path_node)
        action_type_node = node.children[1]
        if isinstance(action_type_node, Tree):
            if action_type_node.data == "topic_type":
                # Handle both quoted strings and NAME ("/" NAME)+ syntax
                if len(action_type_node.children) == 1:
                    action_type = str(action_type_node.children[0]).strip('"')
                else:
                    # Concatenate all children for NAME ("/" NAME)+ syntax
                    action_type = "/".join(str(child) for child in action_type_node.children)
            else:
                action_type = str(action_type_node.children[0]).strip('"')
        else:
            action_type = str(action_type_node).strip('"')
        qos = None
        if len(node.children) > 2:
            config = self._process_action_config(node.children[2])
            qos = config.get("qos") if config else None
        return ActionNode(name=name, action_type=action_type, qos=qos)

    def _process_action_config(self, node: Tree):
        """Process action configuration."""
        settings = {}
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "qos_config":
                    settings["qos"] = self._process_qos_config(child)
                elif child.data == "action_setting":
                    # action_setting can be either qos_config or NAME ":" value
                    if len(child.children) == 1 and child.children[0].data == "qos_config":
                        settings["qos"] = self._process_qos_config(child.children[0])
                    elif len(child.children) >= 2:
                        name = str(child.children[0])
                        value = str(child.children[1])
                        settings[name] = value
        return settings

    def _process_flag(self, node: Tree) -> FlagNode:
        """Process flag."""
        name = str(node.children[0])
        value = str(node.children[1]) == "true"
        return FlagNode(name=name, value=value)

    def _process_kernel_def(self, node: Tree) -> KernelNode:
        """Process kernel definition."""
        try:
            name = str(node.children[1])  # NAME is at index 1, KERNEL is at index 0
            content = self._process_kernel_content(node.children[3])  # content is at index 3, LBRACE is at index 2
            return KernelNode(name=name, content=content)
        except Exception as e:
            if self.debug:
                print(f"Error processing kernel definition: {e}")
            self.errors.append(f"Kernel definition error: {e}")
            return KernelNode(name="", content=KernelContentNode())

    def _process_kernel_content(self, node: Tree) -> KernelContentNode:
        content = KernelContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"[DEBUG] Processing kernel content child: {child.data}")
                    if child.data == "block_size":
                        # The tuple_expr is at index 0
                        if len(child.children) > 0:
                            if self.debug:
                                print(f"[DEBUG] Processing block_size, children: {child.children}")
                                print(f"[DEBUG] block_size child[0]: {child.children[0]}")
                            content.block_size = self._parse_tuple_expression(child.children[0])
                    elif child.data == "grid_size":
                        # The tuple_expr is at index 0
                        if len(child.children) > 0:
                            if self.debug:
                                print(f"[DEBUG] Processing grid_size, children: {child.children}")
                                print(f"[DEBUG] grid_size child[0]: {child.children[0]}")
                            content.grid_size = self._parse_tuple_expression(child.children[0])
                    elif child.data == "shared_memory":
                        # The expr is at index 0
                        expr_child = child.children[0] if len(child.children) > 0 else None
                        if expr_child is not None:
                            if self.debug:
                                print(f"Processing shared_memory expr: {expr_child}")
                            content.shared_memory = self._parse_expression(expr_child)
                    elif child.data == "use_thrust":
                        # The BOOLEAN is at index 0
                        if len(child.children) > 0:
                            value = str(child.children[0])
                            content.use_thrust = value == "true"
                    elif child.data == "kernel_input_param":
                        # The kernel_param_list is at index 0
                        if len(child.children) > 0:
                            content.parameters.extend(self._process_kernel_param_list(child.children[0], KernelParameterDirection.IN))
                    elif child.data == "kernel_output_param":
                        # The kernel_param_list is at index 0
                        if len(child.children) > 0:
                            content.parameters.extend(self._process_kernel_param_list(child.children[0], KernelParameterDirection.OUT))
                    elif child.data == "cuda_include":
                        # Handle CUDA include statements
                        include_path = self._parse_include_path(child.children[0])
                        content.cuda_includes.append(include_path)
                    elif child.data == "kernel_parameters":
                        # Handle kernel parameters
                        if self.debug:
                            print(f"[DEBUG] Found kernel_parameters node with {len(child.children)} children")
                            for i, grandchild in enumerate(child.children):
                                print(f"[DEBUG] kernel_parameters child {i}: {type(grandchild)} = {grandchild}")
                                if isinstance(grandchild, Tree):
                                    print(f"[DEBUG] kernel_parameters child {i} data: {grandchild.data}")
                                    print(f"[DEBUG] kernel_parameters child {i} children: {[type(c) for c in grandchild.children]}")
                        if len(child.children) > 1:
                            # The kernel_parameter_list is at index 1 (index 0 is '{', index 2 is '}')
                            content.kernel_parameters = self._process_kernel_parameter_list(child.children[1])
                            if self.debug:
                                print(f"[DEBUG] Processed kernel parameters: {content.kernel_parameters}")
                    elif child.data == "code_block":
                        if len(child.children) > 0:
                            extracted_code = self._extract_code_from_block(child.children[0])
                            content.code = extracted_code
        except Exception as e:
            if self.debug:
                print(f"Error processing kernel content: {e}")
            self.errors.append(f"Kernel content error: {e}")
        return content

    def _process_cuda_kernel_content(self, node: Tree) -> KernelContentNode:
        """Process cuda_kernel content with additional CUDA-specific features."""
        content = KernelContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"[DEBUG] Processing cuda_kernel content child: {child.data}")
                    if child.data == "kernel":
                        # Handle kernel code block
                        if len(child.children) > 0:
                            extracted_code = self._extract_code_from_block(child.children[0])
                            content.code = extracted_code
                    elif child.data == "block_size":
                        # The tuple_expr is at index 0
                        if len(child.children) > 0:
                            if self.debug:
                                print(f"[DEBUG] Processing block_size, children: {child.children}")
                                print(f"[DEBUG] block_size child[0]: {child.children[0]}")
                            content.block_size = self._parse_tuple_expression(child.children[0])
                    elif child.data == "grid_size":
                        # The tuple_expr is at index 0
                        if len(child.children) > 0:
                            if self.debug:
                                print(f"[DEBUG] Processing grid_size, children: {child.children}")
                                print(f"[DEBUG] grid_size child[0]: {child.children[0]}")
                            content.grid_size = self._parse_tuple_expression(child.children[0])
                    elif child.data == "shared_memory":
                        # The expr is at index 0
                        expr_child = child.children[0] if len(child.children) > 0 else None
                        if expr_child is not None:
                            if self.debug:
                                print(f"Processing shared_memory expr: {expr_child}")
                            content.shared_memory = self._parse_expression(expr_child)
                    elif child.data == "use_thrust":
                        # The BOOLEAN is at index 0
                        if len(child.children) > 0:
                            value = str(child.children[0])
                            content.use_thrust = value == "true"
                    elif child.data == "inputs":
                        # Handle inputs array
                        if len(child.children) > 0:
                            content.inputs = self._process_array(child.children[0])
                    elif child.data == "outputs":
                        # Handle outputs array
                        if len(child.children) > 0:
                            content.outputs = self._process_array(child.children[0])
                    elif child.data == "kernel_parameters":
                        # Handle kernel parameters
                        if self.debug:
                            print(f"[DEBUG] Found kernel_parameters node with {len(child.children)} children")
                            for i, grandchild in enumerate(child.children):
                                print(f"[DEBUG] kernel_parameters child {i}: {type(grandchild)} = {grandchild}")
                                if isinstance(grandchild, Tree):
                                    print(f"[DEBUG] kernel_parameters child {i} data: {grandchild.data}")
                                    print(f"[DEBUG] kernel_parameters child {i} children: {[type(c) for c in grandchild.children]}")
                        if len(child.children) > 1:
                            # The kernel_parameter_list is at index 1 (index 0 is '{', index 2 is '}')
                            content.kernel_parameters = self._process_kernel_parameter_list(child.children[1])
                            if self.debug:
                                print(f"[DEBUG] Processed kernel parameters: {content.kernel_parameters}")
                    elif child.data == "cuda_include":
                        # Handle CUDA include statements
                        include_path = self._parse_include_path(child.children[0])
                        content.cuda_includes.append(include_path)
                    elif child.data == "code_block":
                        if len(child.children) > 0:
                            extracted_code = self._extract_code_from_block(child.children[0])
                            content.code = extracted_code
                    elif child.data in ["use_streams", "stream_count", "synchronize", "dynamic_parallelism", "multi_gpu", "gpu_count", "gpu_memory_per_device", "gpu_architectures", "compute_capabilities", "error_handling", "performance_monitoring", "memory_hierarchy", "concurrent_kernel", "advanced_synchronization", "mixed_precision", "memory_pool", "context_management", "driver_api", "runtime_api"]:
                        # Handle individual CUDA advanced features
                        if self.debug:
                            print(f"[DEBUG] Processing CUDA advanced feature: {child.data}")
                        # Store advanced features in content for later processing
                        if not hasattr(content, 'cuda_advanced_features'):
                            content.cuda_advanced_features = {}
                        content.cuda_advanced_features[child.data] = self._process_value(child.children[0]).value
        except Exception as e:
            if self.debug:
                print(f"Error processing cuda_kernel content: {e}")
            self.errors.append(f"CUDA kernel content error: {e}")
        return content

    def _process_kernel_param_list(self, node: Tree, direction) -> List[KernelParamNode]:
        """Process kernel parameter list."""
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "kernel_param":
                params.append(self._process_kernel_param(child, direction))
        return params

    def _process_kernel_parameter_list(self, node: Tree) -> List[dict]:
        """Process kernel parameter list for kernel parameters."""
        if self.debug:
            print(f"[DEBUG] _process_kernel_parameter_list called with node: {node}")
            print(f"[DEBUG] node data: {node.data}")
            print(f"[DEBUG] node children: {[type(c) for c in node.children]}")
            for i, child in enumerate(node.children):
                print(f"[DEBUG] child {i}: {type(child)} = {child}")
                if isinstance(child, Tree):
                    print(f"[DEBUG] child {i} data: {child.data}")
                    print(f"[DEBUG] child {i} children: {[type(c) for c in child.children]}")
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "kernel_parameter_def":
                # Use _extract_cpp_type_string to get the C++ type string
                param_type = self._extract_cpp_type_string(child.children[0])
                param_name = str(child.children[1])
                param_value = self._process_value(child.children[2]).value
                params.append({
                    'type': param_type,
                    'name': param_name,
                    'value': param_value
                })
        if self.debug:
            print(f"[DEBUG] _process_kernel_parameter_list returning: {params}")
        return params

    def _process_kernel_param(self, node: Tree, direction) -> KernelParamNode:
        """Process kernel parameter."""
        # Extract cpp_type as string, including pointer/array modifiers
        type_node = node.children[0]
        if isinstance(type_node, Tree) and type_node.data == "cpp_type":
            # Extract the full cpp_type_name including pointer/array modifiers
            param_type = self._extract_cpp_type_string(type_node)
        else:
            param_type = str(type_node)
        
        param_name = str(node.children[1])
        size_expr = None
        if len(node.children) > 2:
            size_node = node.children[2]
            if isinstance(size_node, Tree) and size_node.data == "kernel_param_size":
                # Extract size expression as list, filtering out parentheses and other non-content tokens
                size_expr = []
                for child in size_node.children:
                    if isinstance(child, Tree) and child.data == "kernel_param_size_list":
                        for size_item in child.children:
                            if isinstance(size_item, Tree) and size_item.data == "kernel_param_size_item":
                                size_expr.append(str(size_item.children[0]))
                            else:
                                # Filter out parentheses and other non-content tokens
                                item_str = str(size_item)
                                if item_str not in ['(', ')', ','] and not item_str.isspace():
                                    size_expr.append(item_str)
                    else:
                        # Filter out parentheses and other non-content tokens
                        child_str = str(child)
                        if child_str not in ['(', ')', ','] and not child_str.isspace():
                            size_expr.append(child_str)
            else:
                # For simple size expressions, filter out parentheses
                size_str = str(size_node)
                if size_str not in ['(', ')', ','] and not size_str.isspace():
                    size_expr = [size_str]
                else:
                    size_expr = []
        return KernelParamNode(direction=direction, param_type=param_type, param_name=param_name, size_expr=size_expr)

    def _parse_tuple_expression(self, node: Tree) -> tuple:
        """Parse a tuple expression like (expr, expr, expr)."""
        if self.debug:
            print(f"[DEBUG] _parse_tuple_expression called with: {node}")
            print(f"[DEBUG] node type: {type(node)}")
            print(f"[DEBUG] node children: {node.children}")
        
        if not isinstance(node, Tree):
            if self.debug:
                print(f"[DEBUG] Not a Tree, returning default values: (256, 1, 1)")
            return (256, 1, 1)  # Default values
        
        # The structure should be: block_size: (expr, expr, expr)
        # children: [LPAR, expr1, COMMA, expr2, COMMA, expr3, RPAR]
        if len(node.children) >= 7:
            expr1 = node.children[1]
            expr2 = node.children[3]
            expr3 = node.children[5]
            val1 = self._parse_expression(expr1)
            val2 = self._parse_expression(expr2)
            val3 = self._parse_expression(expr3)
            result = (val1, val2, val3)
            if self.debug:
                print(f"[DEBUG] Parsed tuple values: {result}")
            return result
        else:
            # Fallback: try to extract expressions from available children
            expressions = []
            for child in node.children:
                if isinstance(child, Tree) and child.data == "expr":
                    expressions.append(self._parse_expression(child))
            
            if len(expressions) >= 3:
                result = tuple(expressions[:3])
                if self.debug:
                    print(f"[DEBUG] Parsed tuple from expr children: {result}")
                return result
            else:
                # Pad with default values
                while len(expressions) < 3:
                    expressions.append(1)
                result = tuple(expressions)
                if self.debug:
                    print(f"[DEBUG] Padded tuple with defaults: {result}")
                return result

    def _parse_expression(self, node):
        """Parse a simple expression and return an integer value or string."""
        # If it's a Tree, recursively descend to the first token
        while isinstance(node, Tree):
            if hasattr(node, 'children') and len(node.children) > 0:
                node = node.children[0]
            else:
                return None
        # Now node should be a Token
        if hasattr(node, 'type'):
            if node.type == 'SIGNED_NUMBER':
                try:
                    return int(node.value)
                except ValueError:
                    return node.value
            elif node.type == 'NAME':
                return str(node.value)
            else:
                return str(node.value)
        return str(node)

    def _process_onnx_model_ref(self, node: Tree) -> OnnxModelNode:
        """Process ONNX model reference."""
        name = str(node.children[0])
        # The structure is: NAME LBRACE onnx_model_content RBRACE
        # So children[2] is the content
        config = self._process_onnx_model_content(node.children[2])
        return OnnxModelNode(name=name, config=config)

    def _process_cuda_kernels_block(self, node: Tree):
        """Process CUDA kernels block."""
        kernels = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "kernel_def":
                kernels.append(self._process_kernel_def(child))
        
        # Create CudaKernelsNode and return it for use in node content
        cuda_kernels_node = CudaKernelsNode(kernels=kernels)
        
        # Only add to global AST if this is NOT inside a node context
        if hasattr(self, 'ast') and self.ast is not None and not self.in_node_context:
            self.ast.cuda_kernels = cuda_kernels_node
        
        return cuda_kernels_node

    def _process_use_kernel(self, node: Tree) -> str:
        """Process use_kernel directive."""
        return str(node.children[0]).strip('"')

    def _process_qos_config(self, node: Tree) -> QoSNode:
        """Process QoS configuration."""
        settings = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "qos_setting":
                settings.append(self._process_qos_setting(child))
        return QoSNode(settings=settings)

    def _process_qos_setting(self, node: Tree) -> QoSSettingNode:
        """Process QoS setting."""
        name = str(node.children[0])
        value_node = node.children[1]
        
        # Extract the actual value from the expression tree
        if isinstance(value_node, Tree):
            if value_node.data == "expr":
                # Handle expressions - traverse the tree to find the actual value
                value = self._extract_value_from_expr(value_node)
            else:
                value = str(value_node)
        else:
            # Direct token
            if hasattr(value_node, 'type'):
                if value_node.type == 'NAME':
                    value = value_node.value
                elif value_node.type == 'SIGNED_NUMBER':
                    try:
                        if '.' in value_node.value:
                            value = float(value_node.value)
                        else:
                            value = int(value_node.value)
                    except ValueError:
                        value = value_node.value
                else:
                    value = value_node.value
            else:
                value = str(value_node)
        
        return QoSSettingNode(name=name, value=value)
    
    def _extract_value_from_expr(self, expr_node: Tree):
        """Extract the actual value from an expression tree by traversing it."""
        # Traverse the expression tree to find the actual value
        current = expr_node
        while isinstance(current, Tree):
            if current.data == "signed_atom":
                # We've reached the signed_atom, extract the value
                if len(current.children) > 0:
                    value = current.children[0]
                    if hasattr(value, 'type'):
                        if value.type == 'NAME':
                            # For QoS enum values like 'reliable', 'best_effort', etc.
                            return value.value
                        elif value.type == 'SIGNED_NUMBER':
                            # For numeric values
                            try:
                                if '.' in value.value:
                                    return float(value.value)
                                else:
                                    return int(value.value)
                            except ValueError:
                                return value.value
                        else:
                            return value.value
                    else:
                        return str(value)
                else:
                    return str(current)
            elif len(current.children) > 0:
                # Continue traversing with the first child
                current = current.children[0]
            else:
                # No more children, return the string representation
                return str(current)
        
        # If we get here, return the string representation
        return str(expr_node)

    def _process_pipeline_def(self, node: Tree):
        """Process pipeline definition."""
        name = str(node.children[0])
        # The structure is: pipeline NAME LBRACE pipeline_content RBRACE
        content = self._process_pipeline_content(node.children[2])
        pipeline_node = PipelineNode(name=name, content=content)
        self.ast.pipelines.append(pipeline_node)

    def _process_pipeline_content(self, node: Tree) -> PipelineContentNode:
        """Process pipeline content."""
        stages = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "stage_def":
                stages.append(self._process_stage_def(child))
        return PipelineContentNode(stages=stages)

    def _process_stage_def(self, node: Tree) -> StageNode:
        """Process stage definition."""
        name = str(node.children[0])
        # The structure is: stage NAME LBRACE stage_content RBRACE
        content = self._process_stage_content(node.children[2])
        return StageNode(name=name, content=content)

    def _process_stage_content(self, node: Tree) -> StageContentNode:
        """Process stage content."""
        content = StageContentNode()
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "stage_input":
                    # Handle stage_input: "input_name"
                    if len(child.children) >= 1:
                        input_name = str(child.children[0]).strip('"')
                        content.inputs.append(StageInputNode(input_name=input_name))
                elif child.data == "stage_output":
                    # Handle stage_output: "output_name"
                    if len(child.children) >= 1:
                        output_name = str(child.children[0]).strip('"')
                        content.outputs.append(StageOutputNode(output_name=output_name))
                elif child.data == "stage_method":
                    # Handle stage_method: "method_name"
                    if len(child.children) >= 1:
                        method_name = str(child.children[0]).strip('"')
                        content.methods.append(StageMethodNode(method_name=method_name))
                elif child.data == "stage_model":
                    # Handle stage_model: "model_name"
                    if len(child.children) >= 1:
                        model_name = str(child.children[0]).strip('"')
                        content.models.append(StageModelNode(model_name=model_name))
                elif child.data == "stage_topic":
                    # Handle stage_topic: /topic/path
                    if len(child.children) >= 1:
                        topic_path = self._topic_path_to_str(child.children[0])
                        content.topics.append(StageTopicNode(topic_path=topic_path))
                elif child.data == "stage_cuda_kernel":
                    # Handle stage_cuda_kernel: "kernel_name"
                    if len(child.children) >= 1:
                        kernel_name = str(child.children[0]).strip('"')
                        content.cuda_kernels.append(StageCudaKernelNode(kernel_name=kernel_name))
                elif child.data == "stage_onnx_model":
                    # Handle stage_onnx_model: "model_name"
                    if len(child.children) >= 1:
                        model_name = str(child.children[0]).strip('"')
                        content.onnx_models.append(StageOnnxModelNode(model_name=model_name))
        return content

    def _process_onnx_model(self, node: Tree):
        """Process ONNX model definition."""
        name = str(node.children[0])
        # The structure is: onnx_model NAME LBRACE onnx_model_content RBRACE
        config = self._process_onnx_model_content(node.children[2])
        onnx_node = OnnxModelNode(name=name, config=config)
        self.ast.onnx_models.append(onnx_node)

    def _process_onnx_model_content(self, node: Tree) -> ModelConfigNode:
        """Process ONNX model content."""
        config = ModelConfigNode()
        tensorrt_config = None
        
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "input_def":
                    config.inputs.append(self._process_input_def(child))
                elif child.data == "output_def":
                    config.outputs.append(self._process_output_def(child))
                elif child.data == "device":
                    config.device = DeviceNode(device=str(child.children[0]))
                elif child.data == "optimization":
                    config.optimizations.append(OptimizationNode(optimization=str(child.children[0])))
                elif child.data == "tensorrt_config_item":
                    if tensorrt_config is None:
                        tensorrt_config = TensorRTConfigNode()
                    self._process_tensorrt_config_item(child, tensorrt_config)
                elif child.data == "config_block":
                    # Handle config block that contains input_def, output_def, etc.
                    for config_child in child.children:
                        if isinstance(config_child, Tree):
                            if config_child.data == "input_def":
                                config.inputs.append(self._process_input_def(config_child))
                            elif config_child.data == "output_def":
                                config.outputs.append(self._process_output_def(config_child))
                            elif config_child.data == "device":
                                config.device = DeviceNode(device=str(config_child.children[0]))
                            elif config_child.data == "optimization":
                                config.optimizations.append(OptimizationNode(optimization=str(config_child.children[0])))
                            elif config_child.data == "tensorrt_config_item":
                                if tensorrt_config is None:
                                    tensorrt_config = TensorRTConfigNode()
                                self._process_tensorrt_config_item(config_child, tensorrt_config)
        
        # Set TensorRT config if found
        if tensorrt_config:
            config.tensorrt_config = tensorrt_config
            
        return config

    def _process_input_def(self, node: Tree) -> InputDefNode:
        """Process input definition."""
        name = str(node.children[0]).strip('"')
        type_name = str(node.children[1]).strip('"')
        return InputDefNode(name=name, type=type_name)

    def _process_output_def(self, node: Tree) -> OutputDefNode:
        """Process output definition."""
        name = str(node.children[0]).strip('"')
        type_name = str(node.children[1]).strip('"')
        return OutputDefNode(name=name, type=type_name)

    def _process_tensorrt_config_item(self, node: Tree, config: TensorRTConfigNode):
        """Process individual TensorRT configuration item."""
        if node.data == "optimization_level":
            config.optimization_level = int(str(node.children[0]))
        elif node.data == "precision":
            config.precision = str(node.children[0]).strip('"')
        elif node.data == "dynamic_batch":
            config.dynamic_batch = str(node.children[0]).lower() == "true"
        elif node.data == "max_workspace_size":
            config.max_workspace_size = int(str(node.children[0]))
        elif node.data == "tactic_sources":
            tactic_sources = self._process_array(node.children[0])
            config.tactic_sources = [str(v.value) for v in tactic_sources]
        elif node.data == "timing_cache":
            config.timing_cache = str(node.children[0]).lower() == "true"
        elif node.data == "profiling_verbosity":
            config.profiling_verbosity = str(node.children[0]).strip('"')
        elif node.data == "calibration":
            config.calibration = str(node.children[0]).lower() == "true"
        elif node.data == "dynamic_range":
            config.dynamic_range = str(node.children[0]).lower() == "true"
        elif node.data == "min_batch_size":
            config.min_batch_size = int(str(node.children[0]))
        elif node.data == "max_batch_size":
            config.max_batch_size = int(str(node.children[0]))
        elif node.data == "optimal_batch_size":
            config.optimal_batch_size = int(str(node.children[0]))
        elif node.data == "dynamic_shapes":
            config.dynamic_shapes = str(node.children[0]).lower() == "true"
        elif node.data == "shape_optimization":
            config.shape_optimization = str(node.children[0]).lower() == "true"
        elif node.data == "plugins":
            plugins = self._process_array(node.children[0])
            config.plugins = [str(v.value) for v in plugins]
        elif node.data == "performance_tuning":
            config.performance_tuning = str(node.children[0]).lower() == "true"
        elif node.data == "memory_optimization":
            config.memory_optimization = str(node.children[0]).lower() == "true"
        elif node.data == "multi_stream":
            config.multi_stream = str(node.children[0]).lower() == "true"
        elif node.data == "performance_monitoring":
            config.performance_monitoring = str(node.children[0]).lower() == "true"
        elif node.data == "compatibility_mode":
            config.compatibility_mode = str(node.children[0]).lower() == "true"
        elif node.data == "memory_management":
            config.memory_management = str(node.children[0]).lower() == "true"
        elif node.data == "serialization":
            config.serialization = str(node.children[0]).lower() == "true"
        elif node.data == "parallel_execution":
            config.parallel_execution = str(node.children[0]).lower() == "true"
        elif node.data == "error_recovery":
            config.error_recovery = str(node.children[0]).lower() == "true"
        elif node.data == "parallel_streams":
            config.parallel_streams = int(str(node.children[0]))
        elif node.data == "error_handling":
            config.error_handling = str(node.children[0]).strip('"')
        elif node.data == "per_tensor_quantization":
            config.per_tensor_quantization = str(node.children[0]).lower() == "true"
        elif node.data == "memory_pool_size":
            config.memory_pool_size = int(str(node.children[0]))
        elif node.data == "monitoring_metrics":
            metrics = self._process_array(node.children[0])
            config.monitoring_metrics = [str(v.value) for v in metrics]
        elif node.data == "backward_compatibility":
            config.backward_compatibility = str(node.children[0]).lower() == "true"
        elif node.data == "plugin_paths":
            paths = self._process_array(node.children[0])
            config.plugin_paths = [str(v.value) for v in paths]
        elif node.data == "tuning_algorithm":
            config.tuning_algorithm = str(node.children[0]).strip('"')
        elif node.data == "engine_file":
            config.engine_file = str(node.children[0]).strip('"')
        elif node.data == "profiling":
            config.profiling = str(node.children[0]).lower() == "true"
        elif node.data == "debugging":
            config.debugging = str(node.children[0]).lower() == "true"
        elif node.data == "per_channel_quantization":
            config.per_channel_quantization = str(node.children[0]).lower() == "true"
        elif node.data == "calibration_data":
            config.calibration_data = str(node.children[0]).strip('"')
        elif node.data == "calibration_algorithm":
            config.calibration_algorithm = str(node.children[0]).strip('"')
        elif node.data == "calibration_batch_size":
            config.calibration_batch_size = int(str(node.children[0]))

    def _process_simulation_config(self, node: Tree):
        """Process simulation configuration."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "simulation_def":
                    self._process_simulation_def(child)
                elif child.data == "hil_config":
                    self._process_hil_config(child)

    def _process_simulation_def(self, node: Tree):
        """Process simulation definition."""
        simulator_type = str(node.children[0])
        content = self._process_simulation_content(node.children[1])
        simulation_node = SimulationConfigNode(simulator_type=simulator_type, content=content)
        self.ast.simulation = simulation_node

    def _process_simulation_content(self, node: Tree):
        """Process simulation content."""
        # This would need to be implemented based on the simulation AST nodes
        pass

    def _process_hil_config(self, node: Tree):
        """Process hardware-in-the-loop configuration."""
        content = self._process_hil_content(node.children[0])
        hil_node = HardwareInLoopNode(content=content)
        self.ast.hil_config = hil_node

    def _process_hil_content(self, node: Tree):
        """Process HIL content."""
        # This would need to be implemented based on the HIL AST nodes
        pass

    def _process_dynamic_config(self, node: Tree):
        """Process dynamic configuration."""
        for child in node.children:
            if isinstance(child, Tree):
                if child.data == "dynamic_parameters":
                    self._process_dynamic_parameters(child)
                elif child.data == "dynamic_remaps":
                    self._process_dynamic_remaps(child)

    def _process_dynamic_parameters(self, node: Tree):
        """Process dynamic parameters."""
        for child in node.children:
            if isinstance(child, Tree) and child.data == "dynamic_parameter":
                param = self._process_dynamic_parameter(child)
                self.ast.dynamic_parameters.append(param)

    def _process_dynamic_parameter(self, node: Tree) -> DynamicParameterNode:
        """Process dynamic parameter."""
        param_type = str(node.children[0])
        name = str(node.children[1])
        value = self._process_value(node.children[2])
        config = None
        if len(node.children) > 3:
            config = self._process_dynamic_param_config(node.children[3])
        return DynamicParameterNode(type=param_type, name=name, value=value, config=config)

    def _process_dynamic_param_config(self, node: Tree):
        """Process dynamic parameter configuration."""
        # This would need to be implemented based on the dynamic param config AST nodes
        pass

    def _process_dynamic_remaps(self, node: Tree):
        """Process dynamic remaps."""
        for child in node.children:
            if isinstance(child, Tree) and child.data == "dynamic_remap":
                remap = self._process_dynamic_remap(child)
                self.ast.dynamic_remaps.append(remap)

    def _process_dynamic_remap(self, node: Tree) -> DynamicRemapNode:
        """Process dynamic remap."""
        from_topic = self._topic_path_to_str(node.children[0])
        to_topic = self._topic_path_to_str(node.children[1])
        condition = None
        if len(node.children) > 2:
            condition = str(node.children[2])
        return DynamicRemapNode(from_topic=from_topic, to_topic=to_topic, condition=condition)

    def _process_typedef_def(self, node: Tree):
        """Process typedef definition."""
        try:
            # typedef_def: "typedef" cpp_type NAME ";"
            if len(node.children) >= 2:
                cpp_type = self._extract_token_value(node.children[0])
                new_name = self._extract_token_value(node.children[1])
                typedef_node = TypedefNode(cpp_type, new_name)
                self.ast.data_structures.append(typedef_node)
                if self.debug:
                    print(f"Added typedef: {new_name} = {cpp_type}")
            else:
                if self.debug:
                    print(f"typedef_def has {len(node.children)} children, expected at least 2")
                self.errors.append(f"typedef_def has {len(node.children)} children, expected at least 2")
        except Exception as e:
            if self.debug:
                print(f"Error processing typedef definition: {e}")
            self.errors.append(f"Typedef definition error: {e}")
    
    def _process_using_def(self, node: Tree):
        """Process using definition."""
        try:
            # using_def: "using" NAME "=" cpp_type ";"
            if len(node.children) >= 2:
                new_name = self._extract_token_value(node.children[0])
                cpp_type = self._extract_token_value(node.children[1])
                using_node = UsingNode(cpp_type, new_name)
                self.ast.data_structures.append(using_node)
                if self.debug:
                    print(f"Added using: {new_name} = {cpp_type}")
            else:
                if self.debug:
                    print(f"using_def has {len(node.children)} children, expected at least 2")
                self.errors.append(f"using_def has {len(node.children)} children, expected at least 2")
        except Exception as e:
            if self.debug:
                print(f"Error processing using definition: {e}")
            self.errors.append(f"Using definition error: {e}")
    
    def _process_enum_def(self, node: Tree):
        """Process enum definition."""
        try:
            # enum_def: "enum" enum_type? NAME LBRACE enum_content RBRACE
            # enum_type: "class" | "struct"
            # enum_content: enum_value (COMMA enum_value)* COMMA?
            # enum_value: NAME ("=" expr)?
            
            # Find the name, enum_type, and content
            name = None
            enum_type = None
            content_node = None
            
            for child in node.children:
                if isinstance(child, Token) and child.type == "NAME":
                    if name is None:
                        name = child.value
                elif isinstance(child, Tree) and child.data == "enum_content":
                    content_node = child
                elif isinstance(child, Tree) and child.data == "enum_type":
                    # enum_type node contains the "class" or "struct" token
                    if child.children and isinstance(child.children[0], Token):
                        enum_type = child.children[0].value
                elif isinstance(child, Token) and child.value in ["class", "struct"]:
                    enum_type = child.value
            
            if name and content_node:
                content = self._process_enum_content(content_node)
                enum_node = EnumNode(name=name, enum_type=enum_type, content=content)
                self.ast.data_structures.append(enum_node)
                if self.debug:
                    print(f"Added enum: {name} (type: {enum_type})")
            else:
                if self.debug:
                    print(f"enum_def missing name or content")
                self.errors.append(f"enum_def missing name or content")
        except Exception as e:
            if self.debug:
                print(f"Error processing enum definition: {e}")
            self.errors.append(f"Enum definition error: {e}")
    
    def _process_enum_content(self, node: Tree) -> EnumContentNode:
        """Process enum content."""
        content = EnumContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree) and child.data == "enum_value":
                    content.values.append(self._process_enum_value(child))
        except Exception as e:
            if self.debug:
                print(f"Error processing enum content: {e}")
            self.errors.append(f"Enum content error: {e}")
        return content
    
    def _process_enum_value(self, node: Tree) -> EnumValueNode:
        """Process enum value."""
        try:
            name = self._extract_token_value(node.children[0])
            value = None
            if len(node.children) > 1:
                value = self._process_value(node.children[1])
            return EnumValueNode(name=name, value=value)
        except Exception as e:
            if self.debug:
                print(f"Error processing enum value: {e}")
            self.errors.append(f"Enum value error: {e}")
    


    def _process_raw_cpp_code(self, node: Tree) -> RawCppCodeNode:
        """Process raw C++ code block that gets passed through as-is."""
        try:
            # Try to extract code from source text first for perfect preservation
            if self.source_text:
                code = self._extract_cpp_code_from_source(node, self.source_text)
            else:
                # Fallback to the enhanced method for C++ content
                code = self._extract_cpp_code_from_block(node)
            
            location = "node" if self.in_node_context else "global"
            
            raw_cpp_node = RawCppCodeNode(code=code, location=location)
            
            # If this is global code (not inside a node), add it to the AST
            if location == "global":
                self.ast.raw_cpp_code.append(raw_cpp_node)
                self.cpp_block_index += 1  # Increment for next global block
            
            if self.debug:
                print(f"Processed raw C++ code block (location: {location}): {len(code)} characters")
            
            return raw_cpp_node
        except Exception as e:
            if self.debug:
                print(f"Error processing raw C++ code block: {e}")
            self.errors.append(f"Raw C++ code block error: {e}")
            return RawCppCodeNode(code="", location="global")
    
    def _extract_cpp_code_from_block(self, cpp_block_node: Tree) -> str:
        """Extract C++ code from a raw C++ code block, preserving all syntax including comments."""
        if self.debug:
            print(f"[DEBUG] _extract_cpp_code_from_block called")
            print(f"[DEBUG] cpp_block_node: {cpp_block_node}")
            print(f"[DEBUG] cpp_block_node type: {type(cpp_block_node)}")
            print(f"[DEBUG] cpp_block_node children: {cpp_block_node.children}")
        
        def collect_cpp_code(node):
            """Recursively collect C++ code, preserving all tokens including comments and whitespace."""
            if isinstance(node, Token):
                # For tokens, return the value as-is
                return str(node.value)
            elif isinstance(node, Tree):
                # For tree nodes, recursively collect from children
                result = ""
                for child in node.children:
                    result += collect_cpp_code(child)
                return result
            else:
                return str(node)
        
        try:
            # The cpp_block_node should have children that are cpp_raw_content nodes
            code_parts = []
            for child in cpp_block_node.children:
                if isinstance(child, Tree):
                    if child.data == "cpp_raw_content":
                        # Each cpp_raw_content can contain text or nested braces
                        for content_child in child.children:
                            code_parts.append(collect_cpp_code(content_child))
                    elif child.data == "cpp_balanced_braces":
                        # Handle nested braces
                        code_parts.append(collect_cpp_code(child))
                    else:
                        # Fallback: treat as regular content
                        code_parts.append(collect_cpp_code(child))
                else:
                    # Direct token (like LBRACE, RBRACE)
                    code_parts.append(collect_cpp_code(child))
            
            # Join all parts
            code = "".join(code_parts)
            
            if self.debug:
                print(f"[DEBUG] Extracted code: {repr(code)}")
            
            return code
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] Error in _extract_cpp_code_from_block: {e}")
            # Fallback to the original method
            return self._extract_code_from_block(cpp_block_node) 

    def _extract_cpp_code_from_source(self, node: Tree, source_text: str) -> str:
        """Extract C++ code directly from source text for perfect preservation."""
        try:
            # Determine if we're in a node context
            is_node_context = self.in_node_context
            
            if is_node_context:
                # For node-level blocks, we need to find the specific block within the node
                # Since we don't have precise line information, fall back to the block method
                return self._extract_cpp_code_from_block(node)
            else:
                # For global blocks, use the index-based approach
                lines = source_text.split('\n')
                cpp_blocks = []
                
                i = 0
                while i < len(lines):
                    line = lines[i]
                    if 'cpp:' in line:
                        # Find the opening brace
                        brace_pos = line.find('{', line.find('cpp:'))
                        if brace_pos != -1:
                            # Found a cpp block, extract it
                            start_line = i
                            start_col = brace_pos
                            
                            # Find the matching closing brace
                            brace_count = 0
                            end_line = None
                            end_col = None
                            
                            for j, search_line in enumerate(lines[i:], i):
                                if j == i:
                                    # First line: start from the brace position
                                    line_start = brace_pos
                                else:
                                    line_start = 0
                                
                                for k, char in enumerate(search_line[line_start:], line_start):
                                    if char == '{':
                                        if brace_count == 0:
                                            # This is the opening brace
                                            pass
                                        brace_count += 1
                                    elif char == '}':
                                        brace_count -= 1
                                        if brace_count == 0:
                                            # This is the matching closing brace
                                            end_line = j
                                            end_col = k
                                            break
                                
                                if end_line is not None:
                                    break
                            
                            if end_line is not None:
                                # Extract the code between the braces
                                if start_line == end_line:
                                    # Same line
                                    code = lines[start_line][start_col+1:end_col]
                                else:
                                    # Multiple lines
                                    code_parts = []
                                    # First line: from after the opening brace to end
                                    code_parts.append(lines[start_line][start_col+1:])
                                    # Middle lines: full lines
                                    for k in range(start_line + 1, end_line):
                                        code_parts.append(lines[k])
                                    # Last line: from start to before the closing brace
                                    code_parts.append(lines[end_line][:end_col])
                                    code = '\n'.join(code_parts)
                                
                                # If the code is only whitespace, return '{}'
                                if code.strip() == "":
                                    code = "{}"
                                
                                cpp_blocks.append(code)
                                i = end_line + 1
                                continue
                    
                    i += 1
                
                # Return the block at the current index
                if cpp_blocks:
                    if self.cpp_block_index < len(cpp_blocks):
                        return cpp_blocks[self.cpp_block_index]
                    else:
                        # Fallback to first block if index is out of range
                        return cpp_blocks[0]
                
                # If no cpp blocks found but we have source text, try to find empty blocks
                if 'cpp:' in source_text and '{' in source_text and '}' in source_text:
                    # Look for empty cpp blocks like "cpp: { }"
                    import re
                    empty_pattern = r'cpp:\s*\{\s*\}'
                    if re.search(empty_pattern, source_text):
                        return "{}"
            
            # Fallback to the original method
            return self._extract_cpp_code_from_block(node)
            
        except Exception as e:
            if self.debug:
                print(f"Error in _extract_cpp_code_from_source: {e}")
            return self._extract_cpp_code_from_block(node)

    # Advanced C++ Features Processing (Phase 8)
    
    def _process_advanced_cpp_feature(self, node: Tree):
        """Process advanced C++ features."""
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    method_name = f"_process_{child.data}"
                    if hasattr(self, method_name):
                        getattr(self, method_name)(child)
                    else:
                        if self.debug:
                            print(f"No handler for advanced_cpp_feature child: {child.data}")
        except Exception as e:
            if self.debug:
                print(f"Error processing advanced_cpp_feature: {e}")
            self.errors.append(f"Advanced C++ feature error: {e}")

    def _process_template_function(self, node: Tree):
        """Process template function definition."""
        try:
            if self.debug:
                print(f"template_function children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            # Extract template parameters
            template_params = self._process_template_params(node.children[0])
            
            # Extract return type and function name
            return_type = self._extract_cpp_type_string(node.children[1])
            func_name = self._extract_token_value(node.children[2])
            
            # Extract parameter list (skip LPAR, get function_param_list, skip RPAR)
            param_list = self._process_function_param_list(node.children[4])
            
            # Extract code block from balanced_braces
            code = ""
            if len(node.children) > 6:
                code_node = node.children[6]
                if isinstance(code_node, Tree) and code_node.data == "balanced_braces":
                    code = self._extract_code_from_balanced_braces(code_node)
            
            template_function = TemplateFunctionNode(
                name=func_name,
                template_params=template_params,
                parameters=param_list,
                return_type=return_type,
                code=code
            )
            self.ast.advanced_cpp_features.append(template_function)
        except Exception as e:
            if self.debug:
                print(f"Error processing template function: {e}")
            self.errors.append(f"Template function error: {e}")

    def _process_template_def(self, node: Tree):
        """Process template definition."""
        try:
            # Find the actual template type node
            template_type_node = None
            for child in node.children:
                if isinstance(child, Tree):
                    template_type_node = child
                    break
            
            if template_type_node:
                if template_type_node.data == "template_struct":
                    self._process_template_struct(template_type_node)
                elif template_type_node.data == "template_class":
                    self._process_template_class(template_type_node)
                elif template_type_node.data == "template_function":
                    self._process_template_function(template_type_node)
                elif template_type_node.data == "template_alias":
                    self._process_template_alias(template_type_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing template definition: {e}")
            self.errors.append(f"Template definition error: {e}")

    def _process_template_struct(self, node: Tree):
        """Process template struct definition."""
        try:
            if self.debug:
                print(f"template_struct children: {[ (type(child), getattr(child, 'type', getattr(child, 'data', child))) for child in node.children ]}")
            
            # Parse tree structure: template_params struct NAME { block_content }
            params = self._process_template_params(node.children[0])  # template_params at index 0
            name = self._extract_token_value(node.children[2])  # NAME at index 2 (after struct token)
            content = self._process_struct_content(node.children[4])  # block_content at index 4 (after {)
            
            template_struct = TemplateStructNode(name=name, template_params=params, content=content)
            self.ast.advanced_cpp_features.append(template_struct)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing template struct: {e}")
            self.errors.append(f"Template struct error: {e}")

    def _process_template_class(self, node: Tree):
        """Process template class definition."""
        try:
            if self.debug:
                print(f"template_class children: {[ (type(child), getattr(child, 'type', getattr(child, 'data', child))) for child in node.children ]}")
            
            # Parse tree structure: template_params class NAME { block_content }
            params = self._process_template_params(node.children[0])  # template_params at index 0
            name = self._extract_token_value(node.children[2])  # NAME at index 2 (after class token)
            
            # Check for inheritance (optional)
            inheritance = None
            content = self._process_class_content(node.children[4])  # block_content at index 4 (after {)
            
            template_class = TemplateClassNode(name=name, template_params=params, content=content, inheritance=inheritance)
            self.ast.advanced_cpp_features.append(template_class)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing template class: {e}")
            self.errors.append(f"Template class error: {e}")

    def _extract_cpp_type_string(self, node: Tree) -> str:
        """Recursively extract a C++ type string from a Lark tree node."""
        if isinstance(node, str):
            return node
        if hasattr(node, 'type'):
            if node.type == 'NAME':
                return node.value
            if node.type == 'CONST':
                return 'const'
            if node.type in ('STAR', 'AMPERSAND'):
                return node.value
            if node.type == 'LONG':
                return 'long'
            if node.type == 'DOUBLE':
                return 'double'
            if node.type == 'SIGNED_NUMBER':
                return node.value
            if node.type == 'STRING':
                return node.value
            if node.type == 'RULE':
                return ' '.join(self._extract_cpp_type_string(child) for child in node.children)
        if isinstance(node, Tree):
            if node.data == 'cpp_type':
                # Special handling for cpp_type to avoid spaces around pointers/references
                parts = []
                for child in node.children:
                    part = self._extract_cpp_type_string(child)
                    if part:
                        parts.append(part)
                # Join without spaces to handle pointer/reference types correctly
                return ''.join(parts)
            elif node.data == 'base_cpp_type':
                # Join all children with spaces
                return ' '.join(self._extract_cpp_type_string(child) for child in node.children if self._extract_cpp_type_string(child))
            elif node.data == 'cpp_type_name':
                # Join consecutive NAME tokens with '::', and handle template args with no extra spaces
                parts = []
                name_parts = []
                i = 0
                while i < len(node.children):
                    child = node.children[i]
                    if hasattr(child, 'type') and child.type == 'NAME':
                        name_parts.append(child.value)
                        i += 1
                        while i < len(node.children) and hasattr(node.children[i], 'type') and node.children[i].type == 'NAME':
                            name_parts.append(node.children[i].value)
                            i += 1
                        parts.append('::'.join(name_parts))
                        name_parts = []
                    elif hasattr(child, 'type') and child.type == 'LESS_THAN':
                        # Template arguments: emit <args> with no spaces
                        j = i + 1
                        arg_parts = []
                        while j < len(node.children) and (not hasattr(node.children[j], 'type') or node.children[j].type != 'GREATER_THAN'):
                            arg_str = self._extract_cpp_type_string(node.children[j])
                            if arg_str:
                                arg_parts.append(arg_str)
                            j += 1
                        parts[-1] = parts[-1] + '<' + ','.join(arg_parts) + '>'
                        i = j + 1
                    else:
                        part = self._extract_cpp_type_string(child)
                        if part:
                            parts.append(part)
                        i += 1
                return ' '.join(parts)
            # Fallback: join all children with spaces
            return ' '.join(self._extract_cpp_type_string(child) for child in node.children if self._extract_cpp_type_string(child))
        return str(node)

    def _process_template_alias(self, node: Tree):
        """Process template alias definition."""
        try:
            if self.debug:
                print(f"template_alias children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            # Extract template parameters
            template_params = self._process_template_params(node.children[0])
            # Extract alias name
            alias_name = self._extract_token_value(node.children[1])
            # Extract aliased type (preserve punctuation)
            aliased_type = self._extract_cpp_type_string(node.children[2])
            template_alias = TemplateAliasNode(
                name=alias_name,
                template_params=template_params,
                aliased_type=aliased_type
            )
            self.ast.advanced_cpp_features.append(template_alias)
        except Exception as e:
            if self.debug:
                print(f"Error processing template alias: {e}")
            self.errors.append(f"Template alias error: {e}")

    def _process_template_params(self, node: Tree) -> List[TemplateParamNode]:
        """Process template parameters."""
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "template_param":
                param = self._process_template_param(child)
                params.append(param)
        return params

    def _process_template_param(self, node: Tree) -> TemplateParamNode:
        """Process a single template parameter."""
        try:
            if self.debug:
                print(f"template_param children: {[ (type(c), getattr(c, 'type', getattr(c, 'data', c))) for c in node.children ]}")
            if len(node.children) == 1 and isinstance(node.children[0], Tree):
                child = node.children[0]
                if self.debug:
                    print(f"{child.data} children: {[ (type(cc), getattr(cc, 'type', getattr(cc, 'data', cc))) for cc in child.children ]}")
                if child.data == "typename_param" or child.data == "class_param":
                    if len(child.children) == 1:
                        param_type = "typename" if child.data == "typename_param" else "class"
                        param_name = child.children[0].value
                        return TemplateParamNode(name=param_name, param_type=param_type)
                    elif len(child.children) == 2:
                        param_type = child.children[0].value  # 'typename' or 'class'
                        param_name = child.children[1].value
                        return TemplateParamNode(name=param_name, param_type=param_type)
                    else:
                        if self.debug:
                            print(f"{child.data} has {len(child.children)} children, expected 1 or 2.")
                        return TemplateParamNode(name="error", param_type="error")
            if len(node.children) == 2:
                # typename NAME or class NAME (should not occur with new grammar)
                param_type = self._extract_token_value(node.children[0])
                param_name = self._extract_token_value(node.children[1])
                return TemplateParamNode(name=param_name, param_type=param_type)
            elif len(node.children) == 4:
                # typename NAME = TYPE or class NAME = TYPE
                param_type = self._extract_token_value(node.children[0])
                param_name = self._extract_token_value(node.children[1])
                default_value = self._extract_token_value(node.children[3])
                return TemplateParamNode(name=param_name, param_type=param_type, default_value=default_value)
            else:
                if self.debug:
                    print(f"Unexpected template parameter structure: {len(node.children)} children")
                return TemplateParamNode(name="error", param_type="error")
        except Exception as e:
            if self.debug:
                print(f"Error processing template parameter: {e}")
            return TemplateParamNode(name="error", param_type="error")

    def _process_function_decl(self, node: Tree) -> dict:
        """Process function declaration."""
        try:
            # Extract function name
            func_name = self._extract_token_value(node.children[0])
            
            # Extract parameters
            parameters = []
            if len(node.children) > 1 and isinstance(node.children[1], Tree) and node.children[1].data == "function_param_list":
                parameters = self._process_function_param_list(node.children[1])
            
            # Extract return type (if any)
            return_type = None
            if len(node.children) > 2 and isinstance(node.children[2], Tree) and node.children[2].data == "return_type":
                return_type = self._extract_token_value(node.children[2].children[0])
            
            # Extract function body
            code = ""
            if len(node.children) > 3 and isinstance(node.children[3], Tree) and node.children[3].data == "function_body":
                code = self._extract_code_from_block(node.children[3].children[0])
            
            return {
                'name': func_name,
                'parameters': parameters,
                'return_type': return_type,
                'code': code
            }
            
        except Exception as e:
            if self.debug:
                print(f"Error processing function declaration: {e}")
            self.errors.append(f"Function declaration error: {e}")
            return {'name': 'error', 'parameters': [], 'code': ''}

    def _process_function_decl_alt(self, node: Tree) -> dict:
        """Process alternative function declaration (for attributes)."""
        try:
            if self.debug:
                print(f"function_decl_alt children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            # Extract function name (first child)
            func_name = self._extract_token_value(node.children[0])
            
            # Extract parameter list (third child, after LPAR)
            param_list = self._process_function_param_list(node.children[2])
            
            # Extract return type (fifth child, after RPAR, if it exists)
            return_type = None
            code = ""
            
            # Look for return_type and balanced_braces
            for i, child in enumerate(node.children):
                if isinstance(child, Tree):
                    if child.data == "return_type":
                        return_type = self._extract_cpp_type_string(child.children[0])
                    elif child.data == "balanced_braces":
                        code = self._extract_code_from_balanced_braces(child)
            
            return {
                'name': func_name,
                'parameters': param_list,
                'return_type': return_type,
                'code': code
            }
        except Exception as e:
            if self.debug:
                print(f"Error processing function declaration alt: {e}")
            self.errors.append(f"Function declaration alt error: {e}")
            return {'name': '', 'parameters': [], 'return_type': '', 'code': ''}

    def _process_function_param_list(self, node: Tree) -> List[MethodParamNode]:
        """Process function parameter list."""
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "function_param":
                param = self._process_function_param(child)
                params.append(param)
        return params

    def _process_function_param(self, node: Tree) -> MethodParamNode:
        """Process a single function parameter."""
        try:
            param_name = self._extract_token_value(node.children[0])
            param_type = self._extract_cpp_type_string(node.children[2]) if len(node.children) > 2 else self._extract_cpp_type_string(node.children[1])
            default_value = None
            if len(node.children) > 2 and isinstance(node.children[-1], Tree) and node.children[-1].data == "default_value":
                default_value = self._process_value(node.children[-1].children[0])
            return MethodParamNode(
                param_type=param_type,
                param_name=param_name,
                default_value=default_value
            )
        except Exception as e:
            if self.debug:
                print(f"Error processing function parameter: {e}")
            self.errors.append(f"Function parameter error: {e}")
            return MethodParamNode(param_type="void", param_name="error")

    def _process_static_assert(self, node: Tree):
        """Process static assertion."""
        try:
            if self.debug:
                print(f"static_assert children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            # Extract condition expression
            condition = self._extract_expression_string(node.children[1])
            # Extract message string
            message = self._extract_token_value(node.children[3])
            
            static_assert = StaticAssertNode(
                condition=condition,
                message=message
            )
            
            self.ast.advanced_cpp_features.append(static_assert)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing static assertion: {e}")
            self.errors.append(f"Static assertion error: {e}")

    def _extract_expression_string(self, node: Tree) -> str:
        """Extract a string representation of an expression from the parse tree."""
        if self.debug:
            print(f"[DEBUG] _extract_expression_string: node type={type(node)}, data={getattr(node, 'data', None)}, children={[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in getattr(node, 'children', []) ]}")
        if isinstance(node, Token):
            return node.value
        elif isinstance(node, Tree):
            if node.data == "expr":
                return self._extract_expression_string(node.children[0])
            elif node.data == "comparison_expr":
                if len(node.children) == 1:
                    return self._extract_expression_string(node.children[0])
                else:
                    # Multiple parts with comparison operators
                    result = self._extract_expression_string(node.children[0])
                    for i in range(1, len(node.children), 2):
                        if i + 1 < len(node.children):
                            op_node = node.children[i]
                            if isinstance(op_node, Token):
                                op = op_node.value
                            elif isinstance(op_node, Tree):
                                op = self._extract_expression_string(op_node)
                            else:
                                op = str(op_node)
                            right = self._extract_expression_string(node.children[i + 1])
                            result += f" {op} {right}"
                    return result
            elif node.data == "arithmetic_expr":
                if len(node.children) == 1:
                    return self._extract_expression_string(node.children[0])
                else:
                    # Multiple parts with arithmetic operators
                    result = self._extract_expression_string(node.children[0])
                    for i in range(1, len(node.children), 2):
                        if i + 1 < len(node.children):
                            op = self._extract_token_value(node.children[i])
                            right = self._extract_expression_string(node.children[i + 1])
                            result += f" {op} {right}"
                    return result
            elif node.data == "term":
                if len(node.children) == 1:
                    return self._extract_expression_string(node.children[0])
                else:
                    # Multiple parts with multiplication operators
                    result = self._extract_expression_string(node.children[0])
                    for i in range(1, len(node.children), 2):
                        if i + 1 < len(node.children):
                            op = self._extract_token_value(node.children[i])
                            right = self._extract_expression_string(node.children[i + 1])
                            result += f" {op} {right}"
                    return result
            elif node.data == "factor":
                return self._extract_expression_string(node.children[0])
            elif node.data == "signed_atom":
                if len(node.children) == 1:
                    return self._extract_expression_string(node.children[0])
                else:
                    # Handle negative numbers
                    return "-" + self._extract_expression_string(node.children[1])
            elif node.data == "simple_name":
                return self._extract_token_value(node.children[0])
            elif node.data == "sizeof_expr":
                # Children: LPAR, (cpp_type|expr), RPAR
                if len(node.children) == 3:
                    arg = node.children[1]
                    if isinstance(arg, Tree) and arg.data == "cpp_type":
                        arg_str = self._extract_cpp_type_string(arg)
                    else:
                        arg_str = self._extract_expression_string(arg)
                    return f"sizeof({arg_str})"
                else:
                    return "sizeof()"
            elif node.data == "function_call":
                func_name = self._extract_token_value(node.children[0])
                if len(node.children) > 1:
                    args = []
                    for child in node.children[1:]:
                        if isinstance(child, Tree) and child.data == "expr":
                            args.append(self._extract_expression_string(child))
                    return f"{func_name}({', '.join(args)})"
                else:
                    return f"{func_name}()"
            else:
                # Fallback for other trees
                return " ".join(self._extract_expression_string(child) for child in node.children)
        else:
            return str(node)

    def _process_global_var(self, node: Tree):
        """Process global variable definition."""
        try:
            # Find the actual global var type node
            global_type_node = None
            for child in node.children:
                if isinstance(child, Tree):
                    global_type_node = child
                    break
            
            if global_type_node:
                if global_type_node.data == "global_constexpr":
                    self._process_global_constexpr(global_type_node)
                elif global_type_node.data == "global_device_const":
                    self._process_global_device_const(global_type_node)
                elif global_type_node.data == "global_static_inline":
                    self._process_global_static_inline(global_type_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing global variable: {e}")
            self.errors.append(f"Global variable error: {e}")

    def _process_global_constexpr(self, node: Tree):
        """Process global constexpr variable."""
        try:
            if self.debug:
                print(f"global_constexpr children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            name = self._extract_token_value(node.children[0])
            var_type = self._extract_cpp_type_string(node.children[1])
            value = self._process_value(node.children[2])
            
            global_constexpr = GlobalConstexprNode(
                name=name,
                type=var_type,
                value=value
            )
            
            self.ast.advanced_cpp_features.append(global_constexpr)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing global constexpr: {e}")
            self.errors.append(f"Global constexpr error: {e}")

    def _process_global_device_const(self, node: Tree):
        """Process global device constant."""
        try:
            if self.debug:
                print(f"global_device_const children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            # Extract name (index 0)
            name = self._extract_token_value(node.children[0])
            
            # Extract type (index 1)
            const_type = self._extract_cpp_type_string(node.children[1])
            
            # Extract array size from array_spec (index 2)
            array_size = "1"  # default
            if len(node.children) > 2 and isinstance(node.children[2], Tree) and node.children[2].data == "array_spec":
                array_size_node = node.children[2].children[1]  # The size is the second child of array_spec
                array_size = self._extract_expression_string(array_size_node)
            
            # Extract values (index 3)
            values = []
            if len(node.children) > 3 and isinstance(node.children[3], Tree) and node.children[3].data == "array":
                values = self._process_array(node.children[3])
            
            global_device_const = GlobalDeviceConstNode(
                name=name,
                type=const_type,
                array_size=array_size,
                values=values
            )
            
            self.ast.advanced_cpp_features.append(global_device_const)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing global device const: {e}")
            self.errors.append(f"Global device const error: {e}")

    def _process_global_static_inline(self, node: Tree):
        """Process global static inline function."""
        try:
            func_decl = self._process_function_decl(node.children[0])
            
            global_static_inline = GlobalStaticInlineNode(
                name=func_decl['name'],
                parameters=func_decl['parameters'],
                return_type=func_decl.get('return_type'),
                code=func_decl['code']
            )
            
            self.ast.advanced_cpp_features.append(global_static_inline)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing global static inline: {e}")
            self.errors.append(f"Global static inline error: {e}")

    def _process_operator_overload(self, node: Tree):
        """Process operator overload."""
        try:
            if self.debug:
                print(f"operator_overload children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            # Extract operator name (index 0 after "def")
            operator_name_node = node.children[0]
            if isinstance(operator_name_node, Tree) and operator_name_node.data == "operator_name":
                # The operator_name node has operator_symbol as a direct child
                if len(operator_name_node.children) == 1:
                    operator_symbol_node = operator_name_node.children[0]
                    if isinstance(operator_symbol_node, Tree) and operator_symbol_node.data == "operator_symbol":
                        operator_symbol = self._extract_token_value(operator_symbol_node.children[0])
                    else:
                        operator_symbol = self._extract_token_value(operator_symbol_node)
                    operator_name = f"operator{operator_symbol}"
                else:
                    # operator STRING case
                    operator_name = f"operator{self._extract_token_value(operator_name_node.children[1])}"
            else:
                operator_name = self._extract_token_value(operator_name_node)
            
            # Extract parameters (index 2 after operator_name and LPAR)
            parameters = []
            if len(node.children) > 2 and isinstance(node.children[2], Tree) and node.children[2].data == "operator_params":
                parameters = self._process_operator_params(node.children[2])
            
            # Extract return type (index 4 after parameters and RPAR)
            return_type = None
            if len(node.children) > 4 and isinstance(node.children[4], Tree) and node.children[4].data == "return_type":
                return_type = self._extract_cpp_type_string(node.children[4])
            
            # Extract code (index 5 after return_type)
            code = ""
            if len(node.children) > 5 and isinstance(node.children[5], Tree) and node.children[5].data == "balanced_braces":
                code = self._extract_code_from_balanced_braces(node.children[5])
            
            operator_overload = OperatorOverloadNode(
                operator=operator_name,
                parameters=parameters,
                return_type=return_type,
                code=code
            )
            
            self.ast.advanced_cpp_features.append(operator_overload)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing operator overload: {e}")
            self.errors.append(f"Operator overload error: {e}")

    def _process_operator_params(self, node: Tree) -> List[MethodParamNode]:
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "operator_param":
                param_name = self._extract_token_value(child.children[0])
                param_type = self._extract_cpp_type_string(child.children[2]) if len(child.children) > 2 else self._extract_cpp_type_string(child.children[1])
                params.append(MethodParamNode(param_type=param_type, param_name=param_name))
        return params

    def _process_constructor_def(self, node: Tree):
        """Process constructor definition."""
        try:
            # Find the actual constructor type node
            constructor_type_node = None
            for child in node.children:
                if isinstance(child, Tree):
                    constructor_type_node = child
                    break
            
            if constructor_type_node:
                if constructor_type_node.data == "constructor_decl":
                    self._process_constructor_decl(constructor_type_node)
                elif constructor_type_node.data == "destructor_decl":
                    self._process_destructor_decl(constructor_type_node)
                elif constructor_type_node.data == "member_init":
                    self._process_member_init(constructor_type_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing constructor definition: {e}")
            self.errors.append(f"Constructor definition error: {e}")

    def _process_constructor_decl(self, node: Tree):
        """Process constructor declaration."""
        try:
            if self.debug:
                print(f"constructor_decl children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            # Extract parameters (index 1 after LPAR)
            parameters = []
            if len(node.children) > 1 and isinstance(node.children[1], Tree) and node.children[1].data == "constructor_params":
                parameters = self._process_constructor_params(node.children[1])
            
            # Extract member initializers (index 3 after RPAR)
            member_initializers = []
            if len(node.children) > 3 and isinstance(node.children[3], Tree) and node.children[3].data == "member_init_list":
                member_initializers = self._process_member_init_list(node.children[3])
            
            # Extract constructor body (index 4 after member initializers)
            code = ""
            if len(node.children) > 4 and isinstance(node.children[4], Tree) and node.children[4].data == "balanced_braces":
                code = self._extract_code_from_balanced_braces(node.children[4])
            
            constructor = ConstructorNode(
                parameters=parameters,
                member_initializers=member_initializers,
                code=code
            )
            
            self.ast.advanced_cpp_features.append(constructor)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing constructor declaration: {e}")
            self.errors.append(f"Constructor declaration error: {e}")

    def _process_constructor_params(self, node: Tree) -> List[MethodParamNode]:
        """Process constructor parameters."""
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "constructor_param":
                param = self._process_function_param(child)
                params.append(param)
        return params

    def _process_member_init_list(self, node: Tree) -> List[tuple[str, str]]:
        """Process member initializer list."""
        initializers = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "member_init_item":
                member = self._extract_token_value(child.children[0])
                # The value is at index 2 (after LPAR)
                value = self._extract_token_value(child.children[2])
                initializers.append((member, value))
        return initializers

    def _process_destructor_decl(self, node: Tree):
        """Process destructor declaration."""
        try:
            if self.debug:
                print(f"destructor_decl children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            code = ""
            if len(node.children) > 0 and isinstance(node.children[0], Tree) and node.children[0].data == "destructor_body":
                code = self._extract_code_from_block(node.children[0])
            
            destructor = DestructorNode(code=code)
            self.ast.advanced_cpp_features.append(destructor)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing destructor declaration: {e}")
            self.errors.append(f"Destructor declaration error: {e}")

    def _process_member_init(self, node: Tree):
        """Process member initialization."""
        try:
            name = self._extract_token_value(node.children[1])
            member_init_list = self._process_member_init_list(node.children[3])
            
            member_init = ConstructorNode(
                name=name,
                parameters=[],
                member_initializers=member_init_list,
                code=""
            )
            
            self.ast.advanced_cpp_features.append(member_init)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing member initialization: {e}")
            self.errors.append(f"Member initialization error: {e}")



    def _process_preprocessor_directive(self, node: Tree):
        """Process preprocessor directive."""
        try:
            # Find the actual directive type node
            directive_type_node = None
            for child in node.children:
                if isinstance(child, Tree):
                    directive_type_node = child
                    break
            
            if directive_type_node:
                directive_type = directive_type_node.data
                content = " ".join([self._extract_token_value(child) for child in directive_type_node.children])
                
                preprocessor_directive = PreprocessorDirectiveNode(
                    directive_type=directive_type,
                    content=content
                )
                
                self.ast.advanced_cpp_features.append(preprocessor_directive)
                
        except Exception as e:
            if self.debug:
                print(f"Error processing preprocessor directive: {e}")
            self.errors.append(f"Preprocessor directive error: {e}")

    def _process_function_attribute(self, node: Tree):
        """Process function with attributes."""
        try:
            if self.debug:
                print(f"function_attribute children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            attributes = []
            # Find all attribute nodes
            for child in node.children:
                if isinstance(child, Tree) and child.data == "attribute":
                    if self.debug:
                        print(f"attribute children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in child.children ]}")
                    # Extract attribute name from the attribute node
                    attr_child = child.children[0]
                    if isinstance(attr_child, Token):
                        attr_name = attr_child.value
                    elif isinstance(attr_child, Tree):
                        if len(attr_child.children) == 1 and isinstance(attr_child.children[0], Token):
                            attr_name = attr_child.children[0].value
                        elif len(attr_child.children) == 0:
                            attr_name = attr_child.data  # Use the rule name as the attribute name
                        else:
                            attr_name = str(attr_child)
                    else:
                        attr_name = str(attr_child)
                    attributes.append(attr_name)
            
            # Find the function declaration (last child)
            func_decl_node = None
            for child in node.children:
                if isinstance(child, Tree) and (child.data == "function_decl" or child.data == "function_decl_alt"):
                    func_decl_node = child
                    break
            
            if func_decl_node:
                if func_decl_node.data == "function_decl_alt":
                    func_decl = self._process_function_decl_alt(func_decl_node)
                else:
                    func_decl = self._process_function_decl(func_decl_node)
                
                function_attribute = FunctionAttributeNode(
                    attributes=attributes,
                    name=func_decl['name'],
                    parameters=func_decl['parameters'],
                    return_type=func_decl.get('return_type'),
                    code=func_decl['code']
                )
                
                self.ast.advanced_cpp_features.append(function_attribute)
            else:
                if self.debug:
                    print("No function_decl found in function_attribute")
                
        except Exception as e:
            if self.debug:
                print(f"Error processing function attribute: {e}")
            self.errors.append(f"Function attribute error: {e}")

    def _process_concept_def(self, node: Tree):
        """Process concept definition."""
        try:
            name = self._extract_token_value(node.children[0])
            requires = self._process_concept_requires(node.children[1])
            
            concept = ConceptNode(name=name, requires=requires)
            self.ast.advanced_cpp_features.append(concept)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing concept definition: {e}")
            self.errors.append(f"Concept definition error: {e}")

    def _process_concept_requires(self, node: Tree) -> ConceptRequiresNode:
        """Process concept requires clause."""
        try:
            type_param = self._extract_token_value(node.children[1])
            requirements = []
            
            if len(node.children) > 2 and isinstance(node.children[2], Tree) and node.children[2].data == "requires_expr":
                for child in node.children[2].children:
                    if isinstance(child, Tree) and child.data == "requires_clause":
                        req = " ".join([self._extract_token_value(c) for c in child.children])
                        requirements.append(req)
            
            return ConceptRequiresNode(type_param=type_param, requirements=requirements)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing concept requires: {e}")
            self.errors.append(f"Concept requires error: {e}")
            return ConceptRequiresNode(type_param="T", requirements=[])

    def _process_friend_declaration(self, node: Tree):
        """Process friend declaration."""
        try:
            friend_target = self._process_friend_target(node.children[0])
            
            friend_declaration = FriendDeclarationNode(
                friend_type=friend_target['type'],
                target=friend_target['target']
            )
            
            self.ast.advanced_cpp_features.append(friend_declaration)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing friend declaration: {e}")
            self.errors.append(f"Friend declaration error: {e}")

    def _process_friend_target(self, node: Tree) -> dict:
        """Process friend target."""
        try:
            if node.data == "class_friend":
                return {
                    'type': 'class',
                    'target': self._extract_token_value(node.children[0])
                }
            elif node.data == "function_friend":
                return {
                    'type': 'function',
                    'target': self._extract_token_value(node.children[0])
                }
            else:
                raise ValueError(f"Unexpected friend target structure: {node.data}")
        except Exception as e:
            if self.debug:
                print(f"Error processing friend target: {e}")
            self.errors.append(f"Friend target error: {e}")
            return {'type': 'class', 'target': 'error'}

    def _process_user_defined_literal(self, node: Tree):
        """Process user-defined literal."""
        try:
            if self.debug:
                print(f"user_defined_literal children: {[ (type(c), getattr(c, 'data', c) if isinstance(c, Tree) else c) for c in node.children ]}")
            
            # Extract literal suffix (index 1 after "def" "operator")
            literal_suffix = self._extract_token_value(node.children[1])
            
            # Find the return_type node
            return_type = ""
            for child in node.children:
                if isinstance(child, Tree) and child.data == "return_type":
                    if len(child.children) > 0:
                        return_type = self._extract_cpp_type_string(child.children[0])
                    break
            
            # Extract code (index 4 after return type)
            code = ""
            for child in node.children:
                if isinstance(child, Tree) and child.data == "balanced_braces":
                    code = self._extract_code_from_balanced_braces(child)
                    break
            
            user_defined_literal = UserDefinedLiteralNode(
                literal_suffix=literal_suffix,
                return_type=return_type,
                code=code
            )
            
            self.ast.advanced_cpp_features.append(user_defined_literal)
            
        except Exception as e:
            if self.debug:
                print(f"Error processing user-defined literal: {e}")
            self.errors.append(f"User-defined literal error: {e}")

    def _process_array(self, node: Tree) -> List[ValueNode]:
        """Process array values."""
        values = []
        if isinstance(node, Tree) and node.data == "array":
            # Array structure: LSQB value_list RSQB
            # value_list is: (value (COMMA value)*)?
            if len(node.children) >= 2:  # Should have LSQB, value_list, RSQB
                value_list_node = node.children[1]  # value_list is at index 1
                if isinstance(value_list_node, Tree) and value_list_node.data == "value_list":
                    for child in value_list_node.children:
                        if isinstance(child, Tree) and child.data == "value":
                            value = self._process_value(child)
                            values.append(value)
                        # Skip COMMA tokens
        return values

    def _topic_path_to_str(self, node):
        """Convert a topic_path tree to a string like '/foo/bar'."""
        if isinstance(node, str):
            return node
        if hasattr(node, 'children'):
            # Only join NAME tokens, skip SLASH tokens
            parts = [child.value for child in node.children if hasattr(child, 'type') and child.type == 'NAME']
            return '/' + '/'.join(parts)
        return str(node)

    def _process_cuda_kernel_def(self, node):
        """Process standalone cuda_kernel definition and add to ast.cuda_kernels."""
        from ..core.ast import KernelNode, KernelContentNode, CudaKernelsNode
        
        if self.debug:
            print(f"cuda_kernel_def children: {[type(c) for c in node.children]}")
            print(f"cuda_kernel_def children values: {[str(c) for c in node.children]}")
        
        # The structure is: NAME LBRACE cuda_kernel_content RBRACE
        # So children[0] should be the NAME token
        if len(node.children) >= 1:
            name_token = node.children[0]
            if hasattr(name_token, 'value'):
                name = name_token.value
            else:
                name = str(name_token)
        else:
            name = "unknown_kernel"
            if self.debug:
                print(f"Warning: cuda_kernel_def has unexpected structure: {len(node.children)} children")
        
        if self.debug:
            print(f"Extracted kernel name: '{name}'")
        
        # cuda_kernel_content is at children[2]
        if len(node.children) >= 3:
            content = self._process_cuda_kernel_content(node.children[2])
        else:
            content = KernelContentNode()
            if self.debug:
                print(f"Warning: No kernel content found")
        
        kernel = KernelNode(name=name, content=content)
        # Add to ast.cuda_kernels
        if not hasattr(self.ast, 'cuda_kernels') or self.ast.cuda_kernels is None:
            self.ast.cuda_kernels = CudaKernelsNode(kernels=[])
        self.ast.cuda_kernels.kernels.append(kernel)
        return kernel

    def _process_standalone_kernel_def(self, node):
        """Process standalone kernel definition and add to ast.cuda_kernels."""
        from ..core.ast import KernelNode, KernelContentNode, CudaKernelsNode
        
        if self.debug:
            print(f"kernel_def children: {[type(c) for c in node.children]}")
            print(f"kernel_def children values: {[str(c) for c in node.children]}")
        
        # The structure is: KERNEL NAME LBRACE kernel_content RBRACE
        # So children[1] should be the NAME token (children[0] is KERNEL)
        if len(node.children) >= 2:
            name_token = node.children[1]
            if hasattr(name_token, 'value'):
                name = name_token.value
            else:
                name = str(name_token)
        else:
            name = "unknown_kernel"
            if self.debug:
                print(f"Warning: kernel_def has unexpected structure: {len(node.children)} children")
        
        if self.debug:
            print(f"Extracted kernel name: '{name}'")
        
        # kernel_content is at children[3]
        if len(node.children) >= 4:
            content = self._process_kernel_content(node.children[3])
        else:
            content = KernelContentNode()
            if self.debug:
                print(f"Warning: No kernel content found")
        
        kernel = KernelNode(name=name, content=content)
        # Add to ast.cuda_kernels
        if not hasattr(self.ast, 'cuda_kernels') or self.ast.cuda_kernels is None:
            self.ast.cuda_kernels = CudaKernelsNode(kernels=[])
        self.ast.cuda_kernels.kernels.append(kernel)
        return kernel

    def _process_package_def(self, node: Tree):
        """Process package definition."""
        from ..core.ast import PackageNode
        
        if self.debug:
            print(f"Processing package_def: {node.children}")
        
        # Extract package name (first child should be NAME token)
        package_name = self._extract_token_value(node.children[0])
        
        # Initialize package fields
        version = None
        description = None
        dependencies = []
        build_configuration = {}
        cpp_nodes = []
        
        # Initialize all the new package fields
        build_type = None
        cross_compilation = None
        target_platform = None
        toolchain_file = None
        package_management = None
        dependency_resolution = None
        version_constraints = {}
        maintainer = None
        build_optimization = None
        dependency_management = None
        custom_targets = []
        compiler = None
        platform = None
        build_variant = None
        build_performance = None
        parallel_build = None
        build_jobs = None
        build_cache = None
        ccache = None
        ninja = None
        precompiled_headers = None
        unity_build = None
        link_time_optimization = None
        profile_guided_optimization = None
        install_configuration = {}
        test_configuration = {}
        performance_configuration = {}
        build_flags = {}
        architecture = None
        dependency_caching = None
        variant_configuration = {}
        license = None
        url = None
        cross_platform = None
        machine = None
        platform_configuration = {}
        architecture_configuration = {}
        compiler_version = None
        optimization_flags = {}
        custom_target_configuration = {}
        dependency_configuration = {}
        optional_dependencies = []
        system_dependencies = []
        dependency_parallel_download = None
        dependency_verification = None
        dependency_licenses = []
        
        # Process package content (children[2] should be package_content)
        if len(node.children) >= 3:
            content_node = node.children[2]
            if isinstance(content_node, Tree) and content_node.data == "package_content":
                for child in content_node.children:
                    if isinstance(child, Tree):
                        if self.debug:
                            print(f"Processing package child: {child.data}")
                        if child.data == "package_name":
                            # package_name: "name" ":" STRING
                            if len(child.children) >= 2:
                                package_name = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_version":
                            # package_version: "version" ":" STRING
                            if len(child.children) >= 2:
                                version = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_description":
                            # package_description: "description" ":" STRING
                            if len(child.children) >= 2:
                                description = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_dependencies":
                            # package_dependencies: "dependencies" ":" array
                            if len(child.children) >= 1:
                                deps_node = child.children[0]  # Array node is at index 0
                                if isinstance(deps_node, Tree) and deps_node.data == "array":
                                    deps = self._process_array(deps_node)
                                    dependencies = [dep.value for dep in deps]
                        elif child.data == "package_build_config":
                            # package_build_config: "build_configuration" ":" nested_dict
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    build_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_cpp_nodes":
                            # package_cpp_nodes: cpp_node_def
                            cpp_node = self._process_cpp_node_def(child)
                            if cpp_node:
                                cpp_nodes.append(cpp_node)
                        # Handle all the new package fields
                        elif child.data == "package_build_type":
                            if len(child.children) >= 2:
                                build_type = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_cross_compilation":
                            if len(child.children) >= 2:
                                cross_compilation = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_target_platform":
                            if len(child.children) >= 2:
                                target_platform = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_toolchain_file":
                            if len(child.children) >= 2:
                                toolchain_file = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_management":
                            if len(child.children) >= 2:
                                value = self._extract_token_value(child.children[1])
                                if value == "true":
                                    package_management = True
                                elif value == "false":
                                    package_management = False
                                else:
                                    package_management = value.strip('"')
                        elif child.data == "package_dependency_resolution":
                            if len(child.children) >= 2:
                                value = self._extract_token_value(child.children[1])
                                if value == "true":
                                    dependency_resolution = True
                                elif value == "false":
                                    dependency_resolution = False
                                else:
                                    dependency_resolution = value.strip('"')
                        elif child.data == "package_version_constraints":
                            if len(child.children) >= 2:
                                constraints_node = child.children[1]
                                if isinstance(constraints_node, Tree) and constraints_node.data == "nested_dict":
                                    version_constraints = self._process_nested_dict(constraints_node)
                        elif child.data == "package_maintainer":
                            if len(child.children) >= 2:
                                maintainer = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_build_optimization":
                            if len(child.children) >= 2:
                                build_optimization = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_dependency_management":
                            if len(child.children) >= 2:
                                value = self._extract_token_value(child.children[1])
                                if value == "true":
                                    dependency_management = True
                                elif value == "false":
                                    dependency_management = False
                                else:
                                    dependency_management = value.strip('"')
                        elif child.data == "package_custom_targets":
                            if len(child.children) >= 1:
                                targets_node = child.children[0]  # Array node is at index 0
                                if isinstance(targets_node, Tree) and targets_node.data == "array":
                                    targets = self._process_array(targets_node)
                                    custom_targets = [target.value for target in targets]
                        elif child.data == "package_compiler":
                            if len(child.children) >= 2:
                                compiler = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_platform":
                            if len(child.children) >= 2:
                                platform = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_build_variant":
                            if len(child.children) >= 2:
                                build_variant = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_build_performance":
                            if len(child.children) >= 2:
                                build_performance = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_parallel_build":
                            if len(child.children) >= 2:
                                parallel_build = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_build_jobs":
                            if len(child.children) >= 2:
                                build_jobs = int(self._extract_token_value(child.children[1]))
                        elif child.data == "package_build_cache":
                            if len(child.children) >= 2:
                                build_cache = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_ccache":
                            if len(child.children) >= 2:
                                ccache = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_ninja":
                            if len(child.children) >= 2:
                                ninja = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_precompiled_headers":
                            if len(child.children) >= 2:
                                precompiled_headers = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_unity_build":
                            if len(child.children) >= 2:
                                unity_build = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_link_time_optimization":
                            if len(child.children) >= 2:
                                link_time_optimization = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_profile_guided_optimization":
                            if len(child.children) >= 2:
                                profile_guided_optimization = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_install_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    install_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_test_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    test_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_performance_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    performance_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_build_flags":
                            if len(child.children) >= 2:
                                flags_node = child.children[1]
                                if isinstance(flags_node, Tree) and flags_node.data == "nested_dict":
                                    build_flags = self._process_nested_dict(flags_node)
                        elif child.data == "package_architecture":
                            if len(child.children) >= 2:
                                architecture = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_dependency_caching":
                            if len(child.children) >= 2:
                                dependency_caching = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_variant_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    variant_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_license":
                            if len(child.children) >= 2:
                                license = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_url":
                            if len(child.children) >= 2:
                                url = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_cross_platform":
                            if len(child.children) >= 2:
                                cross_platform = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_machine":
                            if len(child.children) >= 2:
                                machine = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_platform_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    platform_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_architecture_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    architecture_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_compiler_version":
                            if len(child.children) >= 2:
                                compiler_version = self._extract_token_value(child.children[1]).strip('"')
                        elif child.data == "package_optimization_flags":
                            if len(child.children) >= 2:
                                flags_node = child.children[1]
                                if isinstance(flags_node, Tree) and flags_node.data == "nested_dict":
                                    optimization_flags = self._process_nested_dict(flags_node)
                        elif child.data == "package_custom_target_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    custom_target_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_dependency_config":
                            if len(child.children) >= 2:
                                config_node = child.children[1]
                                if isinstance(config_node, Tree) and config_node.data == "nested_dict":
                                    dependency_configuration = self._process_nested_dict(config_node)
                        elif child.data == "package_optional_dependencies":
                            if len(child.children) >= 1:
                                deps_node = child.children[0]  # Array node is at index 0
                                if isinstance(deps_node, Tree) and deps_node.data == "array":
                                    deps = self._process_array(deps_node)
                                    optional_dependencies = [dep.value for dep in deps]
                        elif child.data == "package_system_dependencies":
                            if len(child.children) >= 1:
                                deps_node = child.children[0]  # Array node is at index 0
                                if isinstance(deps_node, Tree) and deps_node.data == "array":
                                    deps = self._process_array(deps_node)
                                    system_dependencies = [dep.value for dep in deps]
                        elif child.data == "package_dependency_parallel_download":
                            if len(child.children) >= 2:
                                dependency_parallel_download = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_dependency_verification":
                            if len(child.children) >= 2:
                                dependency_verification = self._extract_token_value(child.children[1]) == "true"
                        elif child.data == "package_dependency_licenses":
                            if len(child.children) >= 1:
                                licenses_node = child.children[0]  # Array node is at index 0
                                if isinstance(licenses_node, Tree) and licenses_node.data == "array":
                                    licenses = self._process_array(licenses_node)
                                    dependency_licenses = [license.value for license in licenses]
        
        # Create package node
        package = PackageNode(
            name=package_name,
            version=version,
            description=description,
            dependencies=dependencies,
            build_configuration=build_configuration,
            cpp_nodes=cpp_nodes,
            build_type=build_type,
            cross_compilation=cross_compilation,
            target_platform=target_platform,
            toolchain_file=toolchain_file,
            package_management=package_management,
            dependency_resolution=dependency_resolution,
            version_constraints=version_constraints,
            maintainer=maintainer,
            build_optimization=build_optimization,
            dependency_management=dependency_management,
            custom_targets=custom_targets,
            compiler=compiler,
            platform=platform,
            build_variant=build_variant,
            build_performance=build_performance,
            parallel_build=parallel_build,
            build_jobs=build_jobs,
            build_cache=build_cache,
            ccache=ccache,
            ninja=ninja,
            precompiled_headers=precompiled_headers,
            unity_build=unity_build,
            link_time_optimization=link_time_optimization,
            profile_guided_optimization=profile_guided_optimization,
            install_configuration=install_configuration,
            test_configuration=test_configuration,
            performance_configuration=performance_configuration,
            build_flags=build_flags,
            architecture=architecture,
            dependency_caching=dependency_caching,
            variant_configuration=variant_configuration,
            license=license,
            url=url,
            cross_platform=cross_platform,
            machine=machine,
            platform_configuration=platform_configuration,
            architecture_configuration=architecture_configuration,
            compiler_version=compiler_version,
            optimization_flags=optimization_flags,
            custom_target_configuration=custom_target_configuration,
            dependency_configuration=dependency_configuration,
            optional_dependencies=optional_dependencies,
            system_dependencies=system_dependencies,
            dependency_parallel_download=dependency_parallel_download,
            dependency_verification=dependency_verification,
            dependency_licenses=dependency_licenses
        )
        
        # Add to AST
        self.ast.packages.append(package)
        
        return package

    def _process_cpp_node_def(self, node: Tree):
        """Process cpp_node definition."""
        from ..core.ast import NodeNode, NodeContentNode
        
        if self.debug:
            print(f"Processing cpp_node_def: {node.children}")
        
        try:
            # Handle the case where cpp_node_def is nested within package_cpp_nodes
            if len(node.children) == 1 and isinstance(node.children[0], Tree) and node.children[0].data == "cpp_node_def":
                # This is the case when called from package_cpp_nodes
                actual_node = node.children[0]
            else:
                # This is the case when called directly
                actual_node = node
            
            if self.debug:
                print(f"actual_node children: {[type(c) for c in actual_node.children]}")
                print(f"actual_node children values: {[str(c) for c in actual_node.children]}")
            
            # Extract node name (first child should be NODE_NAME token)
            node_name = self._extract_token_value(actual_node.children[0])
            
            # Validate node name - subnodes (with dots) are not allowed in robodsl code
            if '.' in node_name:
                error_msg = (
                    f"Invalid node name '{node_name}': Subnodes with dots (.) are not allowed in RoboDSL code. "
                    f"Subnodes are a CLI-only feature for organizing files. "
                    f"Use a simple node name without dots, or create subnodes using the CLI command: "
                    f"'robodsl create-node {node_name}'"
                )
                self.errors.append(error_msg)
                raise ValueError(error_msg)
            
            # Set context to indicate we're processing inside a node
            self.in_node_context = True
            if self.debug:
                print(f"node_content child type: {type(actual_node.children[2])}")
                print(f"node_content child: {actual_node.children[2]}")
            content = self._process_node_content(actual_node.children[2])  # content is at index 2, LBRACE is at index 1
            # Reset context
            self.in_node_context = False
            node_node = NodeNode(name=node_name, content=content)
            self.ast.nodes.append(node_node)
            return node_node
        except Exception as e:
            # Reset context on error
            self.in_node_context = False
            if self.debug:
                print(f"Error processing cpp_node definition: {e}")
            if str(e) not in self.errors:  # Avoid duplicate error messages
                self.errors.append(f"cpp_node definition error: {e}")
            return None

    def _process_nested_dict(self, node: Tree) -> dict:
        """Process nested dictionary."""
        result = {}
        if isinstance(node, Tree) and node.data == "nested_dict":
            for child in node.children:
                if isinstance(child, Tree) and child.data == "dict_list":
                    for dict_item in child.children:
                        if isinstance(dict_item, Tree) and dict_item.data == "dict_item":
                            if len(dict_item.children) >= 2:
                                key = self._extract_token_value(dict_item.children[0])
                                value = self._process_value(dict_item.children[1])
                                result[key] = value.value
        return result

    # In the main parse logic, after processing a cuda_kernel_def, call this method.