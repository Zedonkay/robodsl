"""AST Builder for RoboDSL.

This module builds the AST from the Lark parse tree.
"""
from typing import Any, List, Optional, Union
from lark import Tree, Token
from ..core.ast import (
    RoboDSLAST, IncludeNode, StructNode, ClassNode, PyClassNode, EnumNode,
    TypedefNode, UsingNode, NodeNode, CudaKernelsNode, OnnxModelNode,
    PipelineNode, MessageNode, ServiceNode, CustomActionNode,
    DynamicParameterNode, DynamicRemapNode, SimulationConfigNode,
    HardwareInLoopNode, StructContentNode, ClassContentNode, PyClassContentNode,
    AccessSectionNode, StructMemberNode, CppMethodNode, MethodParamNode,
    PyClassAttributeNode, PyClassMethodNode, PyClassConstructorNode,
    PyClassAccessSectionNode, MessageContentNode, MessageFieldNode,
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
    DeviceNode, OptimizationNode, SimulationWorldNode, SimulationRobotNode,
    SimulationPluginNode, KernelParameterDirection, RawCppCodeNode
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
                if isinstance(child, Tree):
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
                elif definition_node.data == "pyclass_def":
                    self._process_pyclass_def(definition_node)
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
        if isinstance(token_or_tree, Token):
            return token_or_tree.value
        elif isinstance(token_or_tree, Tree):
            # Special handling for cpp_type and cpp_type_name
            if token_or_tree.data == "cpp_type":
                # cpp_type may have children like cpp_type_name, pointer, etc.
                return "".join(self._extract_token_value(child) for child in token_or_tree.children)
            elif token_or_tree.data == "cpp_type_name":
                return "".join(self._extract_token_value(child) for child in token_or_tree.children)
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
            name = str(node.children[0])
            content = self._process_struct_content(node.children[2])
            struct_node = StructNode(name=name, content=content)
            self.ast.data_structures.append(struct_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing struct definition: {e}")
            self.errors.append(f"Struct definition error: {e}")
    
    def _process_struct_content(self, node: Tree) -> StructContentNode:
        """Process struct content."""
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
            type_name = str(node.children[0])
            name = str(node.children[1])
            array_spec = None
            if len(node.children) > 2 and node.children[2].data == "array_spec":
                array_spec = str(node.children[2])
            return StructMemberNode(type=type_name, name=name, array_spec=array_spec)
        except Exception as e:
            if self.debug:
                print(f"Error processing struct member: {e}")
            self.errors.append(f"Struct member error: {e}")
    
    def _process_class_def(self, node: Tree):
        """Process class definition."""
        try:
            name = str(node.children[0])
            inheritance = None
            content_node = None
            
            # Find inheritance and class_content among the children
            for child in node.children[1:]:
                if isinstance(child, Tree):
                    if child.data == "inheritance":
                        inheritance = self._process_inheritance(child)
                    elif child.data == "class_content":
                        content_node = self._process_class_content(child)
            
            if content_node is None:
                content_node = ClassContentNode()
            
            class_node = ClassNode(name=name, content=content_node, inheritance=inheritance)
            self.ast.data_structures.append(class_node)
        except Exception as e:
            if self.debug:
                print(f"Error processing class definition: {e}")
            self.errors.append(f"Class definition error: {e}")
    
    def _process_class_content(self, node: Tree) -> ClassContentNode:
        """Process class content."""
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
                        class_name = self._extract_token_value(child)
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
            print(f"[DEBUG] _process_cpp_method: method name: {name}")
            # Process method_content child
            if len(node.children) > 1 and isinstance(node.children[1], Tree) and node.children[1].data == "method_content":
                method_content = node.children[1]
                print(f"[DEBUG] Processing method_content with {len(method_content.children)} children")
                for child in method_content.children:
                    if isinstance(child, Tree):
                        print(f"[DEBUG] method_content child: {child}, data: {child.data}")
                        if child.data == "input_param":
                            method.inputs.append(self._process_method_param(child))
                        elif child.data == "output_param":
                            method.outputs.append(self._process_method_param(child))
                        elif child.data == "code_block":
                            print(f"[DEBUG] Calling _extract_code_from_block for code_block: {child}")
                            method.code = self._extract_code_from_block(child)
            else:
                # Legacy format - direct children
                for child in node.children[1:]:
                    if isinstance(child, Tree):
                        print(f"[DEBUG] legacy child: {child}, data: {child.data}")
                        if child.data == "method_content":
                            print(f"[DEBUG] Recursively processing legacy method_content child")
                            for subchild in child.children:
                                if isinstance(subchild, Tree):
                                    print(f"[DEBUG] legacy method_content subchild: {subchild}, data: {subchild.data}")
                                    if subchild.data == "input_param":
                                        method.inputs.append(self._process_method_param(subchild))
                                    elif subchild.data == "output_param":
                                        method.outputs.append(self._process_method_param(subchild))
                                    elif subchild.data == "code_block":
                                        print(f"[DEBUG] Calling _extract_code_from_block for code_block: {subchild}")
                                        method.code = self._extract_code_from_block(subchild)
                        elif child.data == "input_param":
                            method.inputs.append(self._process_method_param(child))
                        elif child.data == "output_param":
                            method.outputs.append(self._process_method_param(child))
                        elif child.data == "code_block":
                            print(f"[DEBUG] Calling _extract_code_from_block for code_block: {child}")
                            method.code = self._extract_code_from_block(child)
            return method
        except Exception as e:
            print(f"Error processing C++ method: {e}")
            self.errors.append(f"C++ method error: {e}")
            return CppMethodNode(name="")
    
    def _extract_code_from_block(self, code_block_node: Tree) -> str:
        """Extract code from a code block node, handling nested balanced braces."""
        if self.debug:
            print("[DEBUG] _extract_code_from_block called")
        
        def collect_code(node):
            code_parts = []
            if self.debug:
                print(f"[DEBUG] collect_code visiting node: {node}, type: {type(node)}")
            if isinstance(node, Tree):
                if node.data == "balanced_braces":
                    code_parts.append("{")
                    for child in node.children:
                        code_parts.append(collect_code(child))
                    code_parts.append("}")
                elif node.data == "balanced_content":
                    for child in node.children:
                        code_parts.append(collect_code(child))
                else:
                    for child in node.children:
                        code_parts.append(collect_code(child))
            elif isinstance(node, Token):
                code_parts.append(node.value)
            return ''.join(code_parts)

        try:
            if self.debug:
                print(f"[DEBUG] code_block_node: {code_block_node}")
                print(f"[DEBUG] code_block_node type: {type(code_block_node)}")
                print(f"[DEBUG] code_block_node children: {code_block_node.children}")
            
            # Process all children of the code_block_node, not just the first one
            code_parts = []
            for child in code_block_node.children:
                if isinstance(child, Tree) and child.data in ("balanced_braces", "balanced_content"):
                    code_parts.append(collect_code(child))
                else:
                    code_parts.append(collect_code(child))
            
            code_str = ''.join(code_parts)
            if self.debug:
                print(f"[DEBUG] Extracted code: '{code_str}'")
            return code_str
        except Exception as e:
            if self.debug:
                print(f"Error extracting code from block: {e}")
            return ""
    
    def _process_method_param(self, node: Tree):
        """Process method parameter."""
        try:
            # Extract type name from cpp_type subtree
            def extract_cpp_type(type_node):
                # cpp_type -> cpp_type_name (Token) or pointer/array
                if hasattr(type_node, 'children') and len(type_node.children) > 0:
                    # Recursively extract from first child
                    return extract_cpp_type(type_node.children[0])
                elif hasattr(type_node, 'value'):
                    return type_node.value
                else:
                    return str(type_node)

            param_type = extract_cpp_type(node.children[0])
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
            name = str(node.children[0])
            
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
            content = self._process_node_content(node.children[2])
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
            for child in node.children:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"  Processing child: {child.data}")
                        print(f"    Child type: {type(child)}")
                        print(f"    Child children: {len(child.children)}")
                    if child.data == "parameter":
                        content.parameters.append(self._process_parameter(child))
                    elif child.data == "lifecycle":
                        content.lifecycle = self._process_lifecycle(child)
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
        param_type = str(node.children[0])
        name = str(node.children[1])
        value = self._process_value(node.children[2])
        return ParameterNode(type=param_type, name=name, value=value)

    def _process_lifecycle(self, node: Tree) -> LifecycleNode:
        """Process lifecycle configuration."""
        settings = []
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
            if len(period_node.children) > 0:
                atom = period_node.children[0]
                if isinstance(atom, Tree) and atom.data == "signed_atom":
                    if len(atom.children) > 0:
                        value = atom.children[0]
                        if hasattr(value, 'type') and value.type == 'SIGNED_NUMBER':
                            try:
                                if '.' in value.value:
                                    period = float(value.value)
                                else:
                                    period = int(value.value)
                            except ValueError:
                                period = value.value
                        else:
                            period = str(value)
                    else:
                        period = str(atom)
                else:
                    period = str(atom)
            else:
                period = str(period_node)
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
                name = str(child.children[0])
                value_node = child.children[1]
                if hasattr(value_node, 'type'):
                    if value_node.type == 'BOOLEAN':
                        value = value_node.value == 'true'
                    elif value_node.type == 'NAME':
                        # Handle boolean names like 'true', 'false'
                        value = value_node.value == 'true'
                    else:
                        value = str(value_node)
                else:
                    value = str(value_node)
                settings[name] = value
        return settings

    def _process_remap(self, node: Tree) -> RemapNode:
        """Process remap."""
        if len(node.children) == 2:
            from_topic = str(node.children[0])
            to_topic = str(node.children[1])
        else:
            # Handle "from: topic to: topic" format
            from_topic = str(node.children[1])
            to_topic = str(node.children[3])
        return RemapNode(from_topic=from_topic, to_topic=to_topic)

    def _process_namespace(self, node: Tree) -> NamespaceNode:
        """Process namespace."""
        namespace = str(node.children[0])
        return NamespaceNode(namespace=namespace)

    def _process_publisher(self, node: Tree) -> PublisherNode:
        """Process publisher."""
        topic_node = node.children[0]
        if isinstance(topic_node, Tree):
            # Handle topic_path structure
            topic_parts = []
            for child in topic_node.children:
                if hasattr(child, 'value'):
                    topic_parts.append(child.value)
                else:
                    topic_parts.append(str(child))
            topic = "/" + "/".join(topic_parts)
        else:
            topic = str(topic_node)
        
        msg_type_node = node.children[1]
        if isinstance(msg_type_node, Tree):
            # Handle topic_type structure
            if msg_type_node.data == "topic_type":
                msg_type = str(msg_type_node.children[0]).strip('"')
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
        if isinstance(topic_node, Tree):
            # Handle topic_path structure
            topic_parts = []
            for child in topic_node.children:
                if hasattr(child, 'value'):
                    topic_parts.append(child.value)
                else:
                    topic_parts.append(str(child))
            topic = "/" + "/".join(topic_parts)
        else:
            topic = str(topic_node)
        
        msg_type_node = node.children[1]
        if isinstance(msg_type_node, Tree):
            # Handle topic_type structure
            if msg_type_node.data == "topic_type":
                msg_type = str(msg_type_node.children[0]).strip('"')
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
        
        srv_type_node = node.children[1]
        if isinstance(srv_type_node, Tree):
            # Handle topic_type structure
            if srv_type_node.data == "topic_type":
                srv_type = str(srv_type_node.children[0]).strip('"')
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
        # Extract topic path properly
        topic_path_node = node.children[0]
        if isinstance(topic_path_node, Tree):
            # topic_path is a Tree with children, extract the path
            name = "/" + "/".join(str(child) for child in topic_path_node.children)
        else:
            name = str(topic_path_node)
        
        # Extract action type properly
        action_type_node = node.children[1]
        if isinstance(action_type_node, Tree):
            # topic_type is a Tree, extract the type string
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
            name = str(node.children[0])
            content = self._process_kernel_content(node.children[2])
            return KernelNode(name=name, content=content)
        except Exception as e:
            if self.debug:
                print(f"Error processing kernel definition: {e}")
            self.errors.append(f"Kernel definition error: {e}")
            return KernelNode(name="", content=KernelContentNode())

    def _process_kernel_content(self, node: Tree) -> KernelContentNode:
        """Process kernel content."""
        content = KernelContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    if self.debug:
                        print(f"Processing kernel content child: {child.data}")
                    if child.data == "block_size":
                        # Pass the whole block_size node
                        content.block_size = self._parse_tuple_expression(child)
                    elif child.data == "grid_size":
                        # Pass the whole grid_size node
                        content.grid_size = self._parse_tuple_expression(child)
                    elif child.data == "shared_memory":
                        expr_child = child.children[0] if len(child.children) > 0 else None
                        if expr_child is not None:
                            if self.debug:
                                print(f"Processing shared_memory expr: {expr_child}")
                            content.shared_memory = self._parse_expression(expr_child)
                    elif child.data == "use_thrust":
                        if len(child.children) > 0:
                            value = str(child.children[0])
                            content.use_thrust = value == "true"
                    elif child.data == "kernel_input_param":
                        if len(child.children) > 0:
                            content.parameters.extend(self._process_kernel_param_list(child.children[0], KernelParameterDirection.IN))
                    elif child.data == "kernel_output_param":
                        if len(child.children) > 0:
                            content.parameters.extend(self._process_kernel_param_list(child.children[0], KernelParameterDirection.OUT))
                    elif child.data == "cuda_include":
                        # Handle CUDA include statements
                        include_path = self._parse_include_path(child.children[0])
                        content.cuda_includes.append(include_path)
                    elif child.data == "code_block":
                        if len(child.children) > 0:
                            extracted_code = self._extract_code_from_block(child.children[0])
                            content.code = extracted_code
        except Exception as e:
            if self.debug:
                print(f"Error processing kernel content: {e}")
            self.errors.append(f"Kernel content error: {e}")
        return content

    def _process_kernel_param_list(self, node: Tree, direction) -> List[KernelParamNode]:
        """Process kernel parameter list."""
        params = []
        for child in node.children:
            if isinstance(child, Tree) and child.data == "kernel_param":
                params.append(self._process_kernel_param(child, direction))
        return params

    def _process_kernel_param(self, node: Tree, direction) -> KernelParamNode:
        """Process kernel parameter."""
        # Extract cpp_type as string
        type_node = node.children[0]
        if isinstance(type_node, Tree):
            # Descend to the token
            if len(type_node.children) > 0:
                if isinstance(type_node.children[0], Tree):
                    # e.g., cpp_type -> cpp_type_name -> Token
                    param_type = str(type_node.children[0].children[0])
                else:
                    param_type = str(type_node.children[0])
            else:
                param_type = str(type_node)
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
        
        if not isinstance(node, Tree) or len(node.children) < 3:
            if self.debug:
                print(f"[DEBUG] Returning default values: (256, 1, 1)")
            return (256, 1, 1)  # Default values
        
        # The structure should be: LPAR expr COMMA expr COMMA expr RPAR
        # So children should be: [LPAR, expr1, COMMA, expr2, COMMA, expr3, RPAR]
        # We want indices 1, 3, 5 for the three expressions
        if len(node.children) >= 7:
            expr1 = self._parse_expression(node.children[1])
            expr2 = self._parse_expression(node.children[3])
            expr3 = self._parse_expression(node.children[5])
            result = (expr1, expr2, expr3)
            if self.debug:
                print(f"[DEBUG] Parsed tuple from indices 1,3,5: {result}")
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

    def _parse_expression(self, node: Tree) -> int:
        """Parse a simple expression and return an integer value."""
        if isinstance(node, Tree):
            if node.data == "expr":
                # Handle expression tree
                if len(node.children) > 0:
                    atom = node.children[0]
                    if isinstance(atom, Tree) and atom.data == "signed_atom":
                        if len(atom.children) > 0:
                            value = atom.children[0]
                            if hasattr(value, 'type') and value.type == 'SIGNED_NUMBER':
                                try:
                                    return int(value.value)
                                except ValueError:
                                    return 256  # Default
                            elif hasattr(value, 'type') and value.type == 'NAME':
                                # For variable names, return a default
                                return 256
                            else:
                                return 256
                        else:
                            return 256
                    else:
                        return 256
                else:
                    return 256
            else:
                return 256
        else:
            # Direct token
            if hasattr(node, 'type') and node.type == 'SIGNED_NUMBER':
                try:
                    return int(node.value)
                except ValueError:
                    return 256
            else:
                return 256

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
                # Handle expressions - extract the signed_atom
                if len(value_node.children) > 0:
                    atom = value_node.children[0]
                    if isinstance(atom, Tree) and atom.data == "signed_atom":
                        if len(atom.children) > 0:
                            value = atom.children[0]
                            if hasattr(value, 'type'):
                                if value.type == 'NAME':
                                    # For QoS enum values like 'reliable', 'best_effort', etc.
                                    value = value.value
                                elif value.type == 'SIGNED_NUMBER':
                                    # For numeric values
                                    try:
                                        if '.' in value.value:
                                            value = float(value.value)
                                        else:
                                            value = int(value.value)
                                    except ValueError:
                                        value = value.value
                                else:
                                    value = value.value
                            else:
                                value = str(value)
                        else:
                            value = str(atom)
                    else:
                        value = str(atom)
                else:
                    value = str(value_node)
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
                        topic_path = str(child.children[0])
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
        from_topic = str(node.children[0])
        to_topic = str(node.children[1])
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
    
    def _process_pyclass_def(self, node: Tree):
        """Process Pythonic class definition."""
        try:
            # pyclass_def: "pyclass" NAME inheritance? LBRACE pyclass_content RBRACE
            name = self._extract_token_value(node.children[0])
            inheritance = None
            content_node = None
            
            for child in node.children[1:]:
                if isinstance(child, Tree) and child.data == "inheritance":
                    inheritance = self._process_inheritance(child)
                elif isinstance(child, Tree) and child.data == "pyclass_content":
                    content_node = child
            
            if content_node:
                content = self._process_pyclass_content(content_node)
                pyclass_node = PyClassNode(name=name, inheritance=inheritance, content=content)
                self.ast.data_structures.append(pyclass_node)
                if self.debug:
                    print(f"Added pyclass: {name}")
            else:
                if self.debug:
                    print(f"pyclass_def missing content")
                self.errors.append(f"pyclass_def missing content")
        except Exception as e:
            if self.debug:
                print(f"Error processing pyclass definition: {e}")
            self.errors.append(f"Pyclass definition error: {e}")
    
    def _process_pyclass_content(self, node: Tree) -> PyClassContentNode:
        """Process Pythonic class content."""
        content = PyClassContentNode()
        try:
            for child in node.children:
                if isinstance(child, Tree):
                    if child.data == "pyclass_attribute":
                        content.attributes.append(self._process_pyclass_attribute(child))
                    elif child.data == "pyclass_method":
                        content.methods.append(self._process_pyclass_method(child))
                    elif child.data == "pyclass_constructor":
                        content.constructor = self._process_pyclass_constructor(child)
                    elif child.data == "pyclass_access_section":
                        content.access_sections.append(self._process_pyclass_access_section(child))
        except Exception as e:
            if self.debug:
                print(f"Error processing pyclass content: {e}")
            self.errors.append(f"Pyclass content error: {e}")
        return content
    
    def _process_pyclass_attribute(self, node: Tree) -> PyClassAttributeNode:
        """Process Pythonic class attribute."""
        try:
            name = self._extract_token_value(node.children[0])
            cpp_type = self._extract_token_value(node.children[1])
            default_value = None
            if len(node.children) > 2:
                default_value = self._process_value(node.children[2])
            return PyClassAttributeNode(name=name, cpp_type=cpp_type, default_value=default_value)
        except Exception as e:
            if self.debug:
                print(f"Error processing pyclass attribute: {e}")
            self.errors.append(f"Pyclass attribute error: {e}")
    
    def _process_pyclass_method(self, node: Tree) -> PyClassMethodNode:
        """Process Pythonic class method."""
        try:
            name = self._extract_token_value(node.children[0])
            # TODO: Parse parameters and return type
            return PyClassMethodNode(name=name)
        except Exception as e:
            if self.debug:
                print(f"Error processing pyclass method: {e}")
            self.errors.append(f"Pyclass method error: {e}")
    
    def _process_pyclass_constructor(self, node: Tree) -> PyClassConstructorNode:
        """Process Pythonic class constructor."""
        try:
            # TODO: Parse parameters and content
            return PyClassConstructorNode()
        except Exception as e:
            if self.debug:
                print(f"Error processing pyclass constructor: {e}")
            self.errors.append(f"Pyclass constructor error: {e}")
    
    def _process_pyclass_access_section(self, node: Tree) -> PyClassAccessSectionNode:
        """Process Pythonic class access section."""
        try:
            access_specifier = self._extract_token_value(node.children[0])
            # TODO: Parse section content
            return PyClassAccessSectionNode(access_specifier=access_specifier)
        except Exception as e:
            if self.debug:
                print(f"Error processing pyclass access section: {e}")
            self.errors.append(f"Pyclass access section error: {e}")

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