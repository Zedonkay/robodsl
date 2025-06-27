"""AST Builder for RoboDSL Lark parser.

This module converts Lark parse trees to our AST data structures,
providing a clean representation of parsed configurations.
"""

from typing import Any, Dict, List, Optional, Union
from lark import Tree, Token
from pathlib import Path
import sys

from ..ast import (
    RoboDSLAST, IncludeNode, NodeNode, NodeContentNode, ParameterNode, ValueNode,
    LifecycleNode, LifecycleSettingNode, TimerNode, TimerSettingNode,
    RemapNode, NamespaceNode, FlagNode, QoSNode, QoSSettingNode,
    PublisherNode, SubscriberNode, ServiceNode, ActionNode,
    CudaKernelsNode, KernelNode, KernelContentNode, KernelParamNode,
    QoSReliability, QoSDurability, QoSHistory, QoSLiveliness, KernelParameterDirection,
    CppMethodNode, ClientNode, MethodParamNode
)


class ASTBuilder:
    """Builds AST from Lark parse tree."""
    
    def __init__(self):
        self.ast = RoboDSLAST()
    
    def build(self, tree: Tree) -> RoboDSLAST:
        """Build AST from Lark parse tree."""
        self.ast = RoboDSLAST()
        
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'include_stmt':
                    include_node = self._handle_include(child)
                    self.ast.includes.append(include_node)
                elif child.data == 'node_def':
                    node_node = self._handle_node(child)
                    self.ast.nodes.append(node_node)
                elif child.data == 'cuda_kernels_block':
                    self.ast.cuda_kernels = self._handle_cuda_kernels(child)
        
        return self.ast
    
    def _handle_include(self, tree: Tree) -> IncludeNode:
        """Handle include statement."""
        path = ""
        is_system = False
        
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'STRING':
                    path = child.value.strip('"')
                    is_system = False
                elif child.type == 'NAME':
                    # Handle unquoted system include path
                    path = child.value
                    is_system = True
            elif isinstance(child, Tree) and child.data == 'include_path':
                # Extract path from include_path rule
                path_parts = []
                for path_child in child.children:
                    if isinstance(path_child, Token):
                        path_parts.append(path_child.value)
                # If the last two parts are a filename and extension, join with a dot
                if len(path_parts) > 1 and not '/' in path_parts[-1] and path_parts[-1].isalpha():
                    # e.g., ['system_header', 'h'] -> 'system_header.h'
                    path = '/'.join(path_parts[:-2] + [path_parts[-2] + '.' + path_parts[-1]])
                else:
                    path = '/'.join(path_parts)
                is_system = True
        
        return IncludeNode(path=path, is_system=is_system)
    
    def _handle_node(self, tree: Tree) -> NodeNode:
        """Handle node definition."""
        node_name = None
        node_content = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                node_name = child.value
            elif isinstance(child, Tree) and child.data == 'node_content':
                node_content = self._parse_node_content(child)
        
        if node_name and node_content:
            return NodeNode(name=node_name, content=node_content)
        
        # Fallback
        return NodeNode(name=node_name or "unknown", content=node_content or NodeContentNode())
    
    def _parse_node_content(self, content_tree: Tree) -> NodeContentNode:
        """Parse node content and return NodeContentNode."""
        content = NodeContentNode()
        
        for child in content_tree.children:
            if isinstance(child, Tree):
                if child.data == 'parameter':
                    content.parameters.append(self._handle_parameter(child))
                elif child.data == 'lifecycle':
                    content.lifecycle = self._handle_lifecycle(child)
                elif child.data == 'timer':
                    content.timers.append(self._handle_timer(child))
                elif child.data == 'remap':
                    content.remaps.append(self._handle_remap(child))
                elif child.data == 'namespace':
                    content.namespace = self._handle_namespace(child)
                elif child.data == 'flag':
                    content.flags.append(self._handle_flag(child))
                elif child.data == 'cpp_method':
                    content.cpp_methods.append(self._handle_cpp_method(child))
                elif child.data == 'ros_primitive':
                    # Handle ROS primitives (publisher, subscriber, service, client, action)
                    primitive_child = child.children[0]
                    if primitive_child.data == 'publisher':
                        content.publishers.append(self._handle_publisher(primitive_child))
                    elif primitive_child.data == 'subscriber':
                        content.subscribers.append(self._handle_subscriber(primitive_child))
                    elif primitive_child.data == 'service':
                        content.services.append(self._handle_service(primitive_child))
                    elif primitive_child.data == 'client':
                        content.clients.append(self._handle_client(primitive_child))
                    elif primitive_child.data == 'action':
                        content.actions.append(self._handle_action(primitive_child))
                elif child.data == 'kernel_def':
                    # Handle CUDA kernels inside nodes
                    content.cuda_kernels.append(self._handle_kernel_def(child))
        
        return content
    
    def _handle_parameter(self, tree: Tree) -> ParameterNode:
        """Handle parameter definition."""
        param_name = None
        param_value = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                param_name = child.value
            elif isinstance(child, Tree) and child.data == 'value':
                param_value = self._extract_value(child)
        
        value_node = ValueNode(value=param_value)
        return ParameterNode(name=param_name or "", value=value_node)
    
    def _handle_lifecycle(self, tree: Tree) -> LifecycleNode:
        """Handle lifecycle configuration."""
        settings = []
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'lifecycle_config':
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'lifecycle_setting':
                        name, value = self._extract_setting(setting)
                        settings.append(LifecycleSettingNode(name=name or "", value=value))
        
        return LifecycleNode(settings=settings)
    
    def _handle_timer(self, tree: Tree) -> TimerNode:
        """Handle timer definition."""
        timer_name = None
        period = None
        settings = []
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                timer_name = child.value
            elif isinstance(child, Tree) and child.data == 'number':
                period = self._extract_number(child)
            elif isinstance(child, Tree) and child.data == 'timer_config':
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'timer_setting':
                        name, value = self._extract_setting(setting)
                        settings.append(TimerSettingNode(name=name or "", value=value))
        
        return TimerNode(
            name=timer_name or "",
            period=period or 1.0,
            settings=settings
        )
    
    def _handle_remap(self, tree: Tree) -> RemapNode:
        """Handle remapping rule."""
        from_topic = None
        to_topic = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'TOPIC_PATH':
                if from_topic is None:
                    from_topic = child.value
                else:
                    to_topic = child.value
        
        return RemapNode(
            from_topic=from_topic or "",
            to_topic=to_topic or ""
        )
    
    def _handle_namespace(self, tree: Tree) -> NamespaceNode:
        """Handle namespace definition."""
        namespace = ""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'TOPIC_PATH':
                namespace = child.value
        
        return NamespaceNode(namespace=namespace)
    
    def _handle_flag(self, tree: Tree) -> FlagNode:
        """Handle boolean flags."""
        flag_name = None
        flag_value = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                flag_name = child.value
            elif isinstance(child, Token) and child.type == 'BOOLEAN':
                flag_value = child.value.lower() == 'true'
        
        return FlagNode(name=flag_name or "", value=flag_value or False)
    
    def _handle_publisher(self, tree: Tree) -> PublisherNode:
        """Handle publisher definition."""
        topic = None
        msg_type = None
        qos_config = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'topic_path':
                topic = self._extract_topic_path(child)
            elif isinstance(child, Token) and child.type == 'STRING':
                msg_type = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'publisher_config':
                # Look for publisher_setting -> qos_config
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'publisher_setting':
                        for config_child in setting.children:
                            if isinstance(config_child, Tree) and config_child.data == 'qos_config':
                                qos_config = self._handle_qos_config(config_child)
                                break
        
        return PublisherNode(
            topic=topic or "",
            msg_type=msg_type or "",
            qos=qos_config
        )
    
    def _handle_subscriber(self, tree: Tree) -> SubscriberNode:
        """Handle subscriber definition."""
        topic = None
        msg_type = None
        qos_config = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'topic_path':
                topic = self._extract_topic_path(child)
            elif isinstance(child, Token) and child.type == 'STRING':
                msg_type = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'subscriber_config':
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'subscriber_setting':
                        for config_child in setting.children:
                            if isinstance(config_child, Tree) and config_child.data == 'qos_config':
                                qos_config = self._handle_qos_config(config_child)
                                break
        
        return SubscriberNode(
            topic=topic or "",
            msg_type=msg_type or "",
            qos=qos_config
        )
    
    def _handle_service(self, tree: Tree) -> ServiceNode:
        """Handle service definition."""
        service_name = None
        srv_type = None
        qos_config = None
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'topic_path':
                service_name = self._extract_topic_path(child)
            elif isinstance(child, Token) and child.type == 'TOPIC_PATH':
                service_name = child.value
            elif isinstance(child, Token) and child.type == 'STRING':
                srv_type = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'service_config':
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'service_setting':
                        for config_child in setting.children:
                            if isinstance(config_child, Tree) and config_child.data == 'qos_config':
                                qos_config = self._handle_qos_config(config_child)
                                break
        return ServiceNode(
            service=service_name or "",
            srv_type=srv_type or "",
            qos=qos_config
        )
    
    def _handle_action(self, tree: Tree) -> ActionNode:
        """Handle action definition."""
        action_name = None
        action_type = None
        qos_config = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'topic_path':
                action_name = self._extract_topic_path(child)
            elif isinstance(child, Token) and child.type == 'STRING':
                action_type = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'action_config':
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'action_setting':
                        for config_child in setting.children:
                            if isinstance(config_child, Tree) and config_child.data == 'qos_config':
                                qos_config = self._handle_qos_config(config_child)
                                break
        
        return ActionNode(
            name=action_name or "",
            action_type=action_type or "",
            qos=qos_config
        )
    
    def _handle_client(self, tree: Tree) -> ClientNode:
        """Handle client definition."""
        client_name = None
        srv_type = None
        qos_config = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'topic_path':
                client_name = self._extract_topic_path(child)
            elif isinstance(child, Token) and child.type == 'STRING':
                srv_type = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'client_config':
                for setting in child.children:
                    if isinstance(setting, Tree) and setting.data == 'client_setting':
                        for config_child in setting.children:
                            if isinstance(config_child, Tree) and config_child.data == 'qos_config':
                                qos_config = self._handle_qos_config(config_child)
                                break
        
        return ClientNode(
            service=client_name or "",
            srv_type=srv_type or "",
            qos=qos_config
        )
    
    def _handle_qos_config(self, tree: Tree) -> QoSNode:
        """Handle QoS configuration."""
        settings = []
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'qos_setting':
                name, value = self._extract_qos_setting(child)
                if name is not None and value is not None:
                    settings.append(QoSSettingNode(name=name, value=value))
        return QoSNode(settings=settings)
    
    def _extract_qos_setting(self, tree: Tree) -> tuple[str, Any]:
        """Extract QoS setting name and value."""
        name = None
        value = None
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'NAME' and name is None:
                    name = child.value
                elif child.type == 'NAME' and name is not None:
                    # Second NAME token is the value (e.g., 'reliable', 'best_effort')
                    value = child.value
                elif child.type == 'NUMBER':
                    value = int(child.value) if '.' not in child.value else float(child.value)
                elif child.type == 'BOOLEAN':
                    value = child.value.lower() == 'true'
                elif child.type == 'STRING':
                    value = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'number':
                value = self._extract_number(child)
        return name or "", value
    
    def _handle_cuda_kernels(self, tree: Tree) -> CudaKernelsNode:
        """Handle CUDA kernels block."""
        kernels = []
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'kernel_def':
                kernel_node = self._parse_kernel(child)
                if kernel_node:
                    kernels.append(kernel_node)
        
        return CudaKernelsNode(kernels=kernels)
    
    def _parse_kernel(self, tree: Tree) -> Optional[KernelNode]:
        """Parse CUDA kernel definition."""
        kernel_name = None
        kernel_content = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                kernel_name = child.value
            elif isinstance(child, Tree) and child.data == 'kernel_content':
                kernel_content = self._parse_kernel_content(child)
        
        if kernel_name and kernel_content:
            return KernelNode(name=kernel_name, content=kernel_content)
        
        return None
    
    def _parse_kernel_content(self, tree: Tree) -> KernelContentNode:
        content = KernelContentNode()
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'block_size':
                    content.block_size = self._extract_tuple(child, 3)
                elif child.data == 'grid_size':
                    content.grid_size = self._extract_tuple(child, 3)
                elif child.data == 'shared_memory':
                    content.shared_memory = self._extract_number(child.children[0])
                elif child.data == 'use_thrust':
                    # Set use_thrust based on the boolean value in the tree
                    for token in child.children:
                        if isinstance(token, Token) and token.type == 'BOOLEAN':
                            content.use_thrust = token.value.lower() == 'true'
                elif child.data == 'kernel_param':
                    param = self._handle_kernel_param(child)
                    if param:
                        content.parameters.append(param)
                elif child.data == 'code_block':
                    content.code = self._extract_code_block(child)
        return content
    
    def _handle_kernel_param(self, tree: Tree) -> KernelParamNode:
        direction = None
        param_type = None
        param_name = None
        size_expr = None
        
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'DIRECTION':
                    direction = KernelParameterDirection(child.value)
                elif child.type == 'NAME':
                    # First NAME is the type, second is the parameter name
                    if param_type is None:
                        param_type = child.value
                    elif param_name is None:
                        param_name = child.value
            elif isinstance(child, Tree) and child.data == 'cpp_type':
                # Handle cpp_type which can be NAME or NAME "*"
                type_parts = []
                for token in child.children:
                    if isinstance(token, Token):
                        type_parts.append(token.value)
                param_type = ''.join(type_parts)
            elif isinstance(child, Tree) and child.data == 'kernel_param_size':
                # kernel_param_size_list is the first child
                size_expr = [tok.value for tok in child.children[0].children if isinstance(tok, Token)]
        
        return KernelParamNode(direction=direction, param_type=param_type, param_name=param_name, size_expr=size_expr)
    
    def _extract_value(self, tree: Tree) -> Any:
        """Extract value from value tree."""
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'primitive':
                    return self._extract_primitive(child)
                elif child.data == 'array':
                    return self._extract_array(child)
                elif child.data == 'nested_dict':
                    return self._extract_nested_dict(child)
        
        return None
    
    def _extract_primitive(self, tree: Tree) -> Any:
        """Extract primitive value from tree."""
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'BOOLEAN':
                    return child.value.lower() == 'true'
                elif child.type == 'INTEGER':
                    return int(child.value)
                elif child.type == 'FLOAT':
                    return float(child.value)
                elif child.type == 'STRING':
                    return child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'number':
                return self._extract_number(child)
        return None
    
    def _extract_array(self, tree: Tree) -> list:
        """Extract array from tree."""
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'value_list':
                return [self._extract_value(item) for item in child.children if isinstance(item, Tree)]
        return []
    
    def _extract_nested_dict(self, tree: Tree) -> dict:
        """Extract nested dictionary from tree."""
        result = {}
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'dict_list':
                for item in child.children:
                    if isinstance(item, Tree) and item.data == 'dict_item':
                        name, value = self._extract_dict_item(item)
                        result[name] = value
        return result
    
    def _extract_setting(self, tree: Tree) -> tuple[str, Any]:
        """Extract name-value setting from tree."""
        name = None
        value = None
        
        for i, child in enumerate(tree.children):
            if isinstance(child, Token):
                if child.type == 'NAME' and name is None:
                    name = child.value
                elif child.type == 'BOOLEAN':
                    value = child.value.lower() == 'true'
                elif child.type == 'STRING':
                    value = child.value.strip('"')
                elif child.type == 'NUMBER':
                    value = int(child.value) if '.' not in child.value else float(child.value)
            elif isinstance(child, Tree) and child.data == 'number':
                value = self._extract_number(child)
        
        return name or "", value
    
    def _extract_tuple(self, tree: Tree, size: int) -> tuple:
        """Extract tuple of numbers from tree."""
        numbers = []
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'number':
                numbers.append(self._extract_number(child))
        return tuple(numbers[:size])
    
    def _extract_string(self, tree: Tree) -> str:
        """Extract string value from tree."""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                return child.value.strip('"')
        return ""
    
    def _extract_include(self, tree: Tree) -> str:
        """Extract include path from include statement."""
        return self._extract_string(tree)
    
    def _extract_define(self, tree: Tree) -> tuple[str, str]:
        """Extract define name and value."""
        name = None
        value = ""
        
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'NAME':
                    name = child.value
                elif child.type == 'STRING':
                    value = child.value.strip('"')
        
        return name or "", value
    
    def _extract_dict_item(self, tree: Tree) -> tuple[str, Any]:
        """Extract dictionary item."""
        name = None
        value = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                name = child.value
            elif isinstance(child, Tree) and child.data == 'value':
                value = self._extract_value(child)
        
        return name or "", value
    
    def _handle_cpp_method(self, tree: Tree) -> CppMethodNode:
        method_name = None
        inputs = []
        outputs = []
        code = ""
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                method_name = child.value
            elif isinstance(child, Tree) and child.data == 'method_content':
                for content_child in child.children:
                    if isinstance(content_child, Tree):
                        if content_child.data == 'input_param':
                            inputs.append(self._handle_method_param(content_child))
                        elif content_child.data == 'output_param':
                            outputs.append(self._handle_method_param(content_child))
                        elif content_child.data == 'code_block':
                            code = self._extract_code_block(content_child)
        
        return CppMethodNode(name=method_name or '', inputs=inputs, outputs=outputs, code=code)

    def _handle_method_param(self, tree: Tree) -> MethodParamNode:
        print('DEBUG _handle_method_param:', tree.pretty(), file=sys.stderr)
        param_type = None
        param_name = None
        size_expr = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'cpp_type':
                param_type = self._extract_cpp_type(child)
            elif isinstance(child, Token) and child.type == 'NAME':
                param_name = child.value
            elif isinstance(child, Tree) and (child.data == 'input_param_size' or child.data == 'output_param_size'):
                # Extract size expression from the method_param_size_list
                if child.children and isinstance(child.children[0], Tree) and child.children[0].data == 'method_param_size_list':
                    size_parts = []
                    for token in child.children[0].children:
                        if isinstance(token, Token):
                            size_parts.append(token.value)
                    size_expr = ', '.join(size_parts)
        
        return MethodParamNode(param_type=param_type or '', param_name=param_name or '', size_expr=size_expr)

    def _extract_number(self, tree: Tree) -> float:
        """Extract number from tree."""
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'NUMBER':
                    value = child.value
                    if '.' in value:
                        return float(value)
                    else:
                        return int(value)
        return 0.0

    def _extract_cpp_type(self, tree: Tree) -> str:
        """Extract C++ type from tree."""
        type_parts = []
        pointer = False
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'cpp_type_name':
                type_parts.extend(self._extract_cpp_type_name(child))
            elif isinstance(child, Token):
                if child.type == 'STAR':
                    pointer = True
                else:
                    type_parts.append(child.value)
        type_str = "".join(type_parts)
        if pointer:
            type_str += '*'
        return type_str

    def _extract_cpp_type_name(self, tree: Tree) -> List[str]:
        """Extract C++ type name from tree, handling namespaces and templates."""
        namespace_parts = []
        template_str = ''
        in_template = False
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'NAME':
                    if not in_template:
                        namespace_parts.append(child.value)
                    else:
                        template_str += child.value
                elif child.type == 'LESSTHAN':
                    in_template = True
                elif child.type == 'MORETHAN':
                    in_template = False
                elif child.type == 'COMMA':
                    template_str += ','
                else:
                    template_str += child.value
            elif isinstance(child, Tree) and child.data == 'cpp_type_list':
                # Handle template parameters
                template_args = []
                for template_child in child.children:
                    if isinstance(template_child, Tree) and template_child.data == 'cpp_type_name':
                        template_args.append(''.join(self._extract_cpp_type_name(template_child)))
                    elif isinstance(template_child, Token):
                        template_args.append(template_child.value)
                template_str += '<' + ','.join(template_args) + '>'
        type_str = '::'.join(namespace_parts)
        if template_str:
            type_str += template_str
        return [type_str]

    def _extract_kernel_param_size(self, tree: Tree) -> str:
        for child in tree.children:
            if isinstance(child, Token) and child.type in ('NAME', 'STRING', 'NUMBER'):
                return child.value.strip('"')
        return ""

    def _extract_code_block(self, tree: Tree) -> str:
        """Extract code block content from tree (STRING)."""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                return child.value.strip('"')
        return ""

    def _extract_topic_path(self, tree: Tree) -> str:
        """Extract topic path from topic_path rule."""
        parts = []
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                parts.append(child.value)
        return '/' + '/'.join(parts) if parts else ''

    def _handle_kernel_def(self, tree: Tree) -> KernelNode:
        return self._parse_kernel(tree) 