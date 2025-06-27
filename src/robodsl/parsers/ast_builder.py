"""AST Builder for RoboDSL Lark parser.

This module converts Lark parse trees to our AST data structures,
providing a clean representation of parsed configurations.
"""

from typing import Any, Dict, List, Optional, Union
from lark import Tree, Token
from pathlib import Path
import sys

from ..core.ast import (
    RoboDSLAST, IncludeNode, NodeNode, NodeContentNode, ParameterNode, ValueNode,
    LifecycleNode, LifecycleSettingNode, TimerNode, TimerSettingNode,
    RemapNode, NamespaceNode, FlagNode, QoSNode, QoSSettingNode,
    PublisherNode, SubscriberNode, ServiceNode, ActionNode,
    CudaKernelsNode, KernelNode, KernelContentNode, KernelParamNode,
    QoSReliability, QoSDurability, QoSHistory, QoSLiveliness, KernelParameterDirection,
    CppMethodNode, ClientNode, MethodParamNode,
    OnnxModelNode, ModelConfigNode, InputDefNode, OutputDefNode, DeviceNode, OptimizationNode,
    PipelineNode, PipelineContentNode, StageNode, StageContentNode, StageInputNode, StageOutputNode, StageMethodNode, StageModelNode, StageTopicNode
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
                elif child.data == 'onnx_model':
                    onnx_model_node = self._handle_onnx_model(child)
                    self.ast.onnx_models.append(onnx_model_node)
                elif child.data == 'pipeline_def':
                    pipeline_node = self._handle_pipeline(child)
                    self.ast.pipelines.append(pipeline_node)
        
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
                elif child.data == 'onnx_model_ref':
                    # Handle ONNX models inside nodes
                    content.onnx_models.append(self._handle_onnx_model(child))
        
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
            elif isinstance(child, Tree) and child.data == 'expr':
                period = self._extract_expr(child)
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
                # Find the first qos_config node
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
        print('DEBUG: _handle_qos_config children:', tree.children)
        settings = []
        for child in tree.children:
            print('DEBUG: child type:', type(child), 'data:', getattr(child, 'data', None))
            if isinstance(child, Tree) and child.data == 'qos_setting':
                name, value = self._extract_qos_setting(child)
                if name is not None and value is not None:
                    settings.append(QoSSettingNode(name=name, value=value))
        return QoSNode(settings=settings)
    
    def _extract_qos_setting(self, tree: Tree) -> tuple[str, Any]:
        """Extract QoS setting name and value."""
        # print('DEBUG: _extract_qos_setting children:', tree.children)
        name = None
        value = None
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'NAME' and name is None:
                    name = child.value
            elif isinstance(child, Tree) and child.data == 'qos_value':
                # Find expr child
                for vchild in child.children:
                    if isinstance(vchild, Tree) and vchild.data == 'expr':
                        value = self._extract_expr(vchild)
        # print('DEBUG: extracted name:', name, 'value:', value)
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
                    content.shared_memory = self._extract_expr(child.children[0])
                elif child.data == 'use_thrust':
                    # Set use_thrust based on the boolean value in the tree
                    for token in child.children:
                        if isinstance(token, Token) and token.type == 'BOOLEAN':
                            content.use_thrust = token.value.lower() == 'true'
                elif child.data == 'kernel_input_param':
                    param = self._handle_kernel_input_param(child)
                    if param:
                        content.parameters.append(param)
                elif child.data == 'kernel_output_param':
                    param = self._handle_kernel_output_param(child)
                    if param:
                        content.parameters.append(param)
                elif child.data == 'code_block':
                    content.code = self._extract_code_block(child)
        return content
    
    def _handle_kernel_input_param(self, tree: Tree) -> KernelParamNode:
        param_type = ""
        param_name = None
        size_expr = None
        is_const = False
        # Find cpp_type, NAME, kernel_param_size in children
        for child in tree.children:
            if param_type == "" and (isinstance(child, Tree) or isinstance(child, Token)):
                # First non-keyword, non-colon child is type
                if isinstance(child, Tree) or (isinstance(child, Token) and child.type not in ("NAME", "COLON")):
                    param_type = self._extract_cpp_type(child)
            elif param_name is None and isinstance(child, Token) and child.type == "NAME":
                param_name = child.value
            elif size_expr is None and isinstance(child, Tree) and child.data == "kernel_param_size":
                size_expr = self._extract_kernel_param_size(child)
        if param_type.strip().startswith("const "):
            is_const = True
            param_type = param_type.strip()[6:].strip()
        return KernelParamNode(
            direction=KernelParameterDirection.IN,
            param_type=param_type,
            param_name=param_name,
            size_expr=size_expr,
            is_const=is_const
        )
    
    def _handle_kernel_output_param(self, tree: Tree) -> KernelParamNode:
        param_type = ""
        param_name = None
        size_expr = None
        is_const = False
        # Find cpp_type, NAME, kernel_param_size in children
        for child in tree.children:
            if param_type == "" and (isinstance(child, Tree) or isinstance(child, Token)):
                # First non-keyword, non-colon child is type
                if isinstance(child, Tree) or (isinstance(child, Token) and child.type not in ("NAME", "COLON")):
                    param_type = self._extract_cpp_type(child)
            elif param_name is None and isinstance(child, Token) and child.type == "NAME":
                param_name = child.value
            elif size_expr is None and isinstance(child, Tree) and child.data == "kernel_param_size":
                size_expr = self._extract_kernel_param_size(child)
        if param_type.strip().startswith("const "):
            is_const = True
            param_type = param_type.strip()[6:].strip()
        return KernelParamNode(
            direction=KernelParameterDirection.OUT,
            param_type=param_type,
            param_name=param_name,
            size_expr=size_expr,
            is_const=is_const
        )
    
    def _handle_kernel_param(self, tree: Tree) -> KernelParamNode:
        """Handle legacy kernel parameter (for backward compatibility)."""
        direction = None
        param_type = None
        param_name = None
        size_expr = None
        
        # The structure is: "param" DIRECTION cpp_type NAME kernel_param_size?
        # So children should be: [DIRECTION, cpp_type_tree, NAME, kernel_param_size_tree?]
        children = list(tree.children)
        
        if len(children) >= 3:
            # First child should be DIRECTION
            if isinstance(children[0], Token) and children[0].type == 'DIRECTION':
                direction = KernelParameterDirection(children[0].value)
            
            # Second child should be cpp_type
            if isinstance(children[1], Tree) and children[1].data == 'cpp_type':
                param_type = self._extract_cpp_type(children[1])
            
            # Third child should be NAME (parameter name)
            if isinstance(children[2], Token) and children[2].type == 'NAME':
                param_name = children[2].value
            
            # Fourth child (if exists) should be kernel_param_size
            if len(children) > 3 and isinstance(children[3], Tree) and children[3].data == 'kernel_param_size':
                size_expr = self._extract_kernel_param_size(children[3])
        
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
                elif child.type == 'SIGNED_NUMBER':
                    value = child.value
                    if '.' in value:
                        return float(value)
                    else:
                        return int(value)
                elif child.type == 'STRING':
                    return child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'expr':
                return self._extract_expr(child)
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
            if isinstance(child, Tree) and child.data == 'expr':
                numbers.append(self._extract_expr(child))
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
        """Handle method parameter."""
        param_type = None
        param_name = None
        size_expr = None
        is_const = False
        
        for child in tree.children:
            if param_type is None and isinstance(child, Tree) and child.data == 'cpp_type':
                param_type = self._extract_cpp_type(child)
            elif param_name is None and isinstance(child, Token) and child.type == 'NAME':
                param_name = child.value
            elif size_expr is None and isinstance(child, Tree) and child.data in ('input_param_size', 'output_param_size'):
                size_expr = self._extract_method_param_size(child)
        
        if param_type and param_type.strip().startswith("const "):
            is_const = True
            param_type = param_type.strip()[6:].strip()
        
        return MethodParamNode(
            param_type=param_type or '', 
            param_name=param_name or '', 
            size_expr=size_expr,
            is_const=is_const
        )

    def _extract_expr(self, tree: Tree) -> Any:
        # If the expr is a single signed_atom, extract it
        if len(tree.children) == 1:
            child = tree.children[0]
            if isinstance(child, Tree) and child.data == 'signed_atom':
                return self._extract_signed_atom(child)
        # Otherwise, reconstruct the expression as a string
        return self._expr_to_str(tree)
    
    def _expr_to_str(self, tree: Tree) -> str:
        """Convert expression tree to string representation."""
        parts = []
        for child in tree.children:
            if isinstance(child, Token):
                parts.append(child.value)
            elif isinstance(child, Tree):
                if child.data == 'signed_atom':
                    parts.append(str(self._extract_signed_atom(child)))
                elif child.data == 'binop':
                    parts.append(child.children[0].value if child.children else '')
                else:
                    parts.append(self._expr_to_str(child))
        return ' '.join(parts)
    
    def _extract_signed_atom(self, tree: Tree) -> Any:
        """Extract signed atom value from tree."""
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'SIGNED_NUMBER':
                    value = child.value
                    if '.' in value:
                        return float(value)
                    else:
                        return int(value)
                elif child.type == 'NAME':
                    # Check if it's a boolean value
                    if child.value.lower() == 'true':
                        return True
                    elif child.value.lower() == 'false':
                        return False
                    return child.value
            elif isinstance(child, Tree) and child.data == 'dotted_name':
                # For dotted names, return as string for now
                return self._extract_dotted_name(child)
        return None
    
    def _extract_dotted_name(self, tree: Tree) -> str:
        """Extract dotted name as string."""
        parts = []
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                parts.append(child.value)
            elif isinstance(child, Token) and child.type == 'DOT':
                parts.append('.')
        return ''.join(parts)

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

    def _extract_cpp_type(self, tree) -> str:
        """Extract C++ type from tree or token."""
        if hasattr(tree, 'children'):
            for child in tree.children:
                if isinstance(child, Tree) and child.data == 'cpp_type_name':
                    return self._extract_cpp_type_name(child)
        elif hasattr(tree, 'value'):
            return tree.value
        return ""
    
    def _extract_cpp_type_name(self, tree: Tree) -> str:
        """Extract C++ type name from tree."""
        # With the new regex-based rule, the type name is a single token
        if tree.children and isinstance(tree.children[0], Token):
            return tree.children[0].value
        return ""

    def _extract_kernel_param_size(self, tree: Tree) -> List[str]:
        """Extract kernel parameter size expression from tree."""
        size_parts = []
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'kernel_param_size_list':
                for item in child.children:
                    if isinstance(item, Tree) and item.data == 'kernel_param_size_item':
                        # Extract the value from the item
                        for item_child in item.children:
                            if isinstance(item_child, Token):
                                if item_child.type in ('SIGNED_NUMBER', 'NAME', 'STRING'):
                                    size_parts.append(item_child.value)
                            elif isinstance(item_child, Tree):
                                # Handle nested expressions
                                if item_child.data == 'expr':
                                    size_parts.append(str(self._extract_expr(item_child)))
                    elif isinstance(item, Token):
                        # kernel_param_size_item is a direct token
                        if item.type in ('SIGNED_NUMBER', 'NAME', 'STRING'):
                            size_parts.append(item.value)
        return size_parts if size_parts else None

    def _extract_code_block(self, tree: Tree) -> str:
        """Extract code block from tree."""
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'balanced_braces':
                return self._extract_balanced_braces(child)
        return ""
    
    def _extract_balanced_braces(self, tree: Tree) -> str:
        """Recursively extract all text from balanced_braces."""
        result = []
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'balanced_content':
                result.append(self._extract_balanced_content(child))
            else:
                result.append(str(child))
        return '{' + ''.join(result) + '}'
    
    def _extract_balanced_content(self, tree: Tree) -> str:
        """Extract content from balanced_content rule."""
        result = []
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'balanced_braces':
                result.append(self._extract_balanced_braces(child))
            else:
                result.append(str(child))
        return ''.join(result)

    def _extract_topic_path(self, tree: Tree) -> str:
        """Extract topic path from tree."""
        path_parts = []
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                path_parts.append(child.value)
        return '/' + '/'.join(path_parts)

    def _handle_kernel_def(self, tree: Tree) -> KernelNode:
        """Handle kernel definition inside nodes."""
        return self._parse_kernel(tree)

    def _handle_onnx_model(self, tree: Tree) -> OnnxModelNode:
        """Handle ONNX model definition."""
        model_name = None
        model_config = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                model_name = child.value.strip('"')
            elif isinstance(child, Tree) and child.data == 'model_config':
                model_config = self._handle_model_config(child)
        
        return OnnxModelNode(name=model_name or "", config=model_config or ModelConfigNode())

    def _handle_model_config(self, tree: Tree) -> ModelConfigNode:
        """Handle model configuration."""
        config = ModelConfigNode()
        
        for child in tree.children:
            if isinstance(child, Tree):
                if child.data == 'input_def':
                    config.inputs.append(self._handle_input_def(child))
                elif child.data == 'output_def':
                    config.outputs.append(self._handle_output_def(child))
                elif child.data == 'device':
                    config.device = self._handle_device(child)
                elif child.data == 'optimization':
                    config.optimizations.append(self._handle_optimization(child))
        
        return config

    def _handle_input_def(self, tree: Tree) -> InputDefNode:
        """Handle input definition."""
        input_name = None
        input_type = None
        
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'STRING':
                    if input_name is None:
                        input_name = child.value.strip('"')
                    else:
                        input_type = child.value.strip('"')
        
        return InputDefNode(name=input_name or "", type=input_type or "")

    def _handle_output_def(self, tree: Tree) -> OutputDefNode:
        """Handle output definition."""
        output_name = None
        output_type = None
        
        for child in tree.children:
            if isinstance(child, Token):
                if child.type == 'STRING':
                    if output_name is None:
                        output_name = child.value.strip('"')
                    else:
                        output_type = child.value.strip('"')
        
        return OutputDefNode(name=output_name or "", type=output_type or "")

    def _handle_device(self, tree: Tree) -> DeviceNode:
        """Handle device definition."""
        device_name = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'device_type':
                # Extract the device type from the device_type rule
                for device_child in child.children:
                    if isinstance(device_child, Token):
                        device_name = device_child.value
                        break
            elif isinstance(child, Token) and child.type == 'NAME':
                device_name = child.value
        
        return DeviceNode(device=device_name or "")

    def _handle_optimization(self, tree: Tree) -> OptimizationNode:
        """Handle optimization definition."""
        optimization_name = None
        
        for child in tree.children:
            if isinstance(child, Tree) and child.data == 'optimization_type':
                # Extract the optimization type from the optimization_type rule
                for opt_child in child.children:
                    if isinstance(opt_child, Token):
                        optimization_name = opt_child.value
                        break
            elif isinstance(child, Token) and child.type == 'NAME':
                optimization_name = child.value
        
        return OptimizationNode(optimization=optimization_name or "")

    def _extract_method_param_size(self, tree: Tree) -> str:
        """Extract method parameter size expression."""
        size_parts = []
        if tree.children and isinstance(tree.children[0], Tree) and tree.children[0].data == 'method_param_size_list':
            for item in tree.children[0].children:
                if isinstance(item, Tree) and item.data == 'method_param_size_item':
                    # Extract the value from the item
                    for item_child in item.children:
                        if isinstance(item_child, Token):
                            if item_child.type in ('SIGNED_NUMBER', 'NAME', 'STRING'):
                                size_parts.append(item_child.value)
                        elif isinstance(item_child, Tree):
                            # Handle nested expressions
                            if item_child.data == 'expr':
                                size_parts.append(str(self._extract_expr(item_child)))
                elif isinstance(item, Token):
                    # method_param_size_item is a direct token
                    if item.type in ('SIGNED_NUMBER', 'NAME', 'STRING'):
                        size_parts.append(item.value)
        return ', '.join(size_parts)

    def _handle_pipeline(self, tree: Tree) -> PipelineNode:
        """Handle pipeline definition."""
        pipeline_name = None
        pipeline_content = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                pipeline_name = child.value
            elif isinstance(child, Tree) and child.data == 'pipeline_content':
                pipeline_content = self._parse_pipeline_content(child)
        
        return PipelineNode(name=pipeline_name or "", content=pipeline_content or PipelineContentNode())

    def _parse_pipeline_content(self, content_tree: Tree) -> PipelineContentNode:
        """Parse pipeline content and return PipelineContentNode."""
        content = PipelineContentNode()
        
        for child in content_tree.children:
            if isinstance(child, Tree):
                if child.data == 'stage_def':
                    content.stages.append(self._handle_stage(child))
        
        return content

    def _handle_stage(self, tree: Tree) -> StageNode:
        """Handle stage definition."""
        stage_name = None
        stage_content = None
        
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'NAME':
                stage_name = child.value
            elif isinstance(child, Tree) and child.data == 'stage_content':
                stage_content = self._parse_stage_content(child)
        
        return StageNode(name=stage_name or "", content=stage_content or StageContentNode())

    def _parse_stage_content(self, content_tree: Tree) -> StageContentNode:
        """Parse stage content and return StageContentNode."""
        content = StageContentNode()
        
        for child in content_tree.children:
            if isinstance(child, Tree):
                if child.data == 'stage_input':
                    content.inputs.append(self._handle_stage_input(child))
                elif child.data == 'stage_output':
                    content.outputs.append(self._handle_stage_output(child))
                elif child.data == 'stage_method':
                    content.methods.append(self._handle_stage_method(child))
                elif child.data == 'stage_model':
                    content.models.append(self._handle_stage_model(child))
                elif child.data == 'stage_topic':
                    content.topics.append(self._handle_stage_topic(child))
        
        return content

    def _handle_stage_input(self, tree: Tree) -> StageInputNode:
        """Handle stage input definition."""
        input_name = ""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                input_name = child.value.strip('"')
        return StageInputNode(input_name=input_name)

    def _handle_stage_output(self, tree: Tree) -> StageOutputNode:
        """Handle stage output definition."""
        output_name = ""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                output_name = child.value.strip('"')
        return StageOutputNode(output_name=output_name)

    def _handle_stage_method(self, tree: Tree) -> StageMethodNode:
        """Handle stage method definition."""
        method_name = ""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                method_name = child.value.strip('"')
        return StageMethodNode(method_name=method_name)

    def _handle_stage_model(self, tree: Tree) -> StageModelNode:
        """Handle stage model definition."""
        model_name = ""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'STRING':
                model_name = child.value.strip('"')
        return StageModelNode(model_name=model_name)

    def _handle_stage_topic(self, tree: Tree) -> StageTopicNode:
        """Handle stage topic definition."""
        topic_path = ""
        for child in tree.children:
            if isinstance(child, Token) and child.type == 'TOPIC_PATH':
                topic_path = child.value
        return StageTopicNode(topic_path=topic_path) 