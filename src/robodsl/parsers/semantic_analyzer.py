"""Semantic Analyzer for RoboDSL.

This module validates the semantic correctness of parsed RoboDSL configurations,
checking for errors that can't be caught by the grammar alone.
"""

from typing import List, Set, Dict, Any

from ..core.ast import (
    RoboDSLAST, NodeNode, NodeContentNode, ParameterNode, ValueNode,
    LifecycleNode, LifecycleSettingNode, TimerNode, TimerSettingNode,
    RemapNode, NamespaceNode, FlagNode, QoSNode, QoSSettingNode,
    PublisherNode, SubscriberNode, ServiceNode, ActionNode,
    CudaKernelsNode, KernelNode, KernelContentNode, KernelParamNode,
    QoSReliability, QoSDurability, QoSHistory, QoSLiveliness, KernelParameterDirection
)


class SemanticError(Exception):
    """Raised when a semantic error is found in the configuration."""
    pass


class SymbolTable:
    """Symbol table for tracking names and types in RoboDSL."""
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.kernels: Dict[str, Dict[str, Any]] = {}
        self.topics: Set[str] = set()
        self.services: Set[str] = set()
        self.actions: Set[str] = set()
        self.parameters: Dict[str, str] = {}
        self.errors: List[str] = []

    def add_node(self, name: str):
        if name in self.nodes:
            self.errors.append(f"Duplicate node name: {name}")
        self.nodes[name] = {}

    def add_kernel(self, name: str):
        if name in self.kernels:
            self.errors.append(f"Duplicate kernel name: {name}")
        self.kernels[name] = {}

    def add_topic(self, topic: str):
        if topic in self.topics:
            self.errors.append(f"Duplicate topic: {topic}")
        self.topics.add(topic)

    def add_service(self, service: str):
        if service in self.services:
            self.errors.append(f"Duplicate service: {service}")
        self.services.add(service)

    def add_action(self, action: str):
        if action in self.actions:
            self.errors.append(f"Duplicate action: {action}")
        self.actions.add(action)

    def add_parameter(self, name: str, param_type: str):
        if name in self.parameters:
            self.errors.append(f"Duplicate parameter: {name}")
        self.parameters[name] = param_type

    def get_errors(self) -> List[str]:
        return self.errors


class SemanticAnalyzer:
    """Analyzes semantic correctness of RoboDSL configurations."""
    
    def __init__(self, debug: bool = False):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.symbol_table = SymbolTable()
        self.debug = debug
    
    def analyze(self, ast: RoboDSLAST) -> bool:
        """Analyze the AST for semantic errors.
        
        Returns:
            True if no errors found, False otherwise.
        """
        self.errors.clear()
        self.warnings.clear()
        self.symbol_table = SymbolTable()
        
        # Analyze project-level configuration
        self._analyze_project_config(ast)
        
        # Analyze each node
        for node in ast.nodes:
            self.symbol_table.add_node(node.name)
            self._analyze_node(node)
        
        # Analyze CUDA kernels
        if ast.cuda_kernels:
            kernel_names = set()
            for kernel in ast.cuda_kernels.kernels:
                # Check for duplicate kernel names
                if kernel.name in kernel_names:
                    self.errors.append(f"Duplicate kernel name: {kernel.name}")
                kernel_names.add(kernel.name)
                
                self.symbol_table.add_kernel(kernel.name)
                self._analyze_cuda_kernel(kernel)
        
        # Check for cross-references
        self._analyze_cross_references(ast)
        
        self.errors.extend(self.symbol_table.get_errors())
        return len(self.errors) == 0
    
    def _analyze_project_config(self, ast: RoboDSLAST):
        """Analyze project-level configuration."""
        # Check for duplicate node names
        node_names = set()
        for node in ast.nodes:
            if node.name in node_names:
                self.errors.append(f"Duplicate node name: {node.name}")
            node_names.add(node.name)
        
        # Check for duplicate kernel names
        if ast.cuda_kernels:
            kernel_names = set()
            for kernel in ast.cuda_kernels.kernels:
                if kernel.name in kernel_names:
                    self.errors.append(f"Duplicate CUDA kernel name: {kernel.name}")
                kernel_names.add(kernel.name)
    
    def _analyze_node(self, node: NodeNode):
        """Analyze a single node configuration."""
        # Check node name
        if not node.name or node.name.strip() == "":
            self.errors.append("Node name cannot be empty")
        
        content = node.content
        
        # Check parameters
        self._analyze_parameters(content.parameters, node.name)
        
        # Check publishers
        self._analyze_publishers(content.publishers)
        
        # Check subscribers
        self._analyze_subscribers(content.subscribers)
        
        # Check services
        self._analyze_services(content.services)
        
        # Check actions
        self._analyze_actions(content.actions)
        
        # Check clients
        self._analyze_clients(content.clients)
        
        # Check flags
        self._analyze_flags(content.flags)
        
        # Check timers
        self._analyze_timers(content.timers)
        
        # Check remappings
        self._analyze_remappings(content.remaps)
        
        # Check C++ methods
        if hasattr(content, 'cpp_methods'):
            self._analyze_cpp_methods(content.cpp_methods, node.name)
        
        # Check QoS configurations
        for pub in content.publishers:
            if pub.qos:
                self._analyze_qos_config(pub.qos, f"publisher {pub.topic}")
        
        for sub in content.subscribers:
            if sub.qos:
                self._analyze_qos_config(sub.qos, f"subscriber {sub.topic}")
        
        for srv in content.services:
            if srv.qos:
                self._analyze_qos_config(srv.qos, f"service {srv.service}")
        
        for act in content.actions:
            if act.qos:
                self._analyze_qos_config(act.qos, f"action {act.name}")
    
    def _analyze_parameters(self, parameters: List[ParameterNode], node_name: str):
        """Analyze parameter configurations."""
        param_names = set()
        
        for param in parameters:
            # Check for duplicate parameter names
            if param.name in param_names:
                self.errors.append(f"Duplicate parameter name: {param.name}")
            param_names.add(param.name)
            
            # Check parameter name
            if not param.name or param.name.strip() == "":
                self.errors.append("Parameter name cannot be empty")
            
            # Check parameter value
            if param.value.value is None:
                self.errors.append(f"Parameter '{param.name}' has no value")
            
            # Strong type checking: validate declared type matches value
            if param.type and param.value.value is not None:
                self._validate_parameter_type(param.name, param.type, param.value.value)
            
            # Add to symbol table with declared type (preferred) or inferred type
            param_type = param.type if param.type else self._infer_parameter_type(param.value.value)
            self.symbol_table.add_parameter(param.name, param_type)

    def _infer_parameter_type(self, value: Any) -> str:
        """Infer the type of a parameter value."""
        if isinstance(value, int):
            return "int"
        elif isinstance(value, float):
            return "float"
        elif isinstance(value, str):
            return "str"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, list):
            return "list"
        elif isinstance(value, dict):
            return "dict"
        else:
            return "auto"

    def _analyze_publishers(self, publishers: List[PublisherNode]):
        """Analyze publisher configurations."""
        topics = set()
        
        for pub in publishers:
            # Check for duplicate topics within this node only
            if pub.topic in topics:
                self.errors.append(f"Duplicate publisher topic: {pub.topic}")
            topics.add(pub.topic)
            
            # Check topic name
            if not pub.topic or pub.topic.strip() == "":
                self.errors.append("Publisher topic cannot be empty")
            
            # Check message type
            if not pub.msg_type or pub.msg_type.strip() == "":
                self.errors.append(f"Message type cannot be empty for publisher {pub.topic}")
            
            # Add to symbol table (but don't check for global duplicates)
            self.symbol_table.topics.add(pub.topic)
            
            # Type check: msg_type is a valid ROS type (basic check)
            if not self._is_valid_ros_type(pub.msg_type):
                self.errors.append(f"Publisher '{pub.topic}' has invalid message type '{pub.msg_type}'")
    
    def _analyze_subscribers(self, subscribers: List[SubscriberNode]):
        """Analyze subscriber configurations."""
        topics = set()
        
        for sub in subscribers:
            # Check for duplicate topics within this node only
            if sub.topic in topics:
                self.errors.append(f"Duplicate subscriber topic: {sub.topic}")
            topics.add(sub.topic)
            
            # Check topic name
            if not sub.topic or sub.topic.strip() == "":
                self.errors.append("Subscriber topic cannot be empty")
            
            # Check message type
            if not sub.msg_type or sub.msg_type.strip() == "":
                self.errors.append(f"Message type cannot be empty for subscriber {sub.topic}")
            
            # Add to symbol table (but don't check for global duplicates)
            self.symbol_table.topics.add(sub.topic)
            
            # Type check: msg_type is a valid ROS type (basic check)
            if not self._is_valid_ros_type(sub.msg_type):
                self.errors.append(f"Subscriber '{sub.topic}' has invalid message type '{sub.msg_type}'")
    
    def _analyze_services(self, services: List[ServiceNode]):
        """Analyze service configurations."""
        service_names = set()
        
        for srv in services:
            # Check for duplicate service names
            if srv.service in service_names:
                self.errors.append(f"Duplicate service name: {srv.service}")
            service_names.add(srv.service)
            
            # Check service name
            if not srv.service or srv.service.strip() == "":
                self.errors.append("Service name cannot be empty")
            
            # Check service type
            if not srv.srv_type or srv.srv_type.strip() == "":
                self.errors.append(f"Service type cannot be empty for service {srv.service}")
            
            # Add to symbol table
            self.symbol_table.add_service(srv.service)
            
            # Type check: srv_type is a valid ROS type (basic check)
            if not self._is_valid_ros_type(srv.srv_type):
                self.errors.append(f"Service '{srv.service}' has invalid service type '{srv.srv_type}'")
    
    def _analyze_actions(self, actions: List[ActionNode]):
        """Analyze action configurations."""
        action_names = set()
        
        for act in actions:
            # Check for duplicate action names
            if act.name in action_names:
                self.errors.append(f"Duplicate action name: {act.name}")
            action_names.add(act.name)
            
            # Check action name
            if not act.name or act.name.strip() == "":
                self.errors.append("Action name cannot be empty")
            
            # Check action type
            if not act.action_type or act.action_type.strip() == "":
                self.errors.append(f"Action type cannot be empty for action {act.name}")
            
            # Add to symbol table
            self.symbol_table.add_action(act.name)
            
            # Type check: action_type is a valid ROS type (basic check)
            if not self._is_valid_ros_type(act.action_type):
                self.errors.append(f"Action '{act.name}' has invalid action type '{act.action_type}'")
    
    def _analyze_clients(self, clients: List[ServiceNode]):
        """Analyze service client configurations."""
        client_names = set()
        
        for client in clients:
            # Check for duplicate client names
            if client.service in client_names:
                self.errors.append(f"Duplicate client name: {client.service}")
            client_names.add(client.service)
            
            # Check client name
            if not client.service or client.service.strip() == "":
                self.errors.append("Client name cannot be empty")
            
            # Check service type
            if not client.srv_type or client.srv_type.strip() == "":
                self.errors.append(f"Service type cannot be empty for client {client.service}")
            
            # Add to symbol table
            self.symbol_table.add_service(client.service)
            
            # Type check: srv_type is a valid ROS type (basic check)
            if not self._is_valid_ros_type(client.srv_type):
                self.errors.append(f"Client '{client.service}' has invalid service type '{client.srv_type}'")
    
    def _analyze_flags(self, flags: List[FlagNode]):
        """Analyze flag configurations."""
        flag_names = set()
        
        for flag in flags:
            # Check for duplicate flag names
            if flag.name in flag_names:
                self.errors.append(f"Duplicate flag name: {flag.name}")
            flag_names.add(flag.name)
            
            # Check flag name
            if not flag.name or flag.name.strip() == "":
                self.errors.append("Flag name cannot be empty")
            
            # Check flag value
            if flag.value is None:
                self.errors.append(f"Flag '{flag.name}' has no value")
            
            # Type check: flag value should be boolean
            if not isinstance(flag.value, bool):
                self.errors.append(f"Flag '{flag.name}' value must be a boolean")
    
    def _analyze_timers(self, timers: List[TimerNode]):
        """Analyze timer configurations."""
        timer_names = set()
        
        for timer in timers:
            # Check for duplicate timer names
            if timer.name in timer_names:
                self.errors.append(f"Duplicate timer callback name: {timer.name}")
            timer_names.add(timer.name)
            
            # Check timer period (only if it's a number)
            if isinstance(timer.period, (int, float)):
                if timer.period <= 0:
                    self.errors.append(f"Timer {timer.name} period must be positive")
                if timer.period < 0:
                    self.errors.append(f"Timer {timer.name} period cannot be negative")
            else:
                # If it's an expression or variable, skip the numeric check
                self.warnings.append(f"Timer {timer.name} period is an expression or variable; skipping numeric validation.")
            
            # Check callback name
            if not timer.name or timer.name.strip() == "":
                self.errors.append("Timer callback name cannot be empty")
    
    def _analyze_remappings(self, remappings: List[RemapNode]):
        """Analyze remapping rules."""
        for remap in remappings:
            # Check from topic
            if not remap.from_topic or remap.from_topic.strip() == "":
                self.errors.append("Remap 'from' topic cannot be empty")
            
            # Check to topic
            if not remap.to_topic or remap.to_topic.strip() == "":
                self.errors.append("Remap 'to' topic cannot be empty")
    
    def _analyze_qos_config(self, qos: QoSNode, context: str):
        """Analyze QoS configuration."""
        if not qos:
            return
        
        # Check for duplicate QoS settings
        setting_names = set()
        for setting in qos.settings:
            if setting.name in setting_names:
                self.errors.append(f"Duplicate QoS setting '{setting.name}' in {context}")
            setting_names.add(setting.name)
            
            # Strong type checking for QoS settings
            self._validate_qos_setting(setting.name, setting.value, context)
            
            # Check for large values that might cause performance issues
            if setting.name == "depth" and isinstance(setting.value, (int, float)) and setting.value > 100:
                self.warnings.append(f"Large QoS depth ({setting.value}) in {context} may cause memory issues")
            
            if setting.name == "lease_duration" and isinstance(setting.value, (int, float)) and setting.value > 30:
                self.warnings.append(f"Long QoS lease duration ({setting.value}s) in {context} may cause timeout issues")
    
    def _analyze_cpp_methods(self, cpp_methods, node_name: str):
        """Analyze enhanced C++ method configurations."""
        method_names = set()
        
        for method in cpp_methods:
            # Check for duplicate method names
            if method.name in method_names:
                self.errors.append(f"Node '{node_name}' has duplicate C++ method name: {method.name}")
            method_names.add(method.name)
            
            # Check method name
            if not method.name or method.name.strip() == "":
                self.errors.append(f"Node '{node_name}' C++ method name cannot be empty")
            
            # Check method name follows C++ naming conventions
            if not method.name.replace('_', '').isalnum():
                self.errors.append(f"Node '{node_name}' C++ method name '{method.name}' contains invalid characters")
            
            # Check input parameters
            input_names = set()
            for input_param in method.inputs:
                if not input_param.param_name or input_param.param_name.strip() == "":
                    self.errors.append(f"Node '{node_name}' C++ method '{method.name}' input parameter name cannot be empty")
                
                if input_param.param_name in input_names:
                    self.errors.append(f"Node '{node_name}' C++ method '{method.name}' has duplicate input parameter name: {input_param.param_name}")
                input_names.add(input_param.param_name)
                
                if not input_param.param_type or input_param.param_type.strip() == "":
                    self.errors.append(f"Node '{node_name}' C++ method '{method.name}' input parameter '{input_param.param_name}' type cannot be empty")
                
                # Validate C++ parameter types
                if not self._is_valid_cpp_type(input_param.param_type):
                    self.warnings.append(f"Node '{node_name}' C++ method '{method.name}' input parameter '{input_param.param_name}' has potentially invalid type '{input_param.param_type}'")
            
            # Check output parameters
            output_names = set()
            for output_param in method.outputs:
                if not output_param.param_name or output_param.param_name.strip() == "":
                    self.errors.append(f"Node '{node_name}' C++ method '{method.name}' output parameter name cannot be empty")
                
                if output_param.param_name in output_names:
                    self.errors.append(f"Node '{node_name}' C++ method '{method.name}' has duplicate output parameter name: {output_param.param_name}")
                output_names.add(output_param.param_name)
                
                if not output_param.param_type or output_param.param_type.strip() == "":
                    self.errors.append(f"Node '{node_name}' C++ method '{method.name}' output parameter '{output_param.param_name}' type cannot be empty")
                
                # Validate C++ parameter types
                if not self._is_valid_cpp_type(output_param.param_type):
                    self.warnings.append(f"Node '{node_name}' C++ method '{method.name}' output parameter '{output_param.param_name}' has potentially invalid type '{output_param.param_type}'")
            
            # Check for parameter name conflicts between inputs and outputs
            conflicts = input_names & output_names
            if conflicts:
                self.errors.append(f"Node '{node_name}' C++ method '{method.name}' has parameter name conflicts between inputs and outputs: {', '.join(conflicts)}")
            
            # Check method code
            if not method.code or method.code.strip() == "":
                self.errors.append(f"Node '{node_name}' C++ method '{method.name}' has empty code")
            
            # Basic C++ syntax validation
            if method.code:
                # Check for basic C++ syntax issues
                if '{' in method.code and '}' not in method.code:
                    self.warnings.append(f"Node '{node_name}' C++ method '{method.name}' may have unmatched braces")
                if ';' not in method.code and 'return' not in method.code and 'if' not in method.code and 'for' not in method.code and 'while' not in method.code:
                    self.warnings.append(f"Node '{node_name}' C++ method '{method.name}' may be missing statements")

    def _is_valid_cpp_type(self, type_name: str) -> bool:
        """Check if a type is valid for C++ method parameters."""
        # Strip const if present
        if type_name.strip().startswith("const "):
            type_name = type_name.strip()[6:].strip()
        
        # Basic C++ types
        cpp_types = {
            "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long",
            "float", "double", "char", "unsigned char", "short", "unsigned short",
            "bool", "void", "size_t", "uint32_t", "int32_t", "uint64_t", "int64_t",
            "std::string", "std::vector", "std::array", "std::map", "std::unordered_map",
            "std::shared_ptr", "std::unique_ptr", "std::weak_ptr"
        }
        
        # Check if it's a basic C++ type
        if type_name in cpp_types:
            return True
        
        # Check if it's a pointer type
        if type_name.endswith('*') or type_name.endswith('[]'):
            return True
        
        # Check if it's a template type (e.g., std::vector<int>)
        if '<' in type_name and '>' in type_name:
            return True
        
        # Check if it's a custom type (assume valid if it contains alphanumeric and underscores)
        if type_name.replace('_', '').replace(':', '').replace('<', '').replace('>', '').isalnum():
            return True
        
        return False

    def _analyze_cuda_kernel(self, kernel: KernelNode):
        """Analyze CUDA kernel configuration."""
        # Check kernel name
        if not kernel.name or kernel.name.strip() == "":
            self.errors.append("CUDA kernel name cannot be empty")
        
        content = kernel.content
        
        # Check block size
        if content.block_size:
            if len(content.block_size) != 3:
                self.errors.append(f"CUDA kernel '{kernel.name}' block size must have exactly 3 dimensions")
            for i, size in enumerate(content.block_size):
                if not isinstance(size, int) or size <= 0:
                    self.errors.append(f"CUDA kernel '{kernel.name}' block size dimension {i} must be a positive integer")
                if size == 0:
                    self.errors.append(f"CUDA kernel '{kernel.name}' block size dimension {i} cannot be zero")
                if size > 1024:  # CUDA limit
                    self.errors.append(f"CUDA kernel '{kernel.name}' block size dimension {i} exceeds CUDA limit of 1024")
        
        # Check grid size
        if content.grid_size:
            if len(content.grid_size) != 3:
                self.errors.append(f"CUDA kernel '{kernel.name}' grid size must have exactly 3 dimensions")
            for i, size in enumerate(content.grid_size):
                # Allow both integers and expressions (strings)
                if isinstance(size, int):
                    if size <= 0:
                        self.errors.append(f"CUDA kernel '{kernel.name}' grid size dimension {i} must be a positive integer")
                elif isinstance(size, str):
                    # Expression - validate that it's not empty
                    if not size.strip():
                        self.errors.append(f"CUDA kernel '{kernel.name}' grid size dimension {i} expression cannot be empty")
                else:
                    self.errors.append(f"CUDA kernel '{kernel.name}' grid size dimension {i} must be an integer or expression")
        
        # Check shared memory
        if content.shared_memory is not None:
            if not isinstance(content.shared_memory, int) or content.shared_memory < 0:
                self.errors.append(f"CUDA kernel '{kernel.name}' shared memory must be a non-negative integer")
            if content.shared_memory > 49152:  # Typical CUDA limit
                self.warnings.append(f"CUDA kernel '{kernel.name}' shared memory size may exceed device limits")
        
        # Check parameters
        if content.parameters:
            param_names = set()
            for param in content.parameters:
                # Check for duplicate parameter names
                if param.param_name in param_names:
                    self.errors.append(f"CUDA kernel '{kernel.name}' has duplicate parameter name: {param.param_name}")
                param_names.add(param.param_name)
                
                # Check parameter name
                if not param.param_name or param.param_name.strip() == "":
                    self.errors.append(f"CUDA kernel '{kernel.name}' parameter name cannot be empty")
                
                # Check parameter type
                if not param.param_type or param.param_type.strip() == "":
                    self.errors.append(f"CUDA kernel '{kernel.name}' parameter '{param.param_name}' type cannot be empty")
                
                # Validate CUDA parameter types
                if not self._is_valid_cuda_type(param.param_type):
                    self.errors.append(f"CUDA kernel '{kernel.name}' parameter '{param.param_name}' has invalid type '{param.param_type}'")
                
                # Check parameter direction
                if param.direction not in [KernelParameterDirection.IN, KernelParameterDirection.OUT, KernelParameterDirection.INOUT]:
                    self.errors.append(f"CUDA kernel '{kernel.name}' parameter '{param.param_name}' has invalid direction '{param.direction}'")
                
                # Check size expression
                if param.size_expr:
                    for expr in param.size_expr:
                        if not expr or expr.strip() == "":
                            self.errors.append(f"CUDA kernel '{kernel.name}' parameter '{param.param_name}' has empty size expression")
    
    def _is_valid_cuda_type(self, type_name: str) -> bool:
        """Check if a type is valid for CUDA kernel parameters."""
        # Strip const if present
        if type_name.strip().startswith("const "):
            type_name = type_name.strip()[6:].strip()
        
        # Basic CUDA types
        cuda_types = {
            "int", "unsigned int", "long", "unsigned long", "long long", "unsigned long long",
            "float", "double", "char", "unsigned char", "short", "unsigned short",
            "bool", "void", "size_t", "uint32_t", "int32_t", "uint64_t", "int64_t"
        }
        
        # Check if it's a basic CUDA type
        if type_name in cuda_types:
            return True
        
        # Check if it's a pointer type
        if type_name.endswith('*') or type_name.endswith('[]'):
            return True
        
        # Check if it's a custom type (assume valid if it contains alphanumeric and underscores)
        if type_name.replace('_', '').replace(':', '').isalnum():
            return True
        
        return False
    
    def _analyze_cross_references(self, ast: RoboDSLAST):
        """Analyze cross-references between components."""
        # Build comprehensive topic/service/action maps
        topic_publishers = {}  # topic -> list of (node_name, publisher)
        topic_subscribers = {}  # topic -> list of (node_name, subscriber)
        service_providers = {}  # service -> list of (node_name, service)
        service_clients = {}  # service -> list of (node_name, client)
        action_providers = {}  # action -> list of (node_name, action)
        action_clients = {}  # action -> list of (node_name, client)
        
        # Collect all topics, services, and actions
        for node in ast.nodes:
            # Collect publishers
            for pub in node.content.publishers:
                if pub.topic not in topic_publishers:
                    topic_publishers[pub.topic] = []
                topic_publishers[pub.topic].append((node.name, pub))
            
            # Collect subscribers
            for sub in node.content.subscribers:
                if sub.topic not in topic_subscribers:
                    topic_subscribers[sub.topic] = []
                topic_subscribers[sub.topic].append((node.name, sub))
            
            # Collect services
            for srv in node.content.services:
                if srv.service not in service_providers:
                    service_providers[srv.service] = []
                service_providers[srv.service].append((node.name, srv))
            
            # Collect service clients
            for client in node.content.clients:
                if client.service not in service_clients:
                    service_clients[client.service] = []
                service_clients[client.service].append((node.name, client))
            
            # Collect actions
            for act in node.content.actions:
                if act.name not in action_providers:
                    action_providers[act.name] = []
                action_providers[act.name].append((node.name, act))
        
        # Validate remaps
        self._validate_remaps(ast, topic_publishers, topic_subscribers)
        
        # Validate topic compatibility
        self._validate_topic_compatibility(topic_publishers, topic_subscribers)
        
        # Validate service compatibility
        self._validate_service_compatibility(service_providers, service_clients)
        
        # Validate action compatibility
        self._validate_action_compatibility(action_providers, action_clients)
    
    def _validate_remaps(self, ast: RoboDSLAST, topic_publishers: dict, topic_subscribers: dict):
        """Validate remap rules."""
        for node in ast.nodes:
            for remap in node.content.remaps:
                # Check if the remapped topic exists in the current configuration
                if remap.from_topic and remap.from_topic not in topic_publishers and remap.from_topic not in topic_subscribers:
                    self.warnings.append(f"Node '{node.name}' remap '{remap.from_topic}' -> '{remap.to_topic}' references topic '{remap.from_topic}' that is not defined in the current configuration (may be external)")
                
                # Check for circular remaps (basic check)
                if remap.from_topic == remap.to_topic:
                    self.errors.append(f"Node '{node.name}' has circular remap: '{remap.from_topic}' -> '{remap.to_topic}'")
                
                # Check if remap target is valid
                if remap.to_topic and not remap.to_topic.startswith('/'):
                    self.warnings.append(f"Node '{node.name}' remap target '{remap.to_topic}' should start with '/'")
    
    def _validate_topic_compatibility(self, topic_publishers: dict, topic_subscribers: dict):
        """Validate topic compatibility between publishers and subscribers."""
        all_topics = set(topic_publishers.keys()) | set(topic_subscribers.keys())
        
        for topic in all_topics:
            publishers = topic_publishers.get(topic, [])
            subscribers = topic_subscribers.get(topic, [])
            
            # All publishers should have the same message type (check even if no subscribers)
            if publishers:
                pub_types = set(pub.msg_type for _, pub in publishers)
                if len(pub_types) > 1:
                    pub_nodes = [node_name for node_name, _ in publishers]
                    msg = f"Topic '{topic}' has multiple publishers with different message types: {pub_types} from nodes {pub_nodes}"
                    if self.debug:
                        print("DEBUG: ", msg)
                    self.errors.append(msg)
            
            # Check for subscribers without publishers (warn, not error - could be external)
            if subscribers and not publishers:
                sub_nodes = [node_name for node_name, _ in subscribers]
                self.warnings.append(f"Topic '{topic}' is subscribed to by nodes {sub_nodes} but not published by any node in the current configuration")
            
            # Check message type compatibility between publishers and subscribers
            if publishers and subscribers:
                pub_types = set(pub.msg_type for _, pub in publishers)
                sub_types = set(sub.msg_type for _, sub in subscribers)
                
                # All subscribers should have the same message type as publishers
                if pub_types and sub_types and not pub_types.intersection(sub_types):
                    self.errors.append(f"Topic '{topic}' has incompatible message types: publishers use {pub_types}, subscribers use {sub_types}")
                
                # Check QoS compatibility
                self._validate_topic_qos_compatibility(topic, publishers, subscribers)
    
    def _validate_topic_qos_compatibility(self, topic: str, publishers: list, subscribers: list):
        """Validate QoS compatibility between publishers and subscribers on the same topic."""
        for _, pub in publishers:
            for _, sub in subscribers:
                if pub.qos and sub.qos:
                    # Check reliability compatibility
                    if pub.qos.reliability and sub.qos.reliability:
                        if pub.qos.reliability != sub.qos.reliability:
                            self.warnings.append(f"Topic '{topic}' has QoS reliability mismatch: publisher uses '{pub.qos.reliability}', subscriber uses '{sub.qos.reliability}'")
                    
                    # Check durability compatibility
                    if pub.qos.durability and sub.qos.durability:
                        if pub.qos.durability != sub.qos.durability:
                            self.warnings.append(f"Topic '{topic}' has QoS durability mismatch: publisher uses '{pub.qos.durability}', subscriber uses '{sub.qos.durability}'")
    
    def _validate_service_compatibility(self, service_providers: dict, service_clients: dict):
        """Validate service compatibility."""
        all_services = set(service_providers.keys()) | set(service_clients.keys())
        
        for service in all_services:
            providers = service_providers.get(service, [])
            clients = service_clients.get(service, [])
            
            # Check for clients without providers
            if clients and not providers:
                client_nodes = [node_name for node_name, _ in clients]
                self.warnings.append(f"Service '{service}' is used by clients in nodes {client_nodes} but not provided by any node in the current configuration")
            
            # Check for multiple providers (error - services should have only one provider)
            if len(providers) > 1:
                provider_nodes = [node_name for node_name, _ in providers]
                self.errors.append(f"Service '{service}' is provided by multiple nodes: {provider_nodes}")
            
            # Check service type compatibility
            if providers and clients:
                provider_types = set(srv.srv_type for _, srv in providers)
                client_types = set(client.srv_type for _, client in clients)
                
                if provider_types and client_types and not provider_types.intersection(client_types):
                    self.errors.append(f"Service '{service}' has incompatible types: provider uses {provider_types}, clients use {client_types}")
    
    def _validate_action_compatibility(self, action_providers: dict, action_clients: dict):
        """Validate action compatibility."""
        all_actions = set(action_providers.keys()) | set(action_clients.keys())
        
        for action in all_actions:
            providers = action_providers.get(action, [])
            clients = action_clients.get(action, [])
            
            # Check for clients without providers
            if clients and not providers:
                client_nodes = [node_name for node_name, _ in clients]
                self.warnings.append(f"Action '{action}' is used by clients in nodes {client_nodes} but not provided by any node in the current configuration")
            
            # Check for multiple providers (error - actions should have only one provider)
            if len(providers) > 1:
                provider_nodes = [node_name for node_name, _ in providers]
                self.errors.append(f"Action '{action}' is provided by multiple nodes: {provider_nodes}")
            
            # Check action type compatibility
            if providers and clients:
                provider_types = set(act.action_type for _, act in providers)
                client_types = set(client.action_type for _, client in clients)
                
                if provider_types and client_types and not provider_types.intersection(client_types):
                    self.errors.append(f"Action '{action}' has incompatible types: provider uses {provider_types}, clients use {client_types}")
    
    def get_errors(self) -> List[str]:
        """Get list of semantic errors."""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get list of semantic warnings."""
        return self.warnings.copy()
    
    def has_errors(self) -> bool:
        """Check if there are any semantic errors."""
        return len(self.errors) > 0
    
    def has_warnings(self) -> bool:
        """Check if there are any semantic warnings."""
        return len(self.warnings) > 0
    
    def _is_valid_ros_type(self, msg_type: str) -> bool:
        """Check if a ROS message type is valid."""
        if not msg_type or msg_type.strip() == "":
            return False
        
        # Accept common ROS message type formats:
        # 1. package/msg/Type (standard format)
        # 2. package/Type (short format)
        # 3. package/srv/Type (service format)
        # 4. package/action/Type (action format)
        # 5. Just Type (basic format)
        
        parts = msg_type.split('/')
        
        # Handle different formats
        if len(parts) == 3:
            # package/msg/Type or package/srv/Type or package/action/Type
            package, msg_type_category, type_name = parts
            if msg_type_category in ['msg', 'srv', 'action']:
                return bool(package and type_name and type_name[0].isupper())
        elif len(parts) == 2:
            # package/Type
            package, type_name = parts
            return bool(package and type_name and type_name[0].isupper())
        elif len(parts) == 1:
            # Just Type
            type_name = parts[0]
            return bool(type_name and type_name[0].isupper())
        
        return False
    
    def _validate_parameter_type(self, param_name: str, param_type: str, value: Any) -> None:
        """Validate parameter type consistency."""
        # Check if declared type matches value type
        if param_type == "int":
            if not isinstance(value, int):
                self.errors.append(f"Parameter '{param_name}' declared as 'int' but value '{value}' is not an integer")
        elif param_type == "float":
            if not isinstance(value, (int, float)):
                self.errors.append(f"Parameter '{param_name}' declared as 'float' but value '{value}' is not a number")
        elif param_type == "bool":
            if not isinstance(value, bool):
                self.errors.append(f"Parameter '{param_name}' declared as 'bool' but value '{value}' is not a boolean")
        elif param_type == "string":
            if not isinstance(value, str):
                self.errors.append(f"Parameter '{param_name}' declared as 'string' but value '{value}' is not a string")
        elif param_type == "list":
            if not isinstance(value, list):
                self.errors.append(f"Parameter '{param_name}' declared as 'list' but value '{value}' is not a list")
        elif param_type == "dict":
            if not isinstance(value, dict):
                self.errors.append(f"Parameter '{param_name}' declared as 'dict' but value '{value}' is not a dictionary")
    
    def _validate_qos_setting(self, setting_name: str, setting_value: Any, context: str) -> None:
        """Validate QoS setting values."""
        if setting_name == "reliability":
            valid_string_values = ["reliable", "best_effort"]
            valid_numeric_values = [1, 2]  # 1=reliable, 2=best_effort
            if setting_value not in valid_string_values and setting_value not in valid_numeric_values:
                self.errors.append(f"QoS reliability in {context} must be one of {valid_string_values} or {valid_numeric_values}, got '{setting_value}'")
        
        elif setting_name == "durability":
            valid_string_values = ["volatile", "transient_local"]
            valid_numeric_values = [1, 2]  # 1=volatile, 2=transient_local
            if setting_value not in valid_string_values and setting_value not in valid_numeric_values:
                self.errors.append(f"QoS durability in {context} must be one of {valid_string_values} or {valid_numeric_values}, got '{setting_value}'")
        
        elif setting_name == "history":
            valid_string_values = ["keep_last", "keep_all"]
            valid_numeric_values = [1, 2]  # 1=keep_last, 2=keep_all
            if setting_value not in valid_string_values and setting_value not in valid_numeric_values:
                self.errors.append(f"QoS history in {context} must be one of {valid_string_values} or {valid_numeric_values}, got '{setting_value}'")
        
        elif setting_name == "liveliness":
            valid_string_values = ["automatic", "manual_by_topic", "manual_by_node"]
            valid_numeric_values = [1, 2, 3]  # 1=automatic, 2=manual_by_topic, 3=manual_by_node
            if setting_value not in valid_string_values and setting_value not in valid_numeric_values:
                self.errors.append(f"QoS liveliness in {context} must be one of {valid_string_values} or {valid_numeric_values}, got '{setting_value}'")
        
        elif setting_name in ["depth", "lease_duration", "deadline"]:
            if not isinstance(setting_value, (int, float)) or setting_value <= 0:
                self.errors.append(f"QoS {setting_name} in {context} must be a positive number, got '{setting_value}'") 