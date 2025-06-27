"""Semantic Analyzer for RoboDSL.

This module validates the semantic correctness of parsed RoboDSL configurations,
checking for errors that can't be caught by the grammar alone.
"""

from typing import List, Set, Dict, Any

from ..ast import (
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


class SemanticAnalyzer:
    """Analyzes semantic correctness of RoboDSL configurations."""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def analyze(self, ast: RoboDSLAST) -> bool:
        """Analyze the AST for semantic errors.
        
        Returns:
            True if no errors found, False otherwise.
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Analyze project-level configuration
        self._analyze_project_config(ast)
        
        # Analyze each node
        for node in ast.nodes:
            self._analyze_node(node)
        
        # Analyze CUDA kernels
        if ast.cuda_kernels:
            for kernel in ast.cuda_kernels.kernels:
                self._analyze_cuda_kernel(kernel)
        
        # Check for cross-references
        self._analyze_cross_references(ast)
        
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
        self._analyze_parameters(content.parameters)
        
        # Check publishers
        self._analyze_publishers(content.publishers)
        
        # Check subscribers
        self._analyze_subscribers(content.subscribers)
        
        # Check services
        self._analyze_services(content.services)
        
        # Check actions
        self._analyze_actions(content.actions)
        
        # Check timers
        self._analyze_timers(content.timers)
        
        # Check remappings
        self._analyze_remappings(content.remaps)
        
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
    
    def _analyze_parameters(self, parameters: List[ParameterNode]):
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
    
    def _analyze_publishers(self, publishers: List[PublisherNode]):
        """Analyze publisher configurations."""
        topics = set()
        
        for pub in publishers:
            # Check for duplicate topics
            if pub.topic in topics:
                self.errors.append(f"Duplicate publisher topic: {pub.topic}")
            topics.add(pub.topic)
            
            # Check topic name
            if not pub.topic or pub.topic.strip() == "":
                self.errors.append("Publisher topic cannot be empty")
            
            # Check message type
            if not pub.msg_type or pub.msg_type.strip() == "":
                self.errors.append(f"Message type cannot be empty for publisher {pub.topic}")
    
    def _analyze_subscribers(self, subscribers: List[SubscriberNode]):
        """Analyze subscriber configurations."""
        topics = set()
        
        for sub in subscribers:
            # Check for duplicate topics
            if sub.topic in topics:
                self.errors.append(f"Duplicate subscriber topic: {sub.topic}")
            topics.add(sub.topic)
            
            # Check topic name
            if not sub.topic or sub.topic.strip() == "":
                self.errors.append("Subscriber topic cannot be empty")
            
            # Check message type
            if not sub.msg_type or sub.msg_type.strip() == "":
                self.errors.append(f"Message type cannot be empty for subscriber {sub.topic}")
    
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
    
    def _analyze_timers(self, timers: List[TimerNode]):
        """Analyze timer configurations."""
        timer_names = set()
        
        for timer in timers:
            # Check for duplicate timer names
            if timer.name in timer_names:
                self.errors.append(f"Duplicate timer callback name: {timer.name}")
            timer_names.add(timer.name)
            
            # Check timer period
            if timer.period <= 0:
                self.errors.append(f"Timer {timer.name} period must be positive")
            
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
        for setting in qos.settings:
            if setting.name == 'reliability':
                if setting.value not in ['reliable', 'best_effort']:
                    self.errors.append(f"Invalid reliability value '{setting.value}' for {context}. "
                                     "Valid values: 'reliable', 'best_effort'")
            elif setting.name == 'durability':
                if setting.value not in ['transient_local', 'volatile']:
                    self.errors.append(f"Invalid durability value '{setting.value}' for {context}. "
                                     "Valid values: 'transient_local', 'volatile'")
            elif setting.name == 'history':
                if setting.value not in ['keep_last', 'keep_all']:
                    self.errors.append(f"Invalid history value '{setting.value}' for {context}. "
                                     "Valid values: 'keep_last', 'keep_all'")
            elif setting.name == 'depth':
                if not isinstance(setting.value, (int, float)) or setting.value <= 0:
                    self.errors.append(f"QoS depth must be positive for {context}")
            elif setting.name == 'deadline':
                if not isinstance(setting.value, (int, float)) or setting.value <= 0:
                    self.errors.append(f"QoS deadline must be positive for {context}")
            elif setting.name == 'lifespan':
                if not isinstance(setting.value, (int, float)) or setting.value <= 0:
                    self.errors.append(f"QoS lifespan must be positive for {context}")
            elif setting.name == 'liveliness':
                if setting.value not in ['automatic', 'manual_by_topic']:
                    self.errors.append(f"Invalid liveliness value '{setting.value}' for {context}. "
                                     "Valid values: 'automatic', 'manual_by_topic'")
            elif setting.name == 'liveliness_lease_duration':
                if not isinstance(setting.value, (int, float)) or setting.value <= 0:
                    self.errors.append(f"QoS liveliness lease duration must be positive for {context}")
    
    def _analyze_cuda_kernel(self, kernel: KernelNode):
        """Analyze CUDA kernel configuration."""
        # Check kernel name
        if not kernel.name or kernel.name.strip() == "":
            self.errors.append("CUDA kernel name cannot be empty")
        
        content = kernel.content
        
        # Check block size
        if content.block_size:
            if len(content.block_size) != 3:
                self.errors.append(f"CUDA kernel {kernel.name} block size must have 3 dimensions")
            
            for i, size in enumerate(content.block_size):
                if size <= 0:
                    self.errors.append(f"CUDA kernel {kernel.name} block size dimension {i} must be positive")
        
        # Check grid size if specified
        if content.grid_size:
            if len(content.grid_size) != 3:
                self.errors.append(f"CUDA kernel {kernel.name} grid size must have 3 dimensions")
            
            for i, size in enumerate(content.grid_size):
                if size <= 0:
                    self.errors.append(f"CUDA kernel {kernel.name} grid size dimension {i} must be positive")
        
        # Check shared memory
        if content.shared_memory is not None and content.shared_memory < 0:
            self.errors.append(f"CUDA kernel {kernel.name} shared memory cannot be negative")
        
        # Check parameters
        param_names = set()
        for param in content.parameters:
            if param.param_name and param.param_name in param_names:
                self.errors.append(f"Duplicate parameter name '{param.param_name}' in CUDA kernel {kernel.name}")
            if param.param_name:
                param_names.add(param.param_name)
            
            # Check parameter type
            if not param.param_type or param.param_type.strip() == "":
                self.errors.append(f"CUDA kernel {kernel.name} parameter type cannot be empty")
    
    def _analyze_cross_references(self, ast: RoboDSLAST):
        """Analyze cross-references between components."""
        # Check for topic consistency between publishers and subscribers
        all_pub_topics = set()
        all_sub_topics = set()
        
        for node in ast.nodes:
            for pub in node.content.publishers:
                all_pub_topics.add(pub.topic)
            for sub in node.content.subscribers:
                all_sub_topics.add(sub.topic)
        
        # Warn about topics that are published but not subscribed
        for topic in all_pub_topics - all_sub_topics:
            self.warnings.append(f"Topic '{topic}' is published but never subscribed to")
        
        # Warn about topics that are subscribed but not published
        for topic in all_sub_topics - all_pub_topics:
            self.warnings.append(f"Topic '{topic}' is subscribed to but never published")
    
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