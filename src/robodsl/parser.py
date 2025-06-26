"""Parser for the RoboDSL configuration files."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Union
import re

@dataclass
class QoSConfig:
    """Quality of Service configuration for ROS 2 communication."""
    # Reliability options: 'reliable' or 'best_effort'
    reliability: Optional[str] = None
    # Durability options: 'transient_local' or 'volatile'
    durability: Optional[str] = None
    # History policy: 'keep_last' or 'keep_all'
    history: Optional[str] = None
    # Queue depth (for 'keep_last' history policy)
    depth: int = 10
    # Deadline duration in milliseconds
    deadline: Optional[int] = None
    # Lifespan duration in milliseconds
    lifespan: Optional[int] = None
    # Liveliness policy: 'automatic' or 'manual_by_topic'
    liveliness: Optional[str] = None
    # Liveliness lease duration in milliseconds
    liveliness_lease_duration: Optional[int] = None

@dataclass
class PublisherConfig:
    """Configuration for a ROS 2 publisher."""
    topic: str
    msg_type: str
    qos: Optional[QoSConfig] = None
    queue_size: int = 10

@dataclass
class SubscriberConfig:
    """Configuration for a ROS 2 subscriber."""
    topic: str
    msg_type: str
    qos: Optional[QoSConfig] = None
    queue_size: int = 10

@dataclass
class ServiceConfig:
    """Configuration for a ROS 2 service server."""
    service: str
    srv_type: str
    qos: Optional[QoSConfig] = None

@dataclass
class ActionConfig:
    """Configuration for a ROS 2 action server."""
    name: str
    action_type: str
    qos: Optional[QoSConfig] = None

@dataclass
class ParameterConfig:
    """Configuration for a node parameter.
    
    Attributes:
        name: Name of the parameter
        type: Type of the parameter (e.g., 'int', 'double', 'string', 'bool')
        default: Default value for the parameter
        description: Optional description of the parameter
        read_only: Whether the parameter is read-only
    """
    name: str
    type: str
    default: Any
    description: str = ""
    read_only: bool = False

@dataclass
class LifecycleConfig:
    """Configuration for node lifecycle settings.
    
    Attributes:
        enabled: Whether lifecycle is enabled
        autostart: Whether to automatically transition to active state
        cleanup_on_shutdown: Whether to perform cleanup on shutdown
    """
    enabled: bool = False
    autostart: bool = True
    cleanup_on_shutdown: bool = True

@dataclass
class TimerConfig:
    """Configuration for a node timer.
    
    Attributes:
        period: Timer period in seconds (can be float for sub-second periods)
        callback: Name of the callback function to invoke
        oneshot: If True, the timer will only fire once
        autostart: If True, the timer will start automatically
    """
    period: float
    callback: str
    oneshot: bool = False
    autostart: bool = True

@dataclass
class RemapRule:
    """Configuration for a topic remapping rule.
    
    Attributes:
        from_topic: Original topic name
        to_topic: New topic name to remap to
    """
    from_topic: str
    to_topic: str

@dataclass
class CppMethod:
    """Represents a standard C++ method defined inside a RoboDSL node."""
    return_type: str
    name: str
    params: str  # Raw parameter list string (kept verbatim for now)
    body: str


@dataclass
class NodeConfig:
    """Configuration for a ROS 2 node (extended)."""
    name: str

    # Messaging primitives
    publishers: List[PublisherConfig] = field(default_factory=list)
    subscribers: List[SubscriberConfig] = field(default_factory=list)
    services: List[ServiceConfig] = field(default_factory=list)
    actions: List[ActionConfig] = field(default_factory=list)

    # Parameters and configuration
    parameters: List[ParameterConfig] = field(default_factory=list)
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    timers: List[TimerConfig] = field(default_factory=list)
    namespace: str = ""
    remap: List[RemapRule] = field(default_factory=list)
    
    # Parameter callback handling
    parameter_callbacks: bool = False

    # --- New C++ CPU extensions ---
    cpp_helpers: List[str] = field(default_factory=list)
    methods: List[CppMethod] = field(default_factory=list)
    
    # Backward compatibility for code that might access these
    @property
    def qos_profiles(self) -> Dict[str, Any]:
        """For backward compatibility, return an empty dict."""
        return {}
        
    # Backward compatibility for old dictionary-based parameters
    @property
    def parameters_dict(self) -> Dict[str, Any]:
        """For backward compatibility, return parameters as a dictionary."""
        return {param.name: param.default for param in self.parameters}

@dataclass
class KernelParameter:
    name: str
    type: str
    direction: Literal["in", "out", "inout"]  # optional
    is_const: bool = False
    is_pointer: bool = False
    size_expr: Optional[str] = None  # For dynamic arrays

@dataclass
class CudaKernelConfig:
    """Configuration for a CUDA kernel.
    
    Attributes:
        name: Name of the kernel
        inputs: List of input parameters with their types
        outputs: List of output parameters with their types
        block_size: 3D block dimensions (x, y, z)
        grid_size: 3D grid dimensions (x, y, z). If None, will be calculated
        shared_mem_bytes: Bytes of shared memory to allocate per block
        use_thrust: Whether to include Thrust headers and support
        code: The actual CUDA kernel code
        includes: List of additional CUDA/Thrust includes needed
        defines: Preprocessor definitions for the kernel
    """
    name: str
    parameters: List[KernelParameter]
    block_size: tuple = (256, 1, 1)  # Default block size (1D)
    grid_size: Optional[tuple] = None  # Auto-calculated if None
    shared_mem_bytes: int = 0
    use_thrust: bool = False
    code: str = ""
    includes: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)

@dataclass
class RoboDSLConfig:
    """Top-level configuration for a RoboDSL project.
    
    Attributes:
        project_name: Name of the project
        nodes: List of node configurations
        cuda_kernels: List of CUDA kernel configurations
        includes: Set of include paths used in the project
    """
    project_name: str = "robodsl_project"  # Default project name
    nodes: List[NodeConfig] = field(default_factory=list)
    cuda_kernels: List[CudaKernelConfig] = field(default_factory=list)
    includes: set[str] = field(default_factory=set)  # Track all includes across the project

def parse_robodsl(content: str) -> RoboDSLConfig:
    """Parse a RoboDSL configuration file.
    
    Args:
        content: The content of the .robodsl file
        
    Returns:
        RoboDSLConfig: The parsed configuration
    """
    config = RoboDSLConfig()
    
    # Track all includes in the file
    includes = set()
    
    # Look for include statements
    include_matches = re.findall(r'^\s*include\s+[<"]([^>"]+)[>"]', content, re.MULTILINE)
    includes.update(include_matches)
    
    # Add to config
    config.includes = includes
    
    # Remove comments
    content = re.sub(r'#.*', '', content)
    
    # Parse node configurations
    node_pattern = r'node\s+(\w+)\s*\{([^}]*)\}'
    for match in re.finditer(node_pattern, content, re.DOTALL):
        node_name = match.group(1)
        node_content = match.group(2).strip()
        node = NodeConfig(name=node_name)
        
        # Parse QoS settings from a string to QoSConfig object
        def parse_qos(qos_str: Optional[str]) -> Optional[QoSConfig]:
            if not qos_str or not qos_str.strip():
                return None
                
            qos_dict = {}
            for part in qos_str.split():
                if '=' in part:
                    key, value = part.split('=', 1)
                    # Convert numeric values to int/float if possible
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                        value = float(value)
                    qos_dict[key] = value
            
            # Map the dictionary to QoSConfig fields
            return QoSConfig(
                reliability=qos_dict.get('reliability'),
                durability=qos_dict.get('durability'),
                history=qos_dict.get('history'),
                depth=qos_dict.get('depth', 10),
                deadline=qos_dict.get('deadline'),
                lifespan=qos_dict.get('lifespan'),
                liveliness=qos_dict.get('liveliness'),
                liveliness_lease_duration=qos_dict.get('liveliness_lease_duration')
            ) if qos_dict else None
            
        # Parse parameters
        param_matches = re.finditer(r'parameter\s+(\w+)\s+(\w+)(?:\s*=\s*([^\s]+))?(?:\s*#\s*(.*))?', node_content)
        for param in param_matches:
            param_name = param.group(1)
            param_type = param.group(2)
            param_default = param.group(3) if param.lastindex >= 3 else None
            param_desc = param.group(4).strip() if param.lastindex >= 4 and param.group(4) else ""
            
            # Convert default value to appropriate type
            default_value = None
            if param_default is not None:
                try:
                    if param_type == 'int':
                        default_value = int(param_default)
                    elif param_type == 'double':
                        default_value = float(param_default)
                    elif param_type == 'bool':
                        default_value = param_default.lower() in ('true', '1', 'yes')
                    else:  # string or other types
                        default_value = param_default.strip('"\'')
                except ValueError:
                    default_value = param_default
            
            node.parameters.append(ParameterConfig(
                name=param_name,
                type=param_type,
                default=default_value,
                description=param_desc
            ))
            
        # Parse C++ helper blocks (cpp { ... })
        for cpp_match in re.finditer(r'cpp\s*\{([^}]*)\}', node_content, re.DOTALL):
            node.cpp_helpers.append(cpp_match.group(1).strip())

        # Parse C++ method blocks (method <sig> { ... })
        method_pattern = re.compile(r'method\s+([^\{]+)\{([^}]*)\}', re.DOTALL)
        signature_re = re.compile(r'([\w:<>]+)\s+(\w+)\s*\(([^)]*)\)')
        for m in method_pattern.finditer(node_content):
            signature = m.group(1).strip()
            body = m.group(2).strip()
            sig_match = signature_re.match(signature)
            if not sig_match:
                continue  # Skip invalid signatures
            return_type, method_name, params = sig_match.groups()
            node.methods.append(CppMethod(return_type=return_type,
                                         name=method_name,
                                         params=params.strip(),
                                         body=body))

        # Parse lifecycle settings
        lifecycle_match = re.search(r'lifecycle\s*\{([^}]*)\}', node_content, re.DOTALL)
        if lifecycle_match:
            lifecycle_content = lifecycle_match.group(1)
            lifecycle = LifecycleConfig(enabled=True)
            
            # Parse autostart
            autostart_match = re.search(r'autostart\s*[:=]\s*(true|false|1|0)', lifecycle_content, re.IGNORECASE)
            if autostart_match:
                lifecycle.autostart = autostart_match.group(1).lower() in ('true', '1')
                
            # Parse cleanup_on_shutdown
            cleanup_match = re.search(r'cleanup_on_shutdown\s*[:=]\s*(true|false|1|0)', lifecycle_content, re.IGNORECASE)
            if cleanup_match:
                lifecycle.cleanup_on_shutdown = cleanup_match.group(1).lower() in ('true', '1')
                
            node.lifecycle = lifecycle
            
        # Parse timers
        timer_matches = re.finditer(r'timer\s+([\w_]+)\s*:\s*([0-9.]+)(?:\s*\{([^}]*)\})?', node_content)
        for timer in timer_matches:
            timer_name = timer.group(1)
            period = float(timer.group(2))
            timer_config = timer.group(3) if timer.lastindex == 3 else ""
            
            # Default timer config
            oneshot = False
            autostart = True
            
            # Parse timer configuration if present
            if timer_config:
                oneshot_match = re.search(r'oneshot\s*[:=]\s*(true|false|1|0)', timer_config, re.IGNORECASE)
                if oneshot_match:
                    oneshot = oneshot_match.group(1).lower() in ('true', '1')
                    
                autostart_match = re.search(r'autostart\s*[:=]\s*(true|false|1|0)', timer_config, re.IGNORECASE)
                if autostart_match:
                    autostart = autostart_match.group(1).lower() in ('true', '1')
            
            node.timers.append(TimerConfig(
                period=period,
                callback=timer_name,
                oneshot=oneshot,
                autostart=autostart
            ))
            
        # Parse remappings
        remap_matches = re.finditer(r'remap\s+from\s*[:=]\s*([\w/]+)\s+to\s*[:=]\s*([\w/]+)', node_content)
        for remap in remap_matches:
            node.remap.append(RemapRule(
                from_topic=remap.group(1),
                to_topic=remap.group(2)
            ))
            
        # Parse namespace if specified
        namespace_match = re.search(r'namespace\s*[:=]\s*([\w/]+)', node_content)
        if namespace_match:
            node.namespace = namespace_match.group(1)
            
        # Check if parameter callbacks are enabled
        if re.search(r'enable_parameter_callbacks\s*[:=]\s*(true|1|yes)', node_content, re.IGNORECASE):
            node.parameter_callbacks = True
            
        # Parse publishers
        pub_matches = re.finditer(r'publisher\s+([\w/]+)\s+([\w/.]+)(?:\s+qos\s+([^\n]+))?', node_content)
        for pub in pub_matches:
            topic = pub.group(1)
            msg_type = pub.group(2)
            qos = parse_qos(pub.group(3) if pub.lastindex == 3 else None)
            queue_size = 10  # Default queue size
            
            node.publishers.append(PublisherConfig(
                topic=topic,
                msg_type=msg_type,
                qos=qos,
                queue_size=queue_size
            ))
            
        # Parse subscribers
        sub_matches = re.finditer(r'subscriber\s+([\w/]+)\s+([\w/.]+)(?:\s+qos\s+([^\n]+))?', node_content)
        for sub in sub_matches:
            topic = sub.group(1)
            msg_type = sub.group(2)
            qos = parse_qos(sub.group(3) if sub.lastindex == 3 else None)
            queue_size = 10  # Default queue size
            
            node.subscribers.append(SubscriberConfig(
                topic=topic,
                msg_type=msg_type,
                qos=qos,
                queue_size=queue_size
            ))
            
        # Parse services
        srv_matches = re.finditer(r'service\s+([\w/]+)\s+([\w/.]+)(?:\s+qos\s+([^\n]+))?', node_content)
        for srv in srv_matches:
            service_name = srv.group(1)
            srv_type = srv.group(2)
            qos = parse_qos(srv.group(3) if srv.lastindex == 3 else None)
            
            node.services.append(ServiceConfig(
                service=service_name,
                srv_type=srv_type,
                qos=qos
            ))
            
        # Parse actions
        act_matches = re.finditer(r'action\s+([\w/]+)\s+([\w/.]+)(?:\s+qos\s+([^\n]+))?', node_content)
        for act in act_matches:
            action_name = act.group(1)
            action_type = act.group(2)
            qos = parse_qos(act.group(3) if act.lastindex == 3 else None)
            
            node.actions.append(ActionConfig(
                name=action_name,
                action_type=action_type,
                qos=qos
            ))

        # Parse timers
        timer_matches = re.finditer(r'timer\s+(\w+)\s+(\d+\.?\d*)(?:\s+([^\n]+))?', node_content)
        for timer in timer_matches:
            timer_name = timer.group(1)
            period = float(timer.group(2))
            callback = timer.group(3).strip() if timer.lastindex == 3 else f'on_timer_{timer_name}'
            
            node.timers.append(TimerConfig(
                period=period,
                callback=callback,
                oneshot=False,
                autostart=True
            ))

        # Parse parameters
        param_matches = re.finditer(r'parameter\s+(\w+)\s*[:=]\s*([^\n]+)', node_content)
        for param in param_matches:
            param_name = param.group(1)
            param_value = param.group(2).strip()
            
            # Try to evaluate the parameter value (handles numbers, lists, dicts, etc.)
            try:
                node.parameters.append(ParameterConfig(
                    name=param_name,
                    type='string',
                    default=eval(param_value),
                    description=""
                ))
            except (NameError, SyntaxError):
                # If evaluation fails, store as string
                node.parameters.append(ParameterConfig(
                    name=param_name,
                    type='string',
                    default=param_value.strip('\'"'),
                    description=""
                ))

        # Lifecycle flag
        if re.search(r'lifecycle\s*[:=]\s*true', node_content, re.IGNORECASE):
            node.lifecycle = LifecycleConfig(enabled=True)

        # Parameter callbacks flag
        if re.search(r'parameter_callbacks\s*[:=]\s*true', node_content, re.IGNORECASE):
            node.parameter_callbacks = True

        # Add the parsed node to the config
        config.nodes.append(node)
    
    def find_balanced_braces(text, start_pos=0):
        """Find the matching closing brace for an opening brace."""
        if start_pos >= len(text) or text[start_pos] != '{':
            return -1
        
        balance = 1
        pos = start_pos + 1
        
        while pos < len(text) and balance > 0:
            if text[pos] == '{':
                balance += 1
            elif text[pos] == '}':
                balance -= 1
            pos += 1
        
        return pos if balance == 0 else -1

    # Parse CUDA kernels section if it exists
    print("\nDEBUG: Looking for cuda_kernels section")
    kernel_section_start = content.find('cuda_kernels{')
    if kernel_section_start == -1:
        kernel_section_start = content.find('cuda_kernels {')
    
    if kernel_section_start != -1:
        # Find the opening brace of cuda_kernels
        kernel_section_start = content.find('{', kernel_section_start)
        if kernel_section_start == -1:
            print("DEBUG: Found 'cuda_kernels' but no opening brace")
            return config
            
        # Find the matching closing brace
        kernel_section_end = find_balanced_braces(content, kernel_section_start)
        if kernel_section_end == -1:
            print("DEBUG: Could not find matching closing brace for cuda_kernels")
            return config
            
        section_content = content[kernel_section_start + 1:kernel_section_end - 1]
        print(f"DEBUG: Kernel section content length: {len(section_content)} chars")
        
        # Find all kernel definitions
        kernel_pos = 0
        while True:
            # Find the next 'kernel' keyword
            kernel_match = re.search(r'kernel\s+(\w+)', section_content[kernel_pos:])
            if not kernel_match:
                break
                
            kernel_name = kernel_match.group(1)
            kernel_start = kernel_match.start() + kernel_pos
            content_start = kernel_match.end() + kernel_pos
            
            # Find the opening brace after the kernel name
            brace_pos = section_content.find('{', content_start)
            if brace_pos == -1:
                print(f"DEBUG: No opening brace found for kernel {kernel_name}")
                break
                
            # Find the matching closing brace
            closing_brace = find_balanced_braces(section_content, brace_pos)
            if closing_brace == -1:
                print(f"DEBUG: No matching closing brace for kernel {kernel_name}")
                break
                
            # Extract the kernel content
            kernel_content = section_content[brace_pos + 1:closing_brace - 1].strip()
            print(f"DEBUG: Found kernel: {kernel_name} with content length {len(kernel_content)}")
            
            # Process the kernel content
            block_size = (256, 1, 1)
            grid_size = None
            shared_mem_bytes = 0
            use_thrust = False
            parameters = []
            code = ""
            includes = []
            defines = {}
            
            # Parse block size
            block_match = re.search(r'block_size\s*[:=]\s*[\(\[]?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\)\]]?', kernel_content)
            if block_match:
                block_size = (int(block_match.group(1)), 
                             int(block_match.group(2)), 
                             int(block_match.group(3)))
            
            # Parse inputs and outputs
            for line in kernel_content.split('\n'):
                line = line.strip()
                if not line:
                    continue
                    
                # Match input: Type (attributes)
                input_match = re.match(r'input\s*:\s*(\w+)(?:\s*\(([^)]*)\))?', line)
                if input_match:
                    param_type = input_match.group(1)
                    attributes = input_match.group(2) if input_match.lastindex == 2 else None
                    param_name = f"input_{len([p for p in parameters if p.direction == 'in'])}"
                    
                    parameters.append(KernelParameter(
                        name=param_name,
                        type=param_type,
                        direction='in',
                        is_const=True,
                        is_pointer=True,
                        size_expr=attributes
                    ))
                    continue
                    
                # Match output: Type (attributes)
                output_match = re.match(r'output\s*:\s*(\w+)(?:\s*\(([^)]*)\))?', line)
                if output_match:
                    param_type = output_match.group(1)
                    attributes = output_match.group(2) if output_match.lastindex == 2 else None
                    param_name = f"output_{len([p for p in parameters if p.direction == 'out'])}"
                    
                    parameters.append(KernelParameter(
                        name=param_name,
                        type=param_type,
                        direction='out',
                        is_const=False,
                        is_pointer=True,
                        size_expr=attributes
                    ))
            
            # Create the kernel config
            config.cuda_kernels.append(CudaKernelConfig(
                name=kernel_name,
                parameters=parameters,
                block_size=block_size,
                grid_size=grid_size,
                shared_mem_bytes=shared_mem_bytes,
                use_thrust=use_thrust,
                code=code,
                includes=includes,
                defines=defines
            ))
            
            # Move past this kernel
            kernel_pos = closing_brace
        
        print(f"DEBUG: Found {len(config.cuda_kernels)} kernels")
    else:
        print("DEBUG: No cuda_kernels section found")

    return config

    return config
