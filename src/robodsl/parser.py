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
                if param_type == 'int':
                    default_value = int(param_default)
                elif param_type == 'double':
                    default_value = float(param_default)
                elif param_type == 'bool':
                    default_value = param_default.lower() in ('true', '1', 'yes')
                else:  # string or other types
                    default_value = param_default.strip('"\'')
            
            node.parameters.append(ParameterConfig(
                name=param_name,
                type=param_type,
                default=default_value,
                description=param_desc
            ))
            
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
            
            node.timers.append({
                'name': timer_name,
                'period': period,
                'callback': callback
            })

        # Parse parameters
        param_matches = re.finditer(r'parameter\s+(\w+)\s*[:=]\s*([^\n]+)', node_content)
        for param in param_matches:
            param_name = param.group(1)
            param_value = param.group(2).strip()
            
            # Try to evaluate the parameter value (handles numbers, lists, dicts, etc.)
            try:
                node.parameters[param_name] = eval(param_value)
            except (NameError, SyntaxError):
                # If evaluation fails, store as string
                node.parameters[param_name] = param_value.strip('\'"')

        # Lifecycle flag
        if re.search(r'lifecycle\s*[:=]\s*true', node_content, re.IGNORECASE):
            node.lifecycle = True

        # Parameter callbacks flag
        if re.search(r'parameter_callbacks\s*[:=]\s*true', node_content, re.IGNORECASE):
            node.parameter_callbacks = True

    # Parse CUDA kernels section if it exists
    kernel_section = re.search(r'cuda_kernels\s*\{([^}]*)\}', content, re.DOTALL)
    if kernel_section:
        kernel_blocks = re.finditer(r'kernel\s+(\w+)\s*\{([^}]*)\}', kernel_section.group(1), re.DOTALL)
        
        for kernel_match in kernel_blocks:
            kernel_name = kernel_match.group(1)
            kernel_content = kernel_match.group(2)
            
            # Default values
            block_size = (256, 1, 1)
            grid_size = None
            shared_mem_bytes = 0
            use_thrust = False
            params = []
            code = ""
            includes = []
            defines = {}

            # Parse block size
            block_match = re.search(r'block_size\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', kernel_content)
            if block_match:
                block_size = (int(block_match.group(1)), 
                             int(block_match.group(2)), 
                             int(block_match.group(3)))

            # Parse grid size
            grid_match = re.search(r'grid_size\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)', kernel_content)
            if grid_match:
                grid_size = (int(grid_match.group(1)),
                            int(grid_match.group(2)),
                            int(grid_match.group(3)))

            # Parse shared memory
            shared_match = re.search(r'shared_memory\s*=\s*(\d+)', kernel_content)
            if shared_match:
                shared_mem_bytes = int(shared_match.group(1))

            # Parse Thrust usage
            use_thrust = 'use_thrust' in kernel_content

            # Parse inputs
            input_pattern = r'input\s+([\w:]+)\s+(\w+)(?:\s*\[\s*(\w*)\s*\])?'
            for input_match in re.finditer(input_pattern, kernel_content):
                param_name = input_match.group(2)
                param_type = input_match.group(1)
                size_expr = input_match.group(3) if input_match.lastindex == 3 else None
                
                params.append(KernelParameter(
                    name=param_name,
                    type=param_type,
                    direction='in',
                    is_const=True,
                    is_pointer=True,
                    size_expr=size_expr
                ))

            # Parse outputs
            output_pattern = r'output\s+([\w:]+)\s+(\w+)(?:\s*\[\s*(\w*)\s*\])?'
            for output_match in re.finditer(output_pattern, kernel_content):
                param_name = output_match.group(2)
                param_type = output_match.group(1)
                size_expr = output_match.group(3) if output_match.lastindex == 3 else None
                
                params.append(KernelParameter(
                    name=param_name,
                    type=param_type,
                    direction='out',
                    is_const=False,
                    is_pointer=True,
                    size_expr=size_expr
                ))

            # Parse code block
            code_match = re.search(r'code\s*``[(?:cuda)?\s*([^](cci:1://file:///Users/ishayu/Documents/Random_projects/cuif/src/robodsl/generator.py:40:4-86:30)]+)```', kernel_content, re.DOTALL)
            if code_match:
                code = code_match.group(1).strip()

            # Parse includes
            includes = re.findall(r'include\s+[<"]([^">]+)[">]', kernel_content)

            # Parse defines
            define_matches = re.finditer(r'define\s+(\w+)(?:\s+(.*?))?$', kernel_content, re.MULTILINE)
            for define in define_matches:
                defines[define.group(1)] = define.group(2) or ""

            # Create and add the kernel config
            kernel_config = CudaKernelConfig(
                name=kernel_name,
                parameters=params,
                block_size=block_size,
                grid_size=grid_size,
                shared_mem_bytes=shared_mem_bytes,
                use_thrust=use_thrust,
                code=code,
                includes=includes,
                defines=defines
            )
            config.cuda_kernels.append(kernel_config)

    return config
    
    return config
