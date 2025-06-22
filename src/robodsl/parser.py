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
    default: Any = None
    description: str = ""
    read_only: bool = False
    
    @property
    def value(self):
        """Backward compatibility alias for default (used in tests)."""
        return self.default

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
class CppMethodConfig:
    """Configuration for a C++ method in a node.
    
    Attributes:
        name: Name of the method
        return_type: C++ return type of the method
        parameters: List of parameter declarations as strings
        implementation: The method body as a string
    """
    name: str
    return_type: str
    parameters: List[str]
    implementation: str

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
    methods: List[CppMethodConfig] = field(default_factory=list)
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
    direction: Literal["in", "out", "inout"]
    is_const: bool = False
    is_pointer: bool = False
    size_expr: Optional[str] = None  # For dynamic arrays
    metadata: Dict[str, Any] = field(default_factory=dict)  # For additional parameters like width, height, etc.

    def __post_init__(self):
        # Ensure metadata is always a dict, even if None is passed
        if self.metadata is None:
            self.metadata = {}

    def __getitem__(self, key):
        """Support dictionary-style access for backward compatibility.
        
        Note: For backward compatibility, numeric metadata values are converted to strings
        when accessed via dictionary-style access.
        """
        # First try direct attributes
        if hasattr(self, key):
            return getattr(self, key)
            
        # Then try metadata
        value = self.metadata.get(key)
        
        # For backward compatibility, convert numeric values to strings
        # when accessed via dictionary-style access
        if isinstance(value, (int, float)):
            return str(value)
            
        return value
        
    def get(self, key, default=None):
        """Support dictionary-style get method for backward compatibility."""
        # First try direct attributes, then metadata
        if hasattr(self, key):
            return getattr(self, key, default)
        return self.metadata.get(key, default)

@dataclass
class CudaKernelConfig:
    """Configuration for a CUDA kernel.
    
    Attributes:
        name: Name of the kernel
        parameters: List of kernel parameters
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
    block_size: tuple = (256, 1, 1)
    grid_size: Optional[tuple] = None
    shared_mem_bytes: int = 0
    use_thrust: bool = False
    code: str = ""
    includes: List[str] = field(default_factory=list)
    defines: Dict[str, str] = field(default_factory=dict)
    
    @property
    def inputs(self) -> List[KernelParameter]:
        """Get all input parameters (backward compatibility)."""
        return [p for p in self.parameters if p.direction == 'in']
    
    @property
    def outputs(self) -> List[KernelParameter]:
        """Get all output parameters (backward compatibility)."""
        return [p for p in self.parameters if p.direction == 'out']

@dataclass
class CppMethodConfig:
    """Configuration for a C++ method in a node.
    
    Attributes:
        name: Name of the method
        return_type: C++ return type of the method
        parameters: List of parameter declarations as strings
        implementation: The method body as a string
    """
    name: str
    return_type: str
    parameters: List[str]
    implementation: str

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
    print("=== Starting parse_robodsl ===")
    config = RoboDSLConfig()
    print(f"Created config object: {config}")
    
    # Track all includes in the file
    includes = set()
    
    # Look for include statements
    include_matches = re.findall(r'^\s*include\s+[<"]([^>"]+)[>"]', content, re.MULTILINE)
    includes.update(include_matches)
    
    # Remove comments
    content = re.sub(r'#.*$', '', content, flags=re.MULTILINE)
    
    # Find all top-level blocks (nodes and cuda_kernels)
    blocks = list(re.finditer(r'(node|cuda_kernels)\s+(\w+)?\s*\{', content))
    print(f"Initial config state - nodes: {len(config.nodes)}, cuda_kernels: {len(config.cuda_kernels)}")
    
    for block in blocks:
        block_type, name = block.groups()
        print(f"Found {block_type} block: {name if name else 'anonymous'}")
        
        # Find the matching closing brace
        start_idx = block.end()
        brace_count = 1
        end_idx = start_idx
        
        # Find the matching closing brace
        for i, c in enumerate(content[start_idx:], start=start_idx):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i
                    break
        
        if end_idx <= start_idx:
            print(f"Warning: Could not find matching closing brace for {block_type} {name}")
            continue
            
        block_content = content[start_idx:end_idx].strip()
        
        if block_type == 'node':
            # Parse node content
            node = parse_node(name, block_content)
            if node:
                config.nodes.append(node)
        elif block_type == 'cuda_kernels':
            # Parse CUDA kernels block
            kernel_matches = list(re.finditer(r'kernel\s+(\w+)\s*\{', block_content))
            for kernel_match in kernel_matches:
                kernel_name = kernel_match.group(1)
                kernel_start = kernel_match.end()
                kernel_brace_count = 1
                kernel_end = kernel_start
                
                # Find the matching closing brace for this kernel
                for i, c in enumerate(block_content[kernel_start:], start=kernel_start):
                    if c == '{':
                        kernel_brace_count += 1
                    elif c == '}':
                        kernel_brace_count -= 1
                        if kernel_brace_count == 0:
                            kernel_end = i
                            break
                
                if kernel_end <= kernel_start:
                    print(f"Warning: Could not find matching closing brace for kernel {kernel_name}")
                    continue
                
                kernel_content = block_content[kernel_start:kernel_end].strip()
                kernel_config = parse_cuda_kernel(kernel_name, kernel_content)
                if kernel_config:
                    config.cuda_kernels.append(kernel_config)
    
    # Update includes in the config
    config.includes.update(includes)
    
    return config

def parse_cuda_kernel(kernel_name: str, kernel_content: str) -> Optional[CudaKernelConfig]:
    """Parse a CUDA kernel configuration.
    
    Args:
        kernel_name: Name of the kernel
        kernel_content: Content of the kernel configuration
        
    Returns:
        CudaKernelConfig if successful, None otherwise
    """
    try:
        params = []
        block_size = (256, 1, 1)  # Default block size
        grid_size = None
        shared_mem_bytes = 0
        use_thrust = False
        code = ""
        includes = []
        defines = {}
        
        # Parse inputs (format: input: Type (param=value, ...))
        input_matches = re.finditer(r'input\s*:\s*(\w+)(?:\s*\(([^)]*)\))?', kernel_content)
        for input_match in input_matches:
            param_type = input_match.group(1)
            params_dict = {}
            if input_match.group(2):
                # Parse key=value pairs and store in metadata
                for kv in re.findall(r'(\w+)\s*=\s*([^,\s]+)', input_match.group(2)):
                    key = kv[0]
                    value = kv[1].strip('\'')
                    # Convert to int if it's a numeric value
                    if value.isdigit():
                        value = int(value)
                    params_dict[key] = value
            
            # Create parameter with metadata
            param = KernelParameter(
                name=f"input_{len([p for p in params if p.direction == 'in'])}",
                type=param_type,
                direction='in',
                is_const=True,
                is_pointer=True,
                metadata=params_dict
            )
            params.append(param)
        
        # Parse outputs (format: output: Type (param=value, ...))
        output_matches = re.finditer(r'output\s*:\s*(\w+)(?:\s*\(([^)]*)\))?', kernel_content)
        for output_match in output_matches:
            param_type = output_match.group(1)
            params_dict = {}
            if output_match.group(2):
                # Parse key=value pairs and store in metadata
                for kv in re.findall(r'(\w+)\s*=\s*([^,\s]+)', output_match.group(2)):
                    key = kv[0]
                    value = kv[1].strip('\'')
                    # Convert to int if it's a numeric value
                    if value.isdigit():
                        value = int(value)
                    params_dict[key] = value
            
            # Create parameter with metadata
            param = KernelParameter(
                name=f"output_{len([p for p in params if p.direction == 'out'])}",
                type=param_type,
                direction='out',
                is_const=False,
                is_pointer=True,
                metadata=params_dict
            )
            params.append(param)
        
        # Parse block_size (supports both [x,y,z] and x,y,z formats)
        block_match = re.search(r'block_size\s*[:=]\s*(?:[\[\(]\s*)?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:[\]\)]|(?!\S))', kernel_content)
        if block_match:
            block_size = (
                int(block_match.group(1)),
                int(block_match.group(2)),
                int(block_match.group(3))
            )
        
        # Parse grid_size if specified
        grid_match = re.search(r'grid_size\s*[:=]\s*[\[\(]\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*[\]\)]', kernel_content)
        if grid_match:
            grid_size = (
                int(grid_match.group(1)),
                int(grid_match.group(2)),
                int(grid_match.group(3))
            )
        
        # Parse shared memory
        shared_mem_match = re.search(r'shared_mem_bytes\s*[:=]\s*(\d+)', kernel_content)
        if shared_mem_match:
            shared_mem_bytes = int(shared_mem_match.group(1))
        
        # Parse Thrust usage
        use_thrust = 'use_thrust' in kernel_content and ('true' in kernel_content.lower() or 'yes' in kernel_content.lower())
        
        # Parse code block (triple-quoted string)
        code_match = re.search(r'code\s*[:=]\s*\"\"\"([\s\S]*?)\"\"\"', kernel_content)
        if code_match:
            code = code_match.group(1).strip()
        
        # Parse includes
        includes = re.findall(r'include\s+[<"]([^>"]+)[>"]', kernel_content)
        
        # Parse defines
        define_matches = re.finditer(r'define\s+(\w+)(?:\s+(.*?))?$', kernel_content, re.MULTILINE)
        for match in define_matches:
            defines[match.group(1)] = match.group(2) or ''
        
        # Create and return the kernel config
        return CudaKernelConfig(
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
    except Exception as e:
        print(f"Error parsing CUDA kernel '{kernel_name}': {e}")
        import traceback
        traceback.print_exc()
        return None

def parse_node(node_name: str, node_content: str) -> Optional[NodeConfig]:
    """Parse a single node configuration.
    
    Args:
        node_name: Name of the node
        node_content: Content of the node configuration
        
    Returns:
        NodeConfig if successful, None otherwise
    """
    try:
        node = NodeConfig(name=node_name)
        
        # Parse methods if any
        methods_match = re.search(r'methods\s*=\s*\[([\s\S]*?)\]', node_content)
        if methods_match:
            methods_content = methods_match.group(1).strip()
            # Parse each method in the methods array
            method_pattern = r'{\s*name\s*=\s*"([^"]+)"\s*,\s*return_type\s*=\s*"([^"]+)"\s*,\s*parameters\s*=\s*\[([^\]]*)\]\s*,\s*implementation\s*=\s*"""([^"]*)"""\s*}'
            method_matches = re.finditer(method_pattern, methods_content, re.DOTALL)
            
            for match in method_matches:
                name = match.group(1).strip()
                return_type = match.group(2).strip()
                params_str = match.group(3).strip()
                implementation = match.group(4).strip()
                
                # Parse parameters - handle both quoted and unquoted parameters
                params = []
                if params_str:
                    # Split by comma but respect quoted strings
                    current_param = ""
                    in_quotes = False
                    for char in params_str:
                        if char == '"':
                            in_quotes = not in_quotes
                            current_param += char
                        elif char == ',' and not in_quotes:
                            if current_param.strip():
                                params.append(current_param.strip())
                            current_param = ""
                        else:
                            current_param += char
                    
                    # Add the last parameter
                    if current_param.strip():
                        params.append(current_param.strip())
                
                # Create and add the method
                method = CppMethodConfig(
                    name=name,
                    return_type=return_type,
                    parameters=params,
                    implementation=implementation
                )
                node.methods.append(method)
        
        # Parse publishers (format: publisher /topic type)
        publisher_matches = re.finditer(r'publisher\s+([^\s]+)\s+([^\s\n]+)', node_content)
        for match in publisher_matches:
            topic = match.group(1).strip()
            msg_type = match.group(2).strip()
            node.publishers.append(PublisherConfig(topic=topic, msg_type=msg_type))
        
        # Parse subscribers (format: subscriber /topic type)
        subscriber_matches = re.finditer(r'subscriber\s+([^\s]+)\s+([^\s\n]+)', node_content)
        for match in subscriber_matches:
            topic = match.group(1).strip()
            msg_type = match.group(2).strip()
            node.subscribers.append(SubscriberConfig(topic=topic, msg_type=msg_type))
        
        # Parse services (format: service /service_name type)
        service_matches = re.finditer(r'service\s+([^\s]+)\s+([^\s\n]+)', node_content)
        for match in service_matches:
            service = match.group(1).strip()
            srv_type = match.group(2).strip()
            node.services.append(ServiceConfig(service=service, srv_type=srv_type))
        
        # Parse methods array if present
        print("\n=== DEBUG: Looking for methods in node content ===")
        methods_match = re.search(r'methods\s*=\s*\[([\s\S]*?)\](?:\s*;)?', node_content, re.DOTALL)
        
        if methods_match:
            print("=== DEBUG: Found methods array ===")
            methods_content = methods_match.group(1).strip()
            print(f"Methods content: {methods_content[:200]}...")  # Print first 200 chars
            
            # Initialize variables for parsing
            method_blocks = []
            current_block = []
            brace_level = 0
            in_quotes = False
            in_single_quotes = False
            escape = False
            i = 0
            
            # Process each character in the methods content
            while i < len(methods_content):
                char = methods_content[i]
                
                # Handle escape sequences
                if char == '\\' and not escape:
                    escape = True
                    current_block.append(char)
                    i += 1
                    continue
                    
                # Handle string literals
                if char == '"' and not escape and not in_single_quotes:
                    in_quotes = not in_quotes
                elif char == '\'' and not escape and not in_quotes:
                    in_single_quotes = not in_single_quotes
                    
                # Handle opening brace
                if char == '{' and not in_quotes and not in_single_quotes and not escape:
                    if brace_level == 0:  # Start of a new method block
                        current_block = []
                    else:
                        current_block.append(char)
                    brace_level += 1
                # Handle closing brace
                elif char == '}' and not in_quotes and not in_single_quotes and not escape:
                    brace_level -= 1
                    if brace_level == 0:  # End of a method block
                        method_blocks.append(''.join(current_block).strip())
                        current_block = []
                        i += 1
                        # Skip any trailing commas or whitespace after the block
                        while i < len(methods_content) and methods_content[i] in ' ,\t\n\r':
                            i += 1
                        continue
                    else:
                        current_block.append(char)
                # Add character to current block if inside a method
                elif brace_level > 0:
                    current_block.append(char)
                
                i += 1
                escape = False
            
            print(f"Found {len(method_blocks)} method blocks after parsing")
            
            # Process each method block
            for i, method_block in enumerate(method_blocks, 1):
                print(f"\n=== DEBUG: Parsing method block {i} ===")
                print(f"Method block: {method_block[:200]}...")  # Print first 200 chars
                
                # Initialize method properties
                name = ""
                return_type = "void"
                params = []
                implementation = ""
                
                # Extract method name - handle both quoted and unquoted
                name_match = re.search(r'name\s*=\s*"([^"]+)"', method_block) or \
                              re.search(r"name\s*=\s*'([^']+)'", method_block) or \
                              re.search(r'name\s*=\s*(\S+)', method_block)
                
                if name_match:
                    name = name_match.group(1).strip('\"\'')
                    print(f"Found method name: {name}")
                else:
                    print("WARNING: Could not find method name in block")
                    continue
                
                # Extract return type - handle both quoted and unquoted
                return_type_match = re.search(r'return_type\s*=\s*"([^"]+)"', method_block) or \
                                           re.search(r"return_type\s*=\s*'([^']+)'", method_block) or \
                                           re.search(r'return_type\s*=\s*(\S+)', method_block)
                
                if return_type_match:
                    return_type = return_type_match.group(1).strip('\"\'')
                    print(f"Found return type: {return_type}")
                
                # Extract parameters array
                params_match = re.search(r'parameters\s*=\s*\[([^\]]*)\]', method_block, re.DOTALL)
                if params_match:
                    params_str = params_match.group(1).strip()
                    print(f"Parameters string: {params_str}")
                    
                    # Parse parameters, handling both quoted and unquoted values
                    in_quotes = False
                    in_single_quotes = False
                    escape = False
                    current_param = []
                    param_parts = []
                    
                    for char in params_str:
                        # Handle escape sequences
                        if char == '\\' and not escape:
                            escape = True
                            current_param.append(char)
                            continue
                            
                        # Handle string literals
                        if char == '"' and not escape and not in_single_quotes:
                            in_quotes = not in_quotes
                        elif char == '\'' and not escape and not in_quotes:
                            in_single_quotes = not in_single_quotes
                        
                        # Handle parameter separator
                        if char == ',' and not in_quotes and not in_single_quotes and not escape:
                            param_str = ''.join(current_param).strip()
                            if param_str:
                                param_parts.append(param_str.strip(' \t\n\r\f\v\"\''))
                            current_param = []
                            continue
                            
                        current_param.append(char)
                        escape = False
                    
                    # Add the last parameter if exists
                    if current_param:
                        param_str = ''.join(current_param).strip()
                        if param_str:
                            param_parts.append(param_str.strip(' \t\n\r\f\v\"\''))
                    
                    params = param_parts
                    print(f"Found parameters: {params}")
                
                # Extract implementation (code block)
                impl_match = re.search(r'implementation\s*=\s*`([^`]*)`', method_block, re.DOTALL)
                if not impl_match:
                    impl_match = re.search(r'implementation\s*=\s*\{([^}]*)\}', method_block, re.DOTALL)
                
                if impl_match:
                    implementation = impl_match.group(1).strip()
                    print(f"Found implementation: {implementation[:100]}...")  # Print first 100 chars
                else:
                    print("WARNING: Could not find implementation for method")
                
                # Create and add the method configuration
                method_config = CppMethodConfig(
                    name=name,
                    return_type=return_type,
                    parameters=params,
                    implementation=implementation
                )
                node.methods.append(method_config)
                print(f"Added method: {name} with {len(params)} parameters")
        else:
            print("WARNING: No methods array found in node content")
        
        # Parse parameters (handle both 'parameter name = value' and 'parameter name: value' formats)
        param_matches = re.finditer(r'parameter\s+(\w+)\s*[:=]\s*([^\n;]+)', node_content)
        for match in param_matches:
            name = match.group(1).strip()
            value = match.group(2).strip()
            
            # Remove any trailing semicolon or comma
            if value.endswith(';') or value.endswith(','):
                value = value[:-1].strip()
            
            # Handle different value types
            if value.lower() == 'true':
                value = True
                param_type = 'bool'
            elif value.lower() == 'false':
                value = False
                param_type = 'bool'
            elif value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
                value = int(value)
                param_type = 'int'
            elif (value.replace('.', '', 1).isdigit() and value.count('.') < 2) or \
                 (value.startswith('-') and value[1:].replace('.', '', 1).isdigit() and value[1:].count('.') < 2):
                value = float(value)
                param_type = 'double'
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
                param_type = 'string'
            else:
                # If we can't determine the type, use string
                param_type = 'string'
            
            # Check if this parameter already exists (from a previous format)
            existing_param = next((p for p in node.parameters if p.name == name), None)
            if existing_param:
                # Update the existing parameter
                existing_param.type = param_type
                existing_param.default = value
            else:
                # Add a new parameter
                node.parameters.append(ParameterConfig(
                    name=name,
                    type=param_type,
                    default=value
                ))
        
        return node
    except Exception as e:
        print(f"Error parsing node '{node_name}': {e}")
        import traceback
        traceback.print_exc()
        return None
