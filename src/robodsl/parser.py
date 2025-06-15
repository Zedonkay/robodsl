"""Parser for the RoboDSL configuration files."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re

@dataclass
class NodeConfig:
    """Configuration for a ROS 2 node (extended)."""
    name: str

    # Messaging primitives
    publishers: List[Dict[str, str]] = field(default_factory=list)
    subscribers: List[Dict[str, str]] = field(default_factory=list)
    services: List[Dict[str, str]] = field(default_factory=list)
    actions: List[Dict[str, str]] = field(default_factory=list)

    # Parameters & QoS
    parameters: Dict[str, Any] = field(default_factory=dict)
    qos_profiles: Dict[str, Any] = field(default_factory=dict)

    # Execution model / lifecycle
    lifecycle: bool = False
    timers: List[Dict[str, Any]] = field(default_factory=list)
    namespace: str = ""
    remap: Dict[str, str] = field(default_factory=dict)
    
    # Parameter callback handling
    parameter_callbacks: bool = False

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
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)
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
        
        # Parse publishers
        pub_matches = re.finditer(r'publisher\s+([\w/]+)\s+([\w/]+)(?:\s+qos\s+([^\n]+))?', node_content)
        for pub in pub_matches:
            qos_options = {}
            if pub.lastindex and pub.group(3):
                # Parse qos options key:value
                for kv in pub.group(3).strip().split():
                    if ':' in kv:
                        k, v = kv.split(':', 1)
                        qos_options[k] = v
            node.publishers.append({
                'topic': pub.group(1),
                'msg_type': pub.group(2),
                'qos': qos_options
            })
            
        # Parse subscribers
        sub_matches = re.finditer(r'subscriber\s+([\w/]+)\s+([\w/]+)(?:\s+qos\s+([^\n]+))?', node_content)
        for sub in sub_matches:
            qos_options = {}
            if sub.lastindex and sub.group(3):
                for kv in sub.group(3).strip().split():
                    if ':' in kv:
                        k, v = kv.split(':', 1)
                        qos_options[k] = v
            node.subscribers.append({
                'topic': sub.group(1),
                'msg_type': sub.group(2),
                'qos': qos_options
            })
            
        # Parse services
        srv_matches = re.finditer(r'service\s+([\w/]+)\s+([\w/]+)', node_content)
        for srv in srv_matches:
            node.services.append({
                'service': srv.group(1),
                'srv_type': srv.group(2)
            })

        # Parse parameters
        param_matches = re.finditer(r'parameter\s+(\w+)\s*[:=]\s*([^\n]+)', node_content)
        for param in param_matches:
            value = param.group(2).strip()
            node.parameters[param.group(1)] = value

        # Lifecycle flag
        if re.search(r'\blifecycle\b', node_content):
            node.lifecycle = True

        # Parameter callbacks flag
        if re.search(r'\bparameter_callbacks\b', node_content):
            node.parameter_callbacks = True

        # Timers:   timer <name> <period_ms>
        for tmr in re.finditer(r'timer\s+(\w+)\s+(\d+)', node_content):
            node.timers.append({
                'name': tmr.group(1),
                'period_ms': int(tmr.group(2))
            })

        # Actions: action <name> <goal_type> <result_type> <feedback_type>
        for act in re.finditer(r'action\s+(\w+)\s+([\w/]+)\s+([\w/]+)\s+([\w/]+)', node_content):
            node.actions.append({
                'name': act.group(1),
                'goal_type': act.group(2),
                'result_type': act.group(3),
                'feedback_type': act.group(4)
            })

        # Namespace
        ns_match = re.search(r'namespace\s+([/\w]+)', node_content)
        if ns_match:
            node.namespace = ns_match.group(1)

        # Remap rules: remap <from> <to>
        for rm in re.finditer(r'remap\s+([/\w]+)\s+([/\w]+)', node_content):
            node.remap[rm.group(1)] = rm.group(2)

        config.nodes.append(node)
    
    # Parse CUDA kernel configurations
    kernel_pattern = r'kernel\s+(\w+)\s*\{([^}]*?)(?=^\s*\w|\Z)'
    for match in re.finditer(kernel_pattern, content, re.DOTALL | re.MULTILINE):
        kernel_name = match.group(1)
        kernel_content = match.group(2).strip()
        kernel = CudaKernelConfig(name=kernel_name)
        
        # Parse inputs and outputs
        input_pattern = r'input\s+([\w:]+)\s+(\w+)'
        for input_match in re.finditer(input_pattern, kernel_content):
            kernel.inputs.append({'type': input_match.group(1), 'name': input_match.group(2)})
            
        output_pattern = r'output\s+([\w:]+)\s+(\w+)'
        for output_match in re.finditer(output_pattern, kernel_content):
            kernel.outputs.append({'type': output_match.group(1), 'name': output_match.group(2)})
            
        # Parse block size (x, y, z)
        block_size_match = re.search(
            r'block_size\s*[\[\(]?\s*'
            r'(\d+)\s*[,\s]+'
            r'(\d+)\s*[,\s]+'
            r'(\d+)',
            kernel_content
        )
        if block_size_match:
            kernel.block_size = (
                int(block_size_match.group(1)),
                int(block_size_match.group(2)),
                int(block_size_match.group(3))
            )
            
        # Parse grid size (x, y, z)
        grid_size_match = re.search(
            r'grid_size\s*[\[\(]?\s*'
            r'(\d+)\s*[,\s]+'
            r'(\d+)\s*[,\s]+'
            r'(\d+)',
            kernel_content
        )
        if grid_size_match:
            kernel.grid_size = (
                int(grid_size_match.group(1)),
                int(grid_size_match.group(2)),
                int(grid_size_match.group(3))
            )
            
        # Parse shared memory
        shared_mem_match = re.search(r'shared_memory\s+(\d+)', kernel_content)
        if shared_mem_match:
            kernel.shared_mem_bytes = int(shared_mem_match.group(1))
            
        # Check for Thrust usage
        if 'use_thrust' in kernel_content.lower():
            kernel.use_thrust = True
            kernel.includes.append('thrust/device_vector.h')
            kernel.includes.append('thrust/execution_policy.h')
            
        # Parse additional includes
        include_matches = re.finditer(r'include\s+[<"]([^>"]+)[>"]', kernel_content)
        for include_match in include_matches:
            kernel.includes.append(include_match.group(1))
            
        # Parse defines
        define_matches = re.finditer(r'define\s+(\w+)(?:\s+(.*?))?$', kernel_content, re.MULTILINE)
        for define_match in define_matches:
            kernel.defines[define_match.group(1)] = define_match.group(2) or ''
            
        # Parse kernel code block
        code_match = re.search(r'code\s*\"\"\"(.*?)\"\"\"', kernel_content, re.DOTALL)
        if code_match:
            kernel.code = code_match.group(1).strip()
            
        config.cuda_kernels.append(kernel)
    
    return config
