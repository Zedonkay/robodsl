"""Parser for the RoboDSL configuration files."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re

@dataclass
class NodeConfig:
    """Configuration for a ROS2 node."""
    name: str
    publishers: List[Dict[str, str]] = field(default_factory=list)
    subscribers: List[Dict[str, str]] = field(default_factory=list)
    services: List[Dict[str, str]] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CudaKernelConfig:
    """Configuration for a CUDA kernel."""
    name: str
    inputs: List[Dict[str, str]] = field(default_factory=list)
    outputs: List[Dict[str, str]] = field(default_factory=list)
    block_size: tuple = (32, 1, 1)  # Default block size

@dataclass
class RoboDSLConfig:
    """Top-level configuration for a RoboDSL project."""
    nodes: List[NodeConfig] = field(default_factory=list)
    cuda_kernels: List[CudaKernelConfig] = field(default_factory=list)

def parse_robodsl(content: str) -> RoboDSLConfig:
    """Parse a RoboDSL configuration file.
    
    Args:
        content: The content of the .robodsl file
        
    Returns:
        RoboDSLConfig: The parsed configuration
    """
    config = RoboDSLConfig()
    
    # Remove comments
    content = re.sub(r'#.*', '', content)
    
    # Parse node configurations
    node_pattern = r'node\s+(\w+)\s*\{([^}]*)\}'
    for match in re.finditer(node_pattern, content, re.DOTALL):
        node_name = match.group(1)
        node_content = match.group(2).strip()
        node = NodeConfig(name=node_name)
        
        # Parse publishers
        pub_matches = re.finditer(r'publisher\s+([\w/]+)\s+([\w/]+)', node_content)
        for pub in pub_matches:
            node.publishers.append({
                'topic': pub.group(1),
                'msg_type': pub.group(2)
            })
            
        # Parse subscribers
        sub_matches = re.finditer(r'subscriber\s+([\w/]+)\s+([\w/]+)', node_content)
        for sub in sub_matches:
            node.subscribers.append({
                'topic': sub.group(1),
                'msg_type': sub.group(2)
            })
            
        config.nodes.append(node)
    
    # Parse CUDA kernels
    kernel_pattern = r'kernel\s+(\w+)\s*\{([^}]*)\}'
    for match in re.finditer(kernel_pattern, content, re.DOTALL):
        kernel_name = match.group(1)
        kernel_content = match.group(2).strip()
        kernel = CudaKernelConfig(name=kernel_name)
        
        # Parse inputs and outputs
        io_pattern = r'(input|output):\s*(\w+)(?:\s*\(\s*(.*)\s*\))?'
        for io_match in re.finditer(io_pattern, kernel_content):
            io_type = io_match.group(1)
            data_type = io_match.group(2)
            params = {}
            
            # Parse parameters if they exist
            if io_match.group(3):
                params_str = io_match.group(3).strip()
                for param in re.finditer(r'(\w+)\s*[:=]\s*([^,]+)', params_str):
                    params[param.group(1)] = param.group(2).strip()
            
            if io_type == 'input':
                kernel.inputs.append({'type': data_type, **params})
            else:
                kernel.outputs.append({'type': data_type, **params})
        
        # Parse block size if specified - handle multiple formats:
        # block_size: (x, y, z)
        # block_size: [x, y, z]
        # block_size: x, y, z
        block_size_match = re.search(
            r'block_size\s*[:=]\s*'  # block_size: or block_size=
            r'[\(\[\s]*'  # Optional opening ( or [ or whitespace
            r'(\d+)\s*[,\s]+'  # First number
            r'(\d+)\s*[,\s]+'  # Second number
            r'(\d+)'  # Third number
            r'[\s\]\)]*',  # Optional closing ) or ] or whitespace
            kernel_content
        )
        if block_size_match:
            kernel.block_size = (
                int(block_size_match.group(1)),
                int(block_size_match.group(2)),
                int(block_size_match.group(3))
            )
            
        config.cuda_kernels.append(kernel)
    
    return config
